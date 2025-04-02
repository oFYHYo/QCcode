import numpy as np
import scipy
from functools import partial

einsum = partial(np.einsum,optimize=True)

class MPOtensor:
    '''local tensor class for MPO'''

    def __init__(self,mpo):
        '''MPO[left,phys1,phys2,right]'''
        assert isinstance(mpo, np.ndarray)
        assert mpo.ndim == 4
        assert mpo.shape[2] == mpo.shape[1]
        self.data = mpo
    
    def left(self):
        '''return the left bond dimension'''
        return self.data.shape[0]

    def phys(self):
        '''return the physical bond dimension'''
        return self.data.shape[1]

    def right(self):
        '''return the right bond dimension'''
        return self.data.shape[3]
        

class MPStensor:
    '''local tensor class for MPS'''

    def __init__(self,mps):
        '''MPS[left,phys,right]'''
        assert isinstance(mps, np.ndarray)
        assert mps.ndim == 3
        self.data = mps
    
    def left(self):
        '''return the left bond dimension'''
        return self.data.shape[0]

    def phys(self):
        '''return the physical bond dimension'''
        return self.data.shape[1]

    def right(self):
        '''return the right bond dimension'''
        return self.data.shape[2]

class DMRG:

    def __init__(self,mpo,mps=None,max_dim=25):
        
        self.length = len(mpo)
        self.max_dim = max_dim
        dummyOperator = MPOtensor(np.ones((1,1,1,1)))

        self.mpo = [dummyOperator]+mpo+[dummyOperator]
        if mps is None:
            self.mps = self.initial_mps() 
            for i in range(self.length,1,-1):
                self.right_canonicalize(i)
        else:
            self.mps = mps
        
        self.tensorF = [np.ones((1,1,1))]+[None for _ in range(self.length)]+[np.ones((1,1,1))]
        self.tensorL = self.tensorF.copy()
        self.tensorR = self.tensorF.copy()

        self.energy = None

    def initial_mps(self):
        '''
        initialize the mps
        '''
        
        dummysite = MPStensor(np.ones((1,1,1)))
        lastsite = MPStensor(np.random.rand(self.max_dim, self.mpo[-2].phys(), 1))
        firstsite = MPStensor(np.random.rand(1, self.mpo[1].phys(), self.max_dim))
        mps = [dummysite,firstsite]
        mpo = self.mpo
        for i in range(self.length-2):
            tmp = np.random.rand(self.max_dim, mpo[i+1].phys(), self.max_dim)
            tmp /= np.linalg.norm(tmp)
            mps.append(MPStensor(tmp))
        mps.append(lastsite)
        mps.append(dummysite)


        return mps

    def right_canonicalize(self, idx):
        if idx <=1:
            return
        mps = self.mps[idx]
        u,s,v = np.linalg.svd(mps.data.reshape((mps.left(), mps.right()*mps.phys())), full_matrices=False)
        if s.shape[0] > self.max_dim:
            u = u[:,:self.max_dim]
            s = s[:self.max_dim]
            v = v[:self.max_dim,:]
        mps.data = v.reshape((-1, mps.phys(), mps.right()))
        self.mps[idx-1].data = einsum('iaj,jk,k->iak',self.mps[idx-1].data, u, s)
    
    def left_canonicalize(self, idx):
        if idx >= self.length:
            return
        mps = self.mps[idx]
        u,s,v = np.linalg.svd(mps.data.reshape((mps.phys()*mps.left(), mps.right())), full_matrices=False)
        if s.shape[0] > self.max_dim:
            u = u[:,:self.max_dim]
            s = s[:self.max_dim]
            v = v[:self.max_dim,:]
        mps.data = u.reshape((mps.left(), mps.phys() ,-1))
        self.mps[idx+1].data = einsum('i,ij,jak->iak',s,v,self.mps[idx+1].data)

    def calc_tensorL(self,idx):
        '''calculate the tensor L'''
        mpo = self.mpo
        mps = self.mps
        if idx == 0:
            L = einsum('iuj,auvb,kvl->jbl',mps[idx].data.conjugate(),self.mpo[idx].data,mps[idx].data)
            
        else:
            L_left = self.calc_tensorL(idx-1)

            T_idx  = einsum('iuj,auvb,kvl->iakjbl',mps[idx].data.conjugate(),self.mpo[idx].data,mps[idx].data)
            L = einsum('iak,iakjbl->jbl',L_left,T_idx)
        self.tensorL[idx] = L    

        return L
    
    def calc_tensorR(self,idx):
        '''calculate the tensor R'''
        mpo = self.mpo
        mps = self.mps

        if idx == self.length+1:
            R = einsum('iuj,auvb,kvl->iak',mps[idx].data.conjugate(),self.mpo[idx].data,mps[idx].data)

            
        else:
            R_right = self.calc_tensorR(idx+1)
            # mpo = self.mpo[idx].data
            # mps = self.mps[idx].data
            T_idx  = einsum('iuj,auvb,kvl->iakjbl',mps[idx].data.conjugate(),self.mpo[idx].data,mps[idx].data)
            R = einsum('iakjbl,jbl->iak',T_idx,R_right)

        self.tensorR[idx] = R 


        return R

    def calc_reduce_hamitonion(self,idx,L,R):
        mpo = self.mpo
        H = einsum('iak,auvb,jbl->iujkvl',L,mpo[idx].data,R)
        local_dimension = H.shape[0]*H.shape[1]*H.shape[2]
        H = H.reshape(local_dimension,local_dimension)
        return H,local_dimension
    
    def graph(self,idx):
        print('='*(idx-1)+'*'+'-'*(self.length-idx))
    
    def kernel(self):
        mps = self.mps
        mpo = self.mpo
        old_energy = 0
        sweepCount = 0

        while True:
            sweepCount += 1
            idx = 0
            print('\n*************** sweep: %d ***************' % sweepCount)
            print(">>>>>>>>>> sweep from left to right >>>>>>>>>>")
            for i in range(self.length):
                idx += 1
                # calculate the tensor L
                L = self.calc_tensorL(idx-1)
                # calculate the tensor R
                R = self.calc_tensorR(idx+1)

                H,local_dimension = self.calc_reduce_hamitonion(idx,L,R)

                eigval, eigvec = np.linalg.eigh(H)
                energy = eigval[0]
                vec = eigvec[:,0].reshape(mps[idx].data.shape)
                mps[idx].data = vec
                self.left_canonicalize(idx)
                print("idx= %2d, dmrg_Energy= %.8f" %(idx, energy)+'  ',end='')
                self.graph(idx)
                
            idx += 1
            print("<<<<<<<<<< sweep from right to left <<<<<<<<<<")
            for i in range(self.length):
                idx -= 1
                # calculate the tensor L
                L = self.calc_tensorL(idx-1)
                # calculate the tensor R
                R = self.calc_tensorR(idx+1)

                H,local_dimension = self.calc_reduce_hamitonion(idx,L,R)

                eigval, eigvec = np.linalg.eigh(H)
                energy = eigval[0]
                vec = eigvec[:,0].reshape(mps[idx].data.shape)
                mps[idx].data = vec
                self.right_canonicalize(idx)
                print("idx= %2d, dmrg_Energy= %.8f" %(idx, energy)+'  ',end='')
                
                self.graph(idx)
                
            
            if abs(energy - old_energy) < 1e-8:
                self.energy = energy
                converge = True
                print('\n*************** DMRG converged ***************')
                break
            old_energy = energy

        return energy




def local_operator(J=1.0, Jz=1.0, h=1.0):
    """
    construct local operator for Heisenberg model
    - J:  coupling constant
    - Jz: coupling constant
    - h:  strength of external field
    return: local operator with shape (5, 5, 2, 2)
    """
    # 5x5 matrix, each element is a 2x2 matrix
    # the local operator is 4-dimensional and with shape as (5,5,2,2)
    # S^+, the aising operator
    import numpy
    Sp = numpy.float64([[0, 1],
                        [0, 0]])

    # S^-, the lowering operator
    Sm = numpy.float64([[0, 0],
                        [1, 0]])

    # Sx measurement
    Sx = numpy.float64([[0,   0.5],
                        [0.5, 0  ]])

    # Sz measurement
    Sz = numpy.float64([[0.5,  0   ],
                        [0,   -0.5]])

    # zero matrix block
    zero = numpy.zeros((2, 2))

    # identity matrix block
    identity = numpy.eye((2))
    return numpy.float64([
        [identity, zero,   zero,   zero,  zero],
        [Sp,       zero,   zero,   zero,  zero],
        [Sm,       zero,   zero,   zero,  zero],
        [Sz,       zero,   zero,   zero,  zero],
        [-h*Sz,    J/2*Sm, J/2*Sp, Jz*Sz, identity]
    ])

if __name__ == '__main__':
    onempo = local_operator().transpose(0,2,3,1)

    leftmost = onempo[-1].copy()
    leftmost = MPOtensor(leftmost.reshape((1,) + leftmost.shape))

    rightmost = onempo[:,:,:,0].copy()
    rightmost = MPOtensor(rightmost.reshape(rightmost.shape[0],rightmost.shape[1],rightmost.shape[2],1))
    mpo = [leftmost]+[MPOtensor(onempo.copy())]*10+[rightmost]

    dmrg = DMRG(mpo)
    dmrg.max_dim = 50
    mps = dmrg.mps

    dmrg.kernel()
    print(dmrg.energy)
