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

    def __init__(self,mpo,mps=None,max_dim=100):
        '''
        mpo: MPOtensor's set
        mps: MPStensor's set
        '''
        self.length = len(mpo)
        # 引入哑元，以便便计算tensorF、L、R
        dummysite = MPStensor(np.ones((1,1,1)))
        dummyOperator = MPOtensor(np.zeros((0,0,0,0)))

        self.mpo = [dummyOperator]+mpo+[dummyOperator]

        if mps is None:
            mps = self.initial_mps()
        else:
            mps = mps

        self.mps = [dummysite]+mps+[dummysite]
        assert len(mpo)==len(mps)

        self.tensorF = [np.ones((1,1,1))]*(len(mpo)+2)
        self.max_dim = max_dim

    def initial_mps(self):
        '''Initial MPS with random numbers'''
        length = self.length
        mpo = self.mpo
        mps = []
        for i in range(length):
            mps.append(MPStensor(np.random.rand(
                mpo[i+1].left(), mpo[i+1].phys(), mpo[i+1].right())))
        return mps

    def calc_tensorF(self,idx,direction):

        '''calculate the tensor F'''

        if idx > 0 or idx < self.length+1:
            mpo = self.mpo
            mps = self.mps
            if direction == 'left':
                old_F = self.calc_tensorF(idx-1,'left')
                F = einsum('iuj,auvb,iak,kvl->jbl',
                        mps[idx].data.conjugate(),mpo[idx].data,old_F,mps[idx].data)
            else:
                old_F = self.calc_tensorF(idx+1,'right')
                F = einsum('iuj,auvb,jbl,kvl->iak',
                        mps[idx].data,mpo[idx].data,old_F,mps[idx].data.conjugate())
        else:
            F = np.ones((1,1,1))
        self.tensorF[idx] = F

        return self.tensorF[idx]
    
    def contract_left(self,idx):
        '''contract the left tensor'''
        
        return self.calc_tensorF(idx,'left')
    
    def contract_right(self,idx):
        '''contract the right tensor'''

        return self.calc_tensorF(idx,'right')

    def calc_reduce_hamitonian(self,idx,L,R):
        
        mpo = self.mpo
        H = einsum('iak,auvb,jbl->ikuvjl',L,mpo[idx].data,R)
        H=H.reshape(H.shape[0]*H.shape[1]*H.shape[2],H.shape[3]*H.shape[4]*H.shape[5])
        return H
    
    def update_mps(self,eigvec,idx,max_dim,direction='left'):
        '''update MPS with SVD'''
        mps = self.mps
        vec = eigvec.reshape(-1, mps[idx].phys(), mps[idx].right())
        
        if direction == 'right':
            u,s,v = np.linalg.svd(vec.reshape(-1,vec.shape[-1]),full_matrices=False)
            dim = min(s.shape[0],max_dim)
            u = u[:,:dim]
            s = s[:dim]
            v = v[:dim,:]

            mps[idx].data = u.reshape(mps[idx].left(), mps[idx].phys(), -1)

            mps[idx+1].data = einsum('i,ij,jal->jal',s,v,mps[idx+1].data)

        elif direction == 'left':
            u,s,v = np.linalg.svd(vec.reshape(vec.shape[0],-1),full_matrices=False)
            dim = min(s.shape[0],max_dim)
            u = u[:,:dim]
            s = s[:dim]
            v = v[:dim,:]

            mps[idx].data = v.reshape(-1, mps[idx].phys(), mps[idx].right())

            mps[idx-1].data = einsum('iaj,jk,k->iaj',mps[idx-1].data,u,s)
            
        else:
            raise ValueError("direction must be 'left' or 'right'")
        
    def kernel(self):

        mps = self.mps
        mpo = self.mpo
        energy_old = 0
        while True:
            idx=1
            for i in range(len(mps)-2):
                L = self.contract_left(idx)
                R = self.contract_right(idx)
                H = self.calc_reduce_hamitonian(idx,L,R)
                eigval,eigvec = np.linalg.eigh(H)
                eigvec = eigvec[:,0]

                self.update_mps(eigvec,idx,max_dim=self.max_dim,direction='right')
                idx += 1

            for i in range(len(mps)-2):
                L = self.contract_left(idx)
                R = self.contract_right(idx)
                H = self.calc_reduce_hamitonian(idx,L,R)
                eigval,eigvec = np.linalg.eigh(H)
                eigvec = eigvec[:,0]

                self.update_mps(eigvec,idx,max_dim=self.max_dim,direction='left')
                idx -= 1
           
            
            energy = eigval[0]
            if abs(energy-energy_old) < 1e-8:
                print(energy)
                break

            energy_old = energy

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

if __name__ == "__main__":
    onempo = local_operator().transpose(0,2,3,1)
    leftmost = onempo[-1].copy()
    print(onempo.shape)
    leftmost = MPOtensor(leftmost.reshape((1,) + leftmost.shape))
    rightmost = onempo[:,:,:,0].copy()
    print(rightmost.shape)
    rightmost = MPOtensor(rightmost.reshape(rightmost.shape[0],rightmost.shape[1],rightmost.shape[2],1))
    mpo = [leftmost]+[MPOtensor(onempo.copy())]*8+[rightmost]

    dmrg = DMRG(mpo)
    dmrg.kernel()