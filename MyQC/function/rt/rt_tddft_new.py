import numpy as np
from pyscf import gto,dft,scf
from scipy.linalg import expm,sqrtm,eigh
from time import time
'''
ref: J. Chem. Phy. 2004, 121, 8, 3425-3433.
     Phy. Rev. B 2006, 74, 15, 155112.
'''
class RT_TDDFT:
    

    def __init__(self,mf):

        self.mf = mf
        self.mol = mf.mol
        self.timestep = 0.01
        self.maxstep = 500
        self.mo_coeff = mf.mo_coeff
        self.mo_coeff_inv = np.linalg.inv(self.mo_coeff)
        self.mo_energy = mf.mo_energy
        self.nocc = self.mol.nelec[0]

    def fock_ao2ort(self,fock,type='ao'):

        if type == 'ao':
            X = self.X
            fock_ao = fock
            fock_ort = np.dot(X.T,np.dot(fock_ao,X))
            return fock_ort
        
        elif type == 'ort':
            Xinv = self.Xinv
            fock_ort = fock
            fock_ao = np.einsum('ui,ij,jv->uv',Xinv.T,fock_ort,Xinv)

            return fock_ao

        else:
            raise

    def dm_ao2ort(self,dm,type='ao'):

        if type == 'ao':
            Y = self.Y
            dm_ao = dm
            dm_ort =np.einsum('iu,uv,vj->ij',Y.T,dm_ao,Y)
            return dm_ort
        
        elif type == 'ort':
            Yinv = self.Yinv
            dm_ort = dm
            dm_ao = np.einsum('ui,ij,jv->uv',Yinv.T,dm_ort,Yinv)

            return dm_ao

        else:
            raise

    def gen_magnus_propagator(self,fock_ao,timestep):
        '''生成magnus传播器'''

        #timestep = self.timestep
        fock_ort = self.fock_ao2ort(fock_ao,'ao')
        
        exp = expm(-1j*fock_ort*timestep)
        propagator = exp

        return propagator
    
    def gen_new_dm(self,dm_ao,propagator):
        '''生成新的密度矩阵'''
        dm_ort = self.dm_ao2ort(dm_ao,'ao')

        new_dm_ort = np.dot(propagator,np.dot(dm_ort,np.conjugate(propagator.T)))
        new_dm_ao = self.dm_ao2ort(new_dm_ort,'ort')
        return new_dm_ao
    
    def fock_linear_extra(self,f_0,f_1):
        '''生成两个fock矩阵之间从线性插值，即F(t+dt/4)'''

        f_extra = 1.75*f_1-0.75*f_0

        return f_extra
    
    def get_extenal_field(self,time):
        '''计算外加场大小'''
        return 
    
    def get_dipole(self,u_ao,dm_ao):
        '''计算偶极矩'''

        return np.einsum('xuv,uv->x',u_ao,dm_ao)

    def get_fock(self,dm_mo):
        dm_ao = self.dm_ao2ort(dm_mo,'ort')
        fock_ao = self.mf.get_fock(dm_ao)
        fock_mo = self.fock_ao2mo(fock_ao,'ao')
        return fock_mo


    def propagate(self):
        '''对密度矩阵进行实时传播
        '''
        
        nocc = self.mol.nelec[0]
        timestep = self.timestep

        u_ao = self.mol.intor_symmetric('int1e_r', comp=3)
        dm_ao = 2*np.dot(self.mo_coeff[:,:nocc],self.mo_coeff[:,:nocc].T)

        S = self.mol.intor('int1e_ovlp')
        s,U = np.linalg.eigh(S)
        self.X = np.dot(U,np.diag(s**-0.5))
        self.Xinv = np.linalg.inv(self.X)
        self.Y = np.dot(self.X,np.diag(s))
        self.Yinv = np.linalg.inv(self.Y)
        
        f0_ao = self.mf.get_fock(dm=dm_ao)
        f1_ao = f0_ao.copy()

        Time = np.zeros(self.maxstep+1)
        Density_matrix = np.zeros([self.maxstep+1,dm_ao.shape[0],dm_ao.shape[1]],dtype=np.complex128)
        Energy = np.zeros_like(Time)
        Dipole = np.zeros([self.maxstep+1,3])

        Energy[0] = self.mf.energy_tot(dm_ao)
        Dipole[0] = self.get_dipole(u_ao,dm_ao)
        Density_matrix[0] = dm_ao

        print('   Step          Time         Energy          time_cost                   Dipole')
        for i in range(self.maxstep):
            x = time()

            fock_ao_4dt = self.fock_linear_extra(f0_ao,f1_ao)
            
            propagator_2dt = self.gen_magnus_propagator(fock_ao_4dt,timestep/2)
            #print(propagator_2dt)
            dm_ao_2dt = self.gen_new_dm(dm_ao,propagator_2dt)
            fock_ao_2dt = self.mf.get_fock(dm_ao_2dt)

            propagator_dt = self.gen_magnus_propagator(fock_ao_2dt,timestep)
            dm_ao_dt = self.gen_new_dm(dm_ao,propagator_dt)
            fock_ao_dt = self.mf.get_fock(dm_ao_dt)

            f0_ao = f1_ao.copy()
            f1_ao = fock_ao_dt
            dm_ao = dm_ao_dt

            Density_matrix[i+1] += dm_ao
            Time[i+1] += (i+1)*timestep
            Energy[i+1] += self.mf.energy_tot(dm_ao)
            Dipole[i+1] += self.get_dipole(u_ao,dm_ao.real)
            
            y = time()
            print(f'{i+1:6d}         {Time[i+1]:6.2f}       {Energy[i+1]:12.6f}         {y-x:.2f}           {Dipole[i+1,0]:12.6f}{Dipole[i+1,1]:12.6f}{Dipole[i+1,2]:12.6f}')

        self.time = Time
        self.e = Energy
        self.dm = Density_matrix
        self.dipole = Dipole
        return Time,Density_matrix,Energy,Dipole
    
    def plot(self):
        '''绘制各物理量的时间演化'''
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        plt.plot(self.time,self.e)
        plt.xlabel('time')
        plt.ylabel('Energy')
        plt.title('RT-TDDFT')
        plt.show()



if __name__ == '__main__':
    mol = gto.M()
    mol.atom = ''' 
    C                  0.00000000    0.00000000   -0.59750000
    C                  0.00000000    0.00000000    0.59750000
    H                  0.00000000    0.00000000   -1.65850000
    H                  0.00000000    0.00000000    1.65850000
    '''
    mol.basis = '6-31G*'
    mol.build()

    mf = scf.RKS(mol)
    
    Func = 'PBE'
    mf.xc = Func
    mf.kernel()
    rt_td = RT_TDDFT(mf)
    rt_td.maxstep=1
    rt_td.propagate()


    