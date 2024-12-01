import numpy as np
from pyscf import gto,dft
from scipy.linalg import expm,sqrtm
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
        self.mo_energy = mf.mo_energy
        self.nocc = self.mol.nelec[0]

    def gen_magnus_propagator(self,fock_ao,Sinv_2,S_2,timestep):
        '''生成magnus传播器'''

        timestep = self.timestep
        exp = expm(-1j*np.dot(np.dot(Sinv_2,fock_ao),Sinv_2)*timestep)
        propagator = np.dot(Sinv_2,np.dot(exp,S_2))

        return propagator
    
    def gen_new_dm(self,dm,propagator):
        '''生成新的密度矩阵'''

        new_dm = np.dot(propagator,np.dot(dm,np.conjugate(propagator.T)))

        return new_dm
    
    def fock_linear_extra(self,f_0,f_1):
        '''生成两个fock矩阵之间从线性插值，即F(t+dt/4)'''

        f_extra = 1.75*f_1-0.75*f_0

        return f_extra
    
    def get_extenal_field(self,time):
        '''计算外加场大小'''
        return 
    
    def get_dipole(self,u_ao,dm):
        '''计算偶极矩'''

        return np.einsum('xuv,uv->x',u_ao,dm)

    def propagate(self):
        '''对密度矩阵进行实时传播
        '''
        
        nocc = self.mol.nelec[0]
        timestep = self.timestep
        get_fock = self.mf.get_fock

        u_ao = self.mol.intor_symmetric('int1e_r', comp=3)
        S = mol.intor('int1e_ovlp')
        S_2 = sqrtm(S)
        Sinv_2 = np.linalg.inv(S_2)
        dm = 2*np.dot(self.mo_coeff[:,:nocc],self.mo_coeff[:,:nocc].T)
        
        f0 = get_fock(dm)
        f1 = f0.copy()

        Time = np.zeros(self.maxstep+1)
        Density_matrix = np.zeros([self.maxstep+1,dm.shape[0],dm.shape[1]],dtype=np.complex128)
        Energy = np.zeros_like(Time)
        Dipole = np.zeros([self.maxstep+1,3])

        Energy[0] = self.mf.energy_tot(dm)
        Dipole[0] = self.get_dipole(u_ao,dm)
        Density_matrix[0] = dm

        print('   Step          Time         Energy          time_cost                   Dipole')
        for i in range(self.maxstep):
            x = time()

            fock_4dt = self.fock_linear_extra(f0,f1)

            propagator_2dt = self.gen_magnus_propagator(fock_4dt,Sinv_2,S_2,timestep/2)
            dm_2dt = self.gen_new_dm(dm,propagator_2dt)
            fock_2dt = get_fock(dm_2dt)

            propagator_dt = self.gen_magnus_propagator(fock_2dt,Sinv_2,S_2,timestep)
            dm_dt = self.gen_new_dm(dm,propagator_dt)
            fock_dt = get_fock(dm_dt)

            f0 = f1.copy()
            f1 = fock_dt
            dm = dm_dt

            Density_matrix[i+1] += dm
            Time[i+1] += (i+1)*timestep
            Energy[i+1] += self.mf.energy_tot(dm.real)
            Dipole[i+1] += self.get_dipole(u_ao,dm.real)
            
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



        '''
        grids =  dft.gen_grid.Grids(self.mol)
        grids.build()
        coords = grids.coords
        ni = dft.numint.NumInt()
        ao_0 = ni.eval_ao(self.mol,coords,deriv=0)
        rho = np.einsum('tuv,ku,kv->tk',self.dm,ao_0,ao_0)

        fig, ax = plt.subplots()
        x = []
        y = []
        tmp = []
        for i in range(rho.shape[0]):
            if coords[i,0] == 0 and coords[i,1] == 0 and -2<coords[i,2]<2:
                x.append(coords[i,2])
                y.append(rho[])

        temp = ax.plot(coords[],)
        tmp.append(temp)
        
        ani = animation.ArtistAnimation(fig, tmp, interval=200, repeat_delay=1000)
        ani.save("fig.gif", writer='pillow')
        '''







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

    mf = dft.RKS(mol)
    
    Func = 'PBE'
    mf.xc = Func
    mf.kernel()
    nocc = mol.nelec[0]
    dm = mf.make_rdm1()
    mo_c = mf.mo_coeff
    print()
    #print(np.diag(np.einsum('iu,uv,vj->ij',np.linalg.inv(mo_c),dm,np.linalg.inv(mo_c.T))))

    rt_td = RT_TDDFT(mf)
    rt_td.maxstep=100
    rt_td.propagate()
    A = np.einsum('iu,uv,vj->ij',np.linalg.inv(mo_c),rt_td.dm[0],np.linalg.inv(mo_c.T))
    dm1 = 2*mo_c[:,:nocc].T@mo_c[:,:nocc]
    print(np.where(dm1>1e-8,dm1,0))
    #rt_td.plot()
