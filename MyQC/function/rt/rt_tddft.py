import numpy as np
from pyscf import gto,dft,scf
from scipy.linalg import expm,sqrtm,eigh
from time import time
from math import exp,cos,sin,pi
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

    def fock_ao2mo(self,fock,type='ao'):
        mo_c = self.mo_coeff
        mo_c_inv = self.mo_coeff_inv
        if type == 'ao':
            fock_ao = fock
            fock_mo = np.einsum('iu,uv,vj->ij',mo_c.T,fock_ao,mo_c)
            return fock_mo
        
        elif type == 'mo':
            fock_mo = fock
            fock_ao = np.einsum('ui,ij,jv->uv',mo_c_inv.T,fock_mo,mo_c_inv)

            return fock_ao

        else:
            raise

    def dm_ao2mo(self,dm,type='ao'):
        mo_c = self.mo_coeff
        mo_c_inv = self.mo_coeff_inv
        if type == 'ao':
            dm_ao = dm
            dm_mo =np.einsum('iu,uv,vj->ij',mo_c_inv,dm_ao,mo_c_inv.T)
            return dm_mo
        
        elif type == 'mo':
            dm_mo = dm
            dm_ao = np.einsum('ui,ij,jv->uv',mo_c,dm_mo,mo_c.T)

            return dm_ao

        else:
            raise

    def gen_magnus_propagator(self,fock_mo,timestep):
        '''生成magnus传播器'''

        #timestep = self.timestep
        #fock_mo = self.fock_ao2mo(fock_ao,'ao')
        
        exp = expm(-1j*fock_mo*timestep)
        propagator = exp

        return propagator
    
    def gen_new_dm(self,dm_mo,propagator):
        '''生成新的密度矩阵'''

        new_dm_mo = np.dot(propagator,np.dot(dm_mo,np.conjugate(propagator.T)))

        return new_dm_mo
    
    def fock_linear_extra(self,f_0,f_1):
        '''生成两个fock矩阵之间从线性插值，即F(t+dt/4)'''

        f_extra = 1.75*f_1-0.75*f_0

        return f_extra
    
    def get_extenal_field(self,time,type='Gaussian'):
        '''计算外加场大小'''
        if type == 'Gaussian':
            field = 2*1e-5*exp(-(time-3)**2/(2*0.2**2))
            #*cos(time*2*pi/1)

        return np.array([field,0,0])
    
    def get_dipole(self,u_ao,dm_ao):
        '''计算偶极矩'''

        return np.einsum('xuv,uv->x',u_ao,dm_ao)

    def get_fock_mo(self,dm_mo,external_field=None):
        

        dm_ao = self.dm_ao2mo(dm_mo,'mo')
        fock_ao = self.mf.get_fock(dm=dm_ao)
        if external_field is not None:
            fock_ao += np.einsum('xuv,x',self.u_ao,external_field)
        fock_mo = self.fock_ao2mo(fock_ao,'ao')
        return fock_mo


    def propagate(self):
        '''对密度矩阵进行实时传播
        '''
        
        nocc = self.mol.nelec[0]
        timestep = self.timestep

        u_ao = self.mol.intor_symmetric('int1e_r', comp=3)
        self.u_ao = u_ao
        dm_ao = 2*np.dot(self.mo_coeff[:,:nocc],self.mo_coeff[:,:nocc].T)
        dm_mo = self.dm_ao2mo(dm_ao,'ao')
        
        f0_mo = self.get_fock_mo(dm_mo)
        f1_mo = f0_mo.copy()

        Time = np.zeros(self.maxstep+1)
        Density_matrix = np.zeros([self.maxstep+1,dm_ao.shape[0],dm_ao.shape[1]],dtype=np.complex128)
        Energy = np.zeros_like(Time)
        Dipole = np.zeros([self.maxstep+1,3],dtype=np.complex128)
        Ext_field = np.zeros_like(Dipole)

        Energy[0] = self.mf.energy_tot(dm_ao)
        Dipole[0] = self.get_dipole(u_ao,dm_ao)
        Density_matrix[0] = dm_ao
        Ext_field[0] = 0

        print('   Step          Time         Energy          time_cost                   Dipole')
        for i in range(self.maxstep):
            x = time()

            fock_mo_4dt = self.fock_linear_extra(f0_mo,f1_mo)
            
            propagator_2dt = self.gen_magnus_propagator(fock_mo_4dt,timestep/2)
            dm_mo_2dt = self.gen_new_dm(dm_mo,propagator_2dt)
            ef_2dt = self.get_extenal_field((i+1/2)*self.timestep)
            fock_mo_2dt = self.get_fock_mo(dm_mo_2dt,external_field=ef_2dt)

            propagator_dt = self.gen_magnus_propagator(fock_mo_2dt,timestep)
            dm_mo_dt = self.gen_new_dm(dm_mo,propagator_dt)
            ef_dt = self.get_extenal_field((i+1)*self.timestep)
            fock_mo_dt = self.get_fock_mo(dm_mo_dt,external_field=ef_dt)

            f0_mo = f1_mo.copy()
            f1_mo = fock_mo_dt
            dm_mo = dm_mo_dt

            Density_matrix[i+1] += dm_mo
            Time[i+1] += (i+1)*timestep
            dm_ao = self.dm_ao2mo(dm_mo,'mo')
            Energy[i+1] += self.mf.energy_tot(dm=dm_ao)
            Dipole[i+1] += self.get_dipole(u_ao,dm_ao)
            Ext_field[i+1] = ef_dt
            
            y = time()
            print(f'{i+1:6d}         {Time[i+1]:6.2f}       {Energy[i+1]:12.6f}         {y-x:.2f}           {Dipole[i+1,0].real:12.6f}{Dipole[i+1,1].real:12.6f}{Dipole[i+1,2].real:12.6f}')

        self.time = Time
        self.e = Energy
        self.dm = Density_matrix
        self.dipole = Dipole
        self.field = Ext_field
        return Time,Density_matrix,Energy,Dipole
    
    def plot(self):
        '''绘制各物理量的时间演化'''
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        fig , ax1 = plt.subplots()
        dipole_fft = np.fft.fft(self.dipole[:,2])
        field_fft = np.fft.fft(self.field[:,2])
        freq = np.fft.fftfreq(self.time.shape[-1])
        
        strength = (dipole_fft/field_fft).imag
        #print(freq)
        ax1.plot(self.time*0.024,self.dipole[:,0].real,'b',label='energy')
        
        ax1.set_ylabel('Dipole moment(a. u.)')

        # ax2 = ax1.twinx()
        # ax2.set_ylim(-0.02,0.04)
        # ax2.plot(self.time,self.field[:,2],'r',label='field')
        # ax2.set_ylabel('External_field(a. u.)')
        ax1.set_xlabel('time(fs)')
        

        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc=0)
        
        

        
        ax1.set_title('RT-TDDFT')
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
 C                  1.20809735    0.69749533   -0.00000000
 C                  0.00000000    1.39499067   -0.00000000
 C                 -1.20809735    0.69749533   -0.00000000
 C                 -1.20809735   -0.69749533   -0.00000000
 C                  0.00000000   -1.39499067   -0.00000000
 C                  1.20809735   -0.69749533   -0.00000000
 H                  2.16038781    1.24730049   -0.00000000
 H                  0.00000000    2.49460097   -0.00000000
 H                 -2.16038781    1.24730049   -0.00000000
 H                 -2.16038781   -1.24730049   -0.00000000
 H                  0.00000000   -2.49460097   -0.00000000
 H                  2.16038781   -1.24730049   -0.00000000
    '''
    mol.basis = '6-31G*'
    mol.build()

    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    Func = 'B3LYP'
    mf.xc = Func
    mf.kernel()

    rt_td = RT_TDDFT(mf)
    rt_td.timestep=0.5
    rt_td.maxstep=2000
    rt_td.propagate()
    u_ao = mol.intor_symmetric('int1e_r', comp=3)
    

    
    rt_td.plot()

