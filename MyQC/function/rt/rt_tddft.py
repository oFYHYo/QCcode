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
            field = 8*1e-1*exp(-(time-3)**2/(2*0.2**2))
            #*cos(time*2*pi/20)

        return np.array([0,0,field])
    
    def get_dipole(self,u_ao,dm_ao):
        '''计算偶极矩'''
        mol = self.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        dipole_ele=np.einsum('xuv,uv->x',u_ao,dm_ao).real
        dipole_nuc=np.einsum('i,ix->x',charges,coords)
        return -dipole_ele+dipole_nuc

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
        Density_matrix[0] = dm_mo
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
        
        # fig , ax1 = plt.subplots()

        # ax1.plot(self.time*0.024,self.dipole[:,0].real,'b',label='energy')
        
        # ax1.set_ylabel('Dipole moment(a. u.)')


        # ax1.set_xlabel('time(fs)')

        # ax1.set_title('RT-TDDFT')
        # plt.show()
        
        grids =  dft.gen_grid.Grids(self.mol)
        grids.build()
        coords = grids.coords
        ni = dft.numint.NumInt()
        ao_0 = ni.eval_ao(self.mol,coords,deriv=0)
        
        mo_c = self.mo_coeff
        mo_0 = np.einsum('ui,ku->ik',mo_c,ao_0)
        rho = np.einsum('tij,ik,jk->tk',self.dm,mo_0,mo_0,optimize=True).real

        idx=[]
        
        for i in range(coords.shape[0]):
            if coords[i,0] == 0 and coords[i,1] == 0 and -4<coords[i,2]<4:
                idx.append(i)
        coords = coords[idx,:]
        rho = rho[:,idx]
        
        idex = coords[:,2].argsort()
        coords = coords[idex,:]
        rho = rho[:,idex]
        fig, ax = plt.subplots()
        ax.set_ylim([-0.5,4])
        ax.set_xlabel('Length(a. u.)')
        ax.set_ylabel('Electron density(a. u.)')
        line=ax.plot(coords[:,2],rho[0,:],color='cornflowerblue')

        def update(i):
            
            ax.set_title(f'T={(i)*self.timestep:2.2f}a.u.')
            line[0].set_ydata(rho[i,:])
            
            return line
        ani = animation.FuncAnimation(fig,update,interval=50,frames=range(rho.shape[0]))
        plt.show()
        ani.save("fig1.gif")

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

    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    Func = 'PBE'
    mf.xc = Func
    mf.kernel()

    rt_td = RT_TDDFT(mf)
    rt_td.timestep=0.1
    rt_td.maxstep=1000
    rt_td.propagate()
    # dm_ao = mf.make_rdm1()
    # grids =  dft.gen_grid.Grids(mol)
    # grids.build()
    # coords = grids.coords
    # ni = dft.numint.NumInt()
    # ao_0 = ni.eval_ao(mol,coords,deriv=0)
    
    # mo_c = rt_td.mo_coeff
    # mo_0 = np.einsum('ui,ku->ik',mo_c,ao_0)
    # rho = np.einsum('ij,ik,jk->k',rt_td.dm[0],mo_0,mo_0,optimize=True).real
    # rho1=ni.get_rho(mol,grids=grids,dm=dm_ao)
    # print(np.allclose(rho,rho1))
    
    rt_td.plot()

