import numpy as np
from pyscf import gto,scf
from time import time
from functools import partial
from const import *

einsum = partial(np.einsum,optimize=True)

class AIMD:

    def __init__(self,mol,time_step=1,T_em=298.15,t_heat=30,max_step=1000):
        
        mol.verbose = 0
        self.mol = mol
        self.mf = scf.RHF(mol)
        self.energy = self.mf.kernel()
        
        self.m = mol.atom_mass_list().reshape(-1,1)
        self.natm = self.m.shape[0]
        atom = []
        for i in range(self.natm):
            atom.append(mol.atom_symbol(i))
        self.atom_list = atom

        self.time_step = time_step
        self.max_step = max_step
        self.t_heat = t_heat
        self.T_em = T_em

    def gen_init_vec(self):
        v_rand = np.random.normal(loc=0,scale=1.0,size=[self.natm,3])
        
        mass_tot = self.m.sum()
        v_avg = v_rand.sum(1).reshape(-1,1)*self.m/mass_tot
        v = v_rand - v_avg
        v = v*np.sqrt(kb*self.T_em*J2au/(self.m*dal2au))

        return v

    def Berendson(self,v):

        T = self.calc_temp(v)
        T0 = self.T_em
        t = self.time_step*fs2au
        tau = self.t_heat*fs2au
        f = np.sqrt(1+(T0/T-1)*t/tau)
        # if f > 1.1:
        #     f = 1.1
        # elif f < 0.9:
        #     f = 0.9
        v_scal = v*f

        return v_scal

    def calc_temp(self,v):
        
        v2 = v**2
        Ekin = 0.5*(self.m*v2*dal2au).sum()
        temp = 2.0*Ekin/(3.0*self.natm*kb*J2au)
        
        return temp

    def integrator(self,r,v,F):
        t = self.time_step*fs2au
        r_new = r + v*t + 0.5*F*t**2/(self.m*dal2au)
        mol = self.mol
        mol.set_geom_(r_new)
        mol.unit = 'Bohr'
        mol.build()

        mf = self.mf
        mf = scf.RHF(mol)
        self.energy = mf.kernel()
        g = mf.Gradients()
        F_new = -g.kernel()

        v_new = v + t*(F+F_new)/(2*self.m*dal2au)
        mass_tot = self.m.sum()
        v_avg = (v_new*self.m).sum(0)/mass_tot
        v_new -= v_avg.reshape(1,-1)

        return r_new,v_new,F_new
    
    def save_xyz(self,step,coord):

        if step == 0:
            file = open('pos.xyz','w')
        else:
            file = open('pos.xyz','a+')
        file.write(f'{coord.shape[0]}\n')
        file.write(f'# step={step:6} energy={self.energy:12.8f}\n')
        for i in range(coord.shape[0]):
            file.write(f'{self.atom_list[i]:3} {coord[i,0]:12.8f} {coord[i,1]:12.8f} {coord[i,2]:12.8f}\n')
            
        file.close()
    
    def save_prop(self,step,v):
        Ekin = 0.5*(self.m*v**2*dal2au).sum()
        T = self.calc_temp(v)
        
        if step == 1:
            file = open('prop.log','w')
            file.write('  step          time          temperature          energy_tot          energy_kin          energy_pot\n')
            print('  step          time          temperature          energy_tot          energy_kin          energy_pot')
            file.close()
        file = open('prop.log','a+')
        file.write(f'{step:4} {self.time_step*step:16.2f} {T:18.2f} {self.energy+Ekin:20.6f} {Ekin:19.6f} {self.energy:19.6f}\n')
        print(f'{step:4} {self.time_step*step:16.2f} {T:18.2f} {self.energy+Ekin:20.6f} {Ekin:19.6f} {self.energy:19.6f}')
        file.close()

    def kernel(self):
        
        r = [mol.atom_coords(unit='Bohr')]
        v = [self.gen_init_vec()]
        F = [-self.mf.Gradients().kernel()]
        self.save_xyz(0,r[0]/A2bohr)
        for idx in range(1,self.max_step+1,1):
            r_old = r[idx-1]
            v_old = v[idx-1]
            F_old = F[idx-1]

            r_new,v_new,F_new = self.integrator(r_old,v_old,F_old)
            v_new_scal = self.Berendson(v_new)

            r.append(r_new)
            v.append(v_new_scal)
            F.append(F_new)

            self.save_xyz(idx,r_new/A2bohr)
            self.save_prop(idx,v_new_scal)
            
            


if __name__ == '__main__':

    mol = gto.Mole()
    mol.atom =''' 
 O                 -3.01915195    0.11651239    0.00000000
 H                 -2.12897584   -0.24291284    0.00000000
 H                 -2.97700797    1.07558688    0.00000000
 O                 -0.69769181   -0.67336147    0.00000000
 H                  0.22163321   -0.94984202    0.00000000
 H                 -0.91529972   -1.60362340    0.09412507
    '''
    mol.basis = 'sto-3g'
    mol.build()

    x = time()
    aimd = AIMD(mol,max_step=1000,time_step=0.5)
    aimd.kernel()
    y = time()
    print(f'AIMD simulation is completed in {y-x:.2f} s')