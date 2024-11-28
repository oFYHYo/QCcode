from math_helper import *
import numpy as np
from pyscf import gto,scf,ao2mo
import scipy
import pyscf
from functools import reduce

class FCI:

    def __init__(self,mf_hf):

        self.mol = mf_hf.mol
        self.mf_hf = mf_hf
        self.nelecs = self.mol.nelec
        self.nalph = self.mol.nelec[0]
        self.nbeta = self.mol.nelec[1]
        self.nmo = mf_hf.mo_energy.shape[0]
        
        self.gen_moint()

    def get_matrix_element(self,config1,config2):
        ''' 计算组态12与哈密顿量之间的积分 '''
        
        h1e = self.h1e
        h2e = self.h2e

        comm_idx_alph, comm_idx_beta, diff_idx_1_alph, diff_idx_2_alph, diff_idx_1_beta, diff_idx_2_beta, diff_num = diff_config(config1,config2)

        H = 0
        # Slater-Condon 规则计算矩阵元
        # szabo P72, Table 2.5 & Table 2.6
        if diff_num == 0:
            
            assert len(comm_idx_alph) == self.nalph
            assert len(comm_idx_beta) == self.nbeta
            for ia in comm_idx_alph:
                H += h1e[ia,ia]

                for ja in comm_idx_alph:
                    H += 0.5*h2e[ia,ia,ja,ja]
                    H -= 0.5*h2e[ia,ja,ja,ia]

                for jb in comm_idx_beta:
                    H += 0.5*h2e[ia,ia,jb,jb]

            for ib in comm_idx_beta:
                H += h1e[ib,ib]

                for jb in comm_idx_beta:
                    H += 0.5*h2e[ib,ib,jb,jb]
                    H -= 0.5*h2e[ib,jb,jb,ib]

                for ja in comm_idx_alph:
                    H += 0.5*h2e[ib,ib,ja,ja]
        
        elif diff_num == 1:
            
            if diff_idx_1_alph == []:
                i = min(diff_idx_1_beta[0],diff_idx_2_beta[0])
                a = max(diff_idx_1_beta[0],diff_idx_2_beta[0])
                H += h1e[i,a]
                
                for ja in comm_idx_alph:
                    H += h2e[i,a,ja,ja]
                    #H -= h2e[i,ja,ja,a]
                
                for jb in comm_idx_beta:
                    H += h2e[i,a,jb,jb]
                    H -= h2e[i,jb,jb,a]
                
                H *= (- 1.0) ** sum(config1.occ_idx_beta[i+1:a])

            elif diff_idx_1_beta == []:
                i = min(diff_idx_1_alph[0],diff_idx_2_alph[0])
                a = max(diff_idx_1_alph[0],diff_idx_2_alph[0])
                H += h1e[i,a]

                for jb in comm_idx_beta:
                    H += h2e[i,a,jb,jb]
                    #H -= h2e[i,jb,jb,a]

                for ja in comm_idx_alph:
                    H += h2e[i,a,ja,ja]
                    H -= h2e[i,ja,ja,a]

                H *= (- 1.0) ** sum(config1.occ_idx_alph[i+1:a])
            else:
                raise ValueError('diff_num != 1')

        elif diff_num == 2:

            if diff_idx_1_alph == []:
                i = min(diff_idx_1_beta)
                j = max(diff_idx_1_beta)
                a = min(diff_idx_2_beta)
                b = max(diff_idx_2_beta)
                comm_occ_beta = [1 if i in comm_idx_beta else 0 for i in range(self.nmo)]

                H += h2e[i,a,j,b]
                H -= h2e[i,b,j,a]

                H *= (- 1.0) ** sum(comm_occ_beta[min(i,a)+1:max(i,a)])
                H *= (- 1.0) ** sum(comm_occ_beta[min(j,b)+1:max(j,b)])
                    
            elif diff_idx_1_beta == []:
                i = min(diff_idx_1_alph)
                j = max(diff_idx_1_alph)
                a = min(diff_idx_2_alph)
                b = max(diff_idx_2_alph)
                comm_occ_alph = [1 if i in comm_idx_alph else 0 for i in range(self.nmo)]
               
                H += h2e[i,a,j,b]
                H -= h2e[i,b,j,a]

                H *= (- 1.0) ** sum(comm_occ_alph[min(i,a)+1:max(i,a)])
                H *= (- 1.0) ** sum(comm_occ_alph[min(j,b)+1:max(j,b)])
                
            else:
                
                ia = min([diff_idx_1_alph[0],diff_idx_2_alph[0]])
                aa = max([diff_idx_1_alph[0],diff_idx_2_alph[0]])
                jb = min([diff_idx_1_beta[0],diff_idx_2_beta[0]])
                bb = max([diff_idx_1_beta[0],diff_idx_2_beta[0]])

                H += h2e[ia,aa,jb,bb]
                
                H *= (- 1.0) ** sum(config1.occ_idx_alph[ia+1:aa])
                H *= (- 1.0) ** sum(config1.occ_idx_beta[jb+1:bb])
        return H


    def get_fci_matrix(self):
        ''' 计算FCI矩阵 '''

        # 生成所有的可能组态
        config_comb = gen_all_config(self.nmo,self.nalph,self.nbeta)
        C_a = comb(self.nmo,self.nalph)
        C_b = comb(self.nmo,self.nbeta)
        fci_matrix = np.zeros((C_a*C_b,C_a*C_b))

        for i,config1 in enumerate(config_comb):
            for j,config2 in enumerate(config_comb):
                fci_matrix[i,j] += self.get_matrix_element(config1,config2)
        
        return fci_matrix
    
    def kernel(self):
        # 获得FCI矩阵
        fci_matrix = self.get_fci_matrix()
        
        # 对角化FCI矩阵，获得能量最低本征值，即为体系电子能量
        E,_ = Davidson_fci(fci_matrix)
        E_nuc = self.mol.energy_nuc()
        #E,_ = np.linalg.eigh(fci_matrix)
        self.e = np.sort(E)[0]
        print(f'Full CI 计算完毕')

        print(f'E_FCI = {self.e+E_nuc:.8f}')
        


    def gen_moint(self):
        '''生成mo基下的电子积分
        -----
        h1e,h2e       : ao基下的电子积分
        h1e_mo,h2e_mo : mo基下的电子积分
        '''
        mf_hf = self.mf_hf
        
        h1e = mf_hf.get_hcore()
        h2e = mf_hf._eri
        mo_c = mf_hf.mo_coeff
        nmo = mo_c.shape[1]

        h1e_mo = reduce(np.dot, (mo_c.T, h1e, mo_c))
        h2e_mo = ao2mo.kernel(h2e, mo_c, compact=False)
        h2e_mo = h2e_mo.reshape(nmo, nmo, nmo, nmo)

        self.h1e = h1e_mo
        self.h2e = h2e_mo

        return h1e_mo, h2e_mo


   
if __name__ == '__main__':
    
    mol = pyscf.M()
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.spin = 0
    mol.charge = 0
    mol.basis  = 'sto-3g'
    mol.build()

    E_nuc = mol.energy_nuc()
    mf_hf = mol.RHF()
    mf_hf.kernel()
    
    print('myFCI')
    fci = FCI(mf_hf)
    fci.kernel()
    
    print('PySCF.fci')
    cisolver = pyscf.fci.FCI(mf_hf)
    print('E_FCI = %.8f' % cisolver.kernel()[0])



