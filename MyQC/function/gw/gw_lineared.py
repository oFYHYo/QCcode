import numpy as np
import scipy
from pyscf import gto,scf,ao2mo,dft,df
import time
from functools import partial

einsum = partial(np.einsum,optimize=True)

'''
Ref: 
J. Chem. Theory Comput. 2016, 12, 8, 3623–3635
'''

class GW():
    def __init__(self,dft):
        self.dft = dft
        self.mol = mol

    def quasi_partical_energy(self,mo_energy):
        '''计算准粒子能量'''
        E = 0.0 
        E += mo_energy

        mf = scf.RHF(self.mol)
        vh = mf.get_j()
        vxc = self.dft.get_veff() - vh
        sigma_xc = self.calc_sigma_diag()

        return E + (vh + sigma_xc - vxc)    
    
    def calc_sigma_diag(self,mo_ene):

        sigma_xc = 0.0
        # calculate sigma_xc

        return sigma_xc

    def calc_density_response(self):

        return 1

    def two_pole(self,omega,coeff):
        '''calculate two_pole fit function in omega'''
        a = coeff[:5] + 1j *coeff[5:]
        
        return a[0] + a[1]/(1j*omega + a[2]) + a[3]/(1j*omega + a[4])
    
    def two_pole_fit(self,func):

        x0 = np.random.randn(10)

        return 1
    
    def kernel(self):
        mol = self.mol
        auxbasis = 'ccpvdz-ri'
        auxmol = df.addons.make_auxmol(mol, auxbasis)

        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')

        nocc = mol.n
        nao = mol.nao
        naux = auxmol.nao

        A = scipy.linalg.sqrtm(ints_2c2e)
        df_coef = scipy.linalg.solve(A, ints_3c2e.reshape(nao*nao, naux).T)
        B_Q_uv = df_coef.reshape(naux, nao, nao) # <P|uv>

        mo = self.dft.mo_coeff
        mo_ene = self.dft.mo_energy

        mo_a = np.zeros((mo.shape[1], mo.shape[1]*2), dtype=mo.dtype)
        mo_b = np.zeros((mo.shape[1], mo.shape[1]*2), dtype=mo.dtype)
        so_ene = np.zeros(mo.shape[1]*2, dtype=mo.dtype)
        for i in range(mo.shape[1]):
            mo_a[:,2*i] = mo[0,:,i]
            mo_b[:,2*i+1] = mo[1,:,i]
            so_ene[2*i] = mo_ene[0,i]
            so_ene[2*i+1] = mo_ene[1,i]
        B_Q_pq  = np.einsum('Puv,up,vq->Ppq', B_Q_uv, mo_a, mo_a) 
        B_Q_pq += np.einsum('Puv,up,vq->Ppq', B_Q_uv, mo_b, mo_b)
        F_mo = np.diag(so_ene)
        del ints_2c2e, ints_3c2e
        o = slice(0, 2*nocc)
        v = slice(2*nocc, 2*nao)

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom =''' 
    C                 -0.66295800    0.00000000   -0.00000000
    C                  0.66295800    0.00000000   -0.00000000
    H                 -1.25654334    0.92403753    0.00000000
    H                 -1.25654334   -0.92403753    0.00000000    
    H                  1.25654334   -0.92403753    0.00000000
    H                  1.25654334    0.92403753   -0.00000000
    '''
    mol.basis = '6-31G'
    mol.verbose =3
    mol.build()
    HF = scf.RHF(mol).run()