import numpy as np
from pyscf import gto,scf,ao2mo
import time
from functools import partial

einsum = partial(np.einsum,optimize=True)

class CCD:
    def __init__(self,mf,max_cycle=100,t2_res=1e-8,ene_res=1e-8):
        self.mf = mf
        self.mol = mf.mol
        self.mo_ene = mf.mo_energy
        self.mo = mf.mo_coeff

        self.max_cycle = max_cycle
        self.t2_res=t2_res
        self.ene_res=ene_res

    def gen_ant_eri_mo(self):
        '''生成自旋轨道基下的反对称双电子积分<ij||kl>'''
        eri_ao = self.mol.intor("int2e")
        mo = self.mf.mo_coeff

        from pyscf import ao2mo
        mo_a = np.zeros((mo.shape[0], mo.shape[1]*2), dtype=mo.dtype)
        mo_b = np.zeros((mo.shape[0], mo.shape[1]*2), dtype=mo.dtype)
        for i in range(mo.shape[1]):
            mo_a[:,2*i] = mo[:,i]
            mo_b[:,2*i+1] = mo[:,i]
        eri  = ao2mo.kernel(eri_ao, mo_a)
        eri += ao2mo.kernel(eri_ao, mo_b)
        eri1 = ao2mo.kernel(eri_ao, (mo_a,mo_a,mo_b,mo_b))
        eri += eri1
        eri += eri1.T
        # if eri.dtype == np.double:
        #     eri = ao2mo.restore(1, eri, 28)
        eri = eri.transpose(0,2,1,3)
        ant_eri_mo = eri - eri.transpose(0,1,3,2)

        return ant_eri_mo

    def get_energy(self,amp,w):
        '''计算CCD相关能'''
        t2 = amp
        return 0.25*einsum('ijab,abij->', w.Ioovv, t2)

    def update_amp(self,amp,w):
        '''更新t2振幅'''
        t2=amp
        eabij = w.eabij
        
        res  = w.Ivvoo.copy()

        res += 0.5*einsum('klij,abkl->abij',w.Ioooo,t2)
        res += 0.5*einsum('abcd,cdij->abij',w.Ivvvv,t2)
        
        tmp = einsum('kbcj,acik->abij',w.Iovvo,t2)
        tmp1 = tmp-tmp.transpose(1,0,2,3)
        res += tmp1-tmp1.transpose(0,1,3,2)

        tmp = einsum('klcd,acik,dblj->abij',w.Ioovv,t2,t2)
        tmp1 = tmp-tmp.transpose(1,0,2,3)
        res += 0.5*(tmp1-tmp1.transpose(0,1,3,2))

        res += 0.25*einsum('klcd,cdij,abkl->abij',w.Ioovv,t2,t2)

        tmp = einsum('klcd,acij,bdkl->abij',w.Ioovv,t2,t2)
        tmp1 = tmp-tmp.transpose(1,0,2,3)
        res -= 0.5*tmp1

        tmp = einsum('klcd,abik,cdjl->abij',w.Ioovv,t2,t2)
        tmp1 = tmp-tmp.transpose(0,1,3,2)
        res -= 0.5*tmp1

        t2_new = res/eabij

        return t2_new
    
    def gen_init_amp(self,w):
        '''计算t2振幅初猜（MP2振幅）'''
        return w.Ivvoo/w.eabij

    # def diis(self,X_old):
    #     from numpy import linalg

    #     for X

    #     return X_new


    def kernel(self):
        ant_eri_mo = self.gen_ant_eri_mo()
        w = self.CCD_w(self.mf,ant_eri_mo)

        t2_init = self.gen_init_amp(w)
        ene_init = self.get_energy(t2_init,w)
        print(f'初始MP2能量为{ene_init:3.6f}')

        t2 = t2_init
        ene = ene_init
        for i in range(self.max_cycle):
            t2_new = self.update_amp(t2,w)
            ene_new = self.get_energy(t2_new,w)
            
            t2_res = np.linalg.norm(abs(t2_new-t2))
            ene_res = abs(ene_new-ene)
            if t2_res <= self.t2_res and ene_res <= self.ene_res:
                break

            t2 = t2_new
            ene = ene_new

            
            print(f'iter            E_corr                 |e|                  |t2|')
            print(f'{i:2}          {ene:12.8f}         {ene_res:12.8f}          {t2_res:12.8f}')
        
        self.t2 = t2
        self.ecorr = ene
        return self.ecorr


    class CCD_w:
        '''存放一些需要用到的中间量'''
        def __init__(self,mf,ant_eri_mo):
            nocc = mf.mol.nelectron
            mo_ene = mf.mo_energy
            e = np.vstack([mo_ene,mo_ene]).T.reshape(2*mo_ene.shape[0])
            self.eabij = -e[nocc:,None,None,None]+e[None,None,:nocc,None]-e[None,nocc:,None,None]+e[None,None,None,:nocc]

            f = np.diag(e)
            I = ant_eri_mo


            self.foo = f[:nocc,:nocc]
            self.fvv = f[nocc:,nocc:]
            self.fov = f[:nocc,nocc:]
            self.fvo = f[nocc:,:nocc]

            self.Ioooo = I[:nocc,:nocc,:nocc,:nocc]
            self.Iooov = I[:nocc,:nocc,:nocc,nocc:]
            self.Ioovo = I[:nocc,:nocc,nocc:,:nocc]
            self.Iovoo = I[:nocc,nocc:,:nocc,:nocc]
            self.Ivooo = I[nocc:,:nocc,:nocc,:nocc]
            self.Ioovv = I[:nocc,:nocc,nocc:,nocc:]
            self.Iovov = I[:nocc,nocc:,:nocc,nocc:]
            self.Ivoov = I[nocc:,:nocc,:nocc,nocc:]
            self.Iovvo = I[:nocc,nocc:,nocc:,:nocc]
            self.Ivovo = I[nocc:,:nocc,nocc:,:nocc]
            self.Ivvoo = I[nocc:,nocc:,:nocc,:nocc]
            self.Ivvvo = I[nocc:,nocc:,nocc:,:nocc]
            self.Ivvov = I[nocc:,nocc:,:nocc,nocc:]
            self.Ivovv = I[nocc:,:nocc,nocc:,nocc:]
            self.Iovvv = I[:nocc,nocc:,nocc:,nocc:]
            self.Ivvvv = I[nocc:,nocc:,nocc:,nocc:]
        
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
    mol.basis = 'sto-3g'
    mol.build()
    HF = scf.RHF(mol).run()

    # PySCF CCD
    from pyscf.cc import ccd
    pyscf_ccd = ccd.CCD(HF)
    pyscf_ccd.kernel()
    pyscf_ene = pyscf_ccd.e_corr

    # My CCD
    myccd = CCD(HF)
    myccd.kernel()
    my_ene = myccd.ecorr

    print(f'PySCF\'s CCD correction energy = {pyscf_ene:2.8f}')
    print(f'My CCD correction energy = {my_ene:2.8f}')