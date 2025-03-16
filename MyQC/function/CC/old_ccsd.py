import numpy as np
from pyscf import gto,scf,ao2mo
import time
from functools import partial

einsum = partial(np.einsum,optimize=True)

class CCSD:
    def __init__(self,mf,max_cycle=100,conv_t1=1e-10,conv_t2=1e-10,conv_e=1e-1,use_diis=True,diis_iter=6):
        self.mf = mf
        self.mol = mf.mol
        self.mo_ene = mf.mo_energy
        self.mo = mf.mo_coeff

        self.max_cycle = max_cycle
        self.use_diis = use_diis
        self.diis_iter = diis_iter
        self.conv_t1=conv_t1
        self.conv_t2=conv_t2
        self.conv_e=conv_e

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
        eri = eri.transpose(0,2,1,3)
        ant_eri_mo = eri - eri.transpose(0,1,3,2)

        return ant_eri_mo

    def get_energy(self,amp,w):
        '''计算CCD相关能'''
        t1 = amp[0]
        t2 = amp[1]
        
        E = 0.25*einsum('ijab,abij->', w.Ioovv, t2)
        E += einsum('ia,ai->',w.fov,t1)
        E += 0.5*einsum('ijab,ai,bj->',w.Ioovv,t1,t1)
        
        return E 

    def update_amp(self,amp,w):
        '''更新t2振幅'''
        t1 = amp[0]
        t2 = amp[1]
        eai = w.eai
        eabij = w.eabij

        tmp = np.einsum("ai,bj->abij", t1, t1, optimize = True)
        tmp -= tmp.transpose(1,0,2,3)
        tau = t2 + tmp
        taut = t2 + 0.5*tmp

        Fae = einsum('mafe,fm->ae',w.Iovvv,t1) \
        - 0.5*einsum('afmn,mnef->ae',taut,w.Ioovv)

        Fmi = np.einsum('en,inje->ij', t1, w.Iooov, optimize=True) \
          + 0.5*np.einsum("efjn,inef->ij", taut, w.Ioovv, optimize = True)

        Fme = np.einsum('fn,inaf->ia', t1, w.Ioovv, optimize=True) 

        tmp = einsum('ej,mnie->mnij',t1,w.Iooov)
        Wmnij = w.Ioooo + tmp-tmp.transpose(0,1,3,2)+0.25*einsum('efij,mnef->mnij',tau,w.Ioovv)

        tmp = einsum('amef,bm->abef',w.Ivovv,t1)
        Wabef = w.Ivvvv - tmp + tmp.transpose(1,0,2,3)+0.25*einsum('abmn,mnef->abef',tau,w.Ioovv)

        Wmbej = w.Iovvo + einsum('fj,mbef->mbej',t1,w.Iovvv) - einsum('bn,mnej->mbej',t1,w.Ioovo) \
            - 0.5*einsum('fbjn,mnef->mbej',t2,w.Ioovv) - einsum('fj,bn,mnef->mbej',t1,t1,w.Ioovv)

        # calculate t1
        res1  = w.fvo.copy()
        res1 += einsum('ei,ae->ai',t1,Fae)
        res1 -= einsum('am,mi->ai',t1,Fmi)
        res1 += einsum('aeim,me->ai',t2,Fme) 
        res1 -= einsum('fn,naif->ai',t1,w.Iovov)
        res1 -= 0.5*einsum('efim,maef->ai',t2,w.Iovvv)
        res1 -= 0.5*einsum('aemn,nmei->ai',t2,w.Ioovo)

        t1_new = res1/eai

        # calculate t2
        res2  = w.Ivvoo.copy()
        
        tmp = Fae - 0.5*einsum('bm,me->be',t1,Fme)
        tmp1 = einsum('aeij,be->abij',t2,tmp)
        res2 += tmp1 - tmp1.transpose(1,0,2,3)

        tmp = Fmi + 0.5*einsum('ej,me->mj',t1,Fme)
        tmp1 = einsum('abim,mj->abij',t2,tmp)
        res2 -= tmp1 - tmp1.transpose(0,1,3,2)

        res2 += 0.5*einsum('abmn,mnij->abij',tau,Wmnij)
        res2 += 0.5*einsum('efij,abef->abij',tau,Wabef)

        tmp = einsum('aeim,mbej->abij',t2,Wmbej)
        tmp -= einsum('ei,am,mbej->abij',t1,t1,w.Iovvo)
        tmp1 = tmp - tmp.transpose(1,0,2,3)
        res2 += tmp1-tmp1.transpose(0,1,3,2) 

        tmp = einsum('ei,abej->abij',t1,w.Ivvvo)
        res2 += tmp - tmp.transpose(0,1,3,2)

        tmp = einsum('am,mbij->abij',t1,w.Iovoo)
        res2 -= tmp - tmp.transpose(1,0,2,3)

        t2_new = res2/eabij

        return [t1_new,t2_new]
    
    def gen_init_amp(self,w):
        '''计算t1t2振幅初猜'''
        t1 = np.zeros_like(w.fvo)
        t2 = w.Ivvoo/w.eabij

        return [t1,t2]

    def kernel(self):
        ant_eri_mo = self.gen_ant_eri_mo()
        w = self.CC_w(self.mf,ant_eri_mo)

        amp_init = self.gen_init_amp(w)
        ene_init = self.get_energy(amp_init,w)
        print(f'初始MP2能量为{ene_init:3.8f}')

        t1 = amp_init[0]
        t2 = amp_init[1]
        ene = ene_init

        if self.use_diis:
            diis = self.diis([t1,t2],diis_iter=self.diis_iter)

        for i in range(self.max_cycle):
            t1_new,t2_new = self.update_amp([t1,t2],w)
            
            t1_res = np.linalg.norm(abs(t1_new-t1))
            t2_res = np.linalg.norm(abs(t2_new-t2))

            if self.use_diis:
                diis.append([t1_new,t2_new])

                if i > 1:
                    t1,t2 = diis.update()

                else:
                    t1 = t1_new
                    t2 = t2_new

            else:
                t1 = t1_new
                t2 = t2_new

            ene_new = self.get_energy([t1,t2],w)
            ene_res = abs(ene_new-ene)
            ene = ene_new

            if t2_res+t1_res <= self.conv_t2 and ene_res <= self.conv_e:
                conv = True
                break

            print(f'iter            E_corr                 |e|                  |res_t2|                |res_t1|' )
            print(f'{i+1:2}          {ene:12.8f}         {ene_res:12.8f}          {t2_res:12.8f}           {t1_res:12.8f}')
        
        self.t1 = t1_new
        self.t2 = t2_new
        self.ecorr = ene_new
        return self.ecorr

    class diis:
        '''diis方法求解振幅方程'''
        def __init__(self,amp,diis_iter=6):
            
            self.diis_iter = diis_iter
            self.diis_res_t1 = [amp[0].copy()]
            self.diis_res_t2 = [amp[1].copy()]
            self.err_vec = []
            self.oldt1 = amp[0].copy()
            self.oldt2 = amp[1].copy()

        def append(self,amp):
            self.diis_res_t1.append(amp[0].copy())
            self.diis_res_t2.append(amp[1].copy())

            err_t1 = (self.diis_res_t1[-1]-self.oldt1).ravel()
            err_t2 = (self.diis_res_t2[-1]-self.oldt2).ravel()
            err = np.concatenate((err_t1,err_t2))
            self.err_vec.append(err)

            self.oldt1 = amp[0].copy()
            self.oldt2 = amp[1].copy()

            if len(self.diis_res_t1) > self.diis_iter:
                self.diis_res_t1.pop(0)
                self.diis_res_t2.pop(0)
                self.err_vec.pop(0)

        
        def gen_Bmatrix(self,err_vec):
            '''生成B matrix'''
            B = np.zeros((len(err_vec)+1,len(err_vec)+1))
            B[:,-1] = -1
            B[-1,:] = -1
            B[-1,-1] = 0
            for i in range(len(err_vec)):
                for j in range(len(err_vec)):
                    B[i,j] = np.dot(err_vec[i],err_vec[j])
            B[:-1,:-1] /= abs(B[:-1,:-1]).max()

            return B

        def update(self):

            B = self.gen_Bmatrix(self.err_vec)
            b = np.zeros(B.shape[0])
            b[-1] = -1

            coeff = np.linalg.solve(B,b)
            coeff = coeff[:-1]

            t1_new = np.zeros_like(self.diis_res_t1[0])
            t2_new = np.zeros_like(self.diis_res_t2[0])
            for i in range(len(coeff)):
                t1_new += coeff[i]*self.diis_res_t1[i+1]
                t2_new += coeff[i]*self.diis_res_t2[i+1]

            self.oldt1 = t1_new.copy()
            self.oldt2 = t2_new.copy()
            return [t1_new,t2_new]
        

    class CC_w:
        '''存放一些需要用到的中间量'''
        def __init__(self,mf,ant_eri_mo):
            nocc = mf.mol.nelectron
            mo_ene = mf.mo_energy
            e = np.vstack([mo_ene,mo_ene]).T.reshape(2*mo_ene.shape[0])
            self.eabij = -e[nocc:,None,None,None]+e[None,None,:nocc,None]-e[None,nocc:,None,None]+e[None,None,None,:nocc]
            self.eai = -e[nocc:,None]+e[None,:nocc]

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
    mol.basis = '6-31G'
    mol.verbose =3
    mol.build()
    HF = scf.RHF(mol).run()

    # PySCF CCSD
    from pyscf.cc import ccsd
    pyscf_ccsd = ccsd.CCSD(HF)
    pyscf_ccsd.conv_tol = 1e-8
    pyscf_ccsd.conv_tol_normt = 1e-8
    pyscf_ccsd.kernel()
    pyscf_ene = pyscf_ccsd.e_corr

    # My CCSD
    myccsd = CCSD(HF)
    myccsd.conv_t1 = 1e-8
    myccsd.conv_t2 = 1e-8
    myccsd.conv_e = 1e-8
    myccsd.use_diis = True
    myccsd.kernel()
    my_ene = myccsd.ecorr

    print(f'PySCF CCSD correlation energy = {pyscf_ene:2.8f}')
    print(f'My CCSD correlation energy = {my_ene:2.8f}')
