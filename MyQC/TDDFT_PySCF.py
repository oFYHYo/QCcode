import numpy as np
from pyscf import gto,dft,tdscf,lib
import time
import matplotlib.pyplot as plt
from math import log,exp,pi


class TDA:
    
    def __init__(self,mf_hf):
        
        self.num_eig = 6
        self.mol = mf_hf.mol
        self.nocc =  int(mol.nelectron)//2
        self.mo_ene = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.e = None
        self.xy = None
        self.os = None
        self.FWHM = None
    
    def gen_vind(self):
        
        td = tdscf.TDA(mf_hf)
        vind, _ = td.gen_vind()
        self.vind = vind
        return vind
        

    def gen_hdiag(self):
        nocc = self.nocc
        mo_ene = self.mo_ene
        
        e_ia = (mo_ene[nocc:].reshape(-1,1) - mo_ene[:nocc]).T
        hdiag = e_ia.ravel()

        return hdiag

    def Davidson(self):

        x = time.time()
        
        hdiag = self.gen_hdiag()
        k = self.num_eig

        vind = self.gen_vind()

        A_size = self.nocc*(self.mo_ene.shape[0]-self.nocc)

        eig = 2*k

        # 生成初猜矩阵
        V_old = np.zeros([A_size,eig])
        hdiag = hdiag.reshape(-1,)
        Dsort = hdiag.argsort()
        for j in range(k):
            V_old[Dsort[j], j] = 1.0

        max_iter = 50
        
        print('iter       sub_size       |r|_max')
        for i in range(max_iter):
            
            # 获得子空间下的A矩阵
            #W_old = np.einsum("ki,il->kl",TD_matrix,V_old)
            W_old = vind(V_old.T).T
            sub_A = np.einsum("ik,il->kl",V_old,W_old)
            
            upper = np.triu_indices(n=sub_A.shape[0], k=1)
            lower = (upper[1], upper[0])
            sub_A[lower] = sub_A[upper]

            val,ket = np.linalg.eigh(sub_A)
            
            sub_val = val[:k]
            sub_ket = ket[:,:k]

            # 计算残差
            residual = np.einsum("ki,il->kl",W_old,sub_ket) - np.einsum("l,ki,il->kl",sub_val,V_old,sub_ket)
            
            r_norm = np.linalg.norm(residual,axis=0).tolist()
           
            max_norm = np.max(r_norm)
            print(f'{i+1:3d}{sub_A.shape[1]:12}{max_norm:18.6f}')
            if i>2 and max_norm < 1e-6:
                y = time.time()
                print(f'Davidson对角化完成，耗时{y-x:.6f}s')
                break
            
            # 计算新的初猜矢量
            t = 1e-8
            D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_val
            D = np.where( abs(D) < t, np.sign(D)*t, D)
            new_guess = residual/D

            # 正交化
            V_new = np.hstack([V_old,new_guess])
            V_old,_ = np.linalg.qr(V_new)

        self.e = sub_val[:k]
        self.xy = np.einsum("ki,il->kl",V_old,sub_ket)*np.sqrt(.5)
        
        return sub_val 
    
    def oscillator_strength(self):

        nocc = self.nocc
        
        mo_c = self.mo_coeff
        XY = self.xy
        X = XY.reshape(nocc,mo_c.shape[0]-nocc,-1)
        mol = self.mol

        # ao基下的偶极矩
        u_ao = mol.intor_symmetric('int1e_r', comp=3)

        # 计算基态到各激发态的跃迁偶极矩
        # Phys. Chem. Chem. Phys., 2020, 22, 26838.
        u_mo = lib.einsum("ua,xuv,vi->xia",mo_c[:,nocc:],u_ao,mo_c[:,:nocc])
        u_trans = lib.einsum('iak,xia->xk',X,u_mo)*2
    
        # 计算振子强度
        u2 = (u_trans**2).sum(0)
        f = 2*self.e*(u2)/3

        self.os = f
        print(self.os)

        return f

    def plot(self,type='eV', xrange=None):

        if type == 'eV':
            label = 'Excited energy(eV)'
            ene = self.e*self.Hartree_to_eV
            if self.FWHM == None:
                FWHM = 0.66667
            else:
                FWHM = self.FWHM
        
        elif type == 'nm':
            label = 'Wavelength(nm)'
            ene = 1240.7011/(self.e*self.Hartree_to_eV)
            if self.FWHM == None:
                FWHM = 100
            else:
                FWHM = self.FWHM

        if xrange == None:
            ma = np.max(ene)
            mi = np.min(ene)
            L = (ma-mi)/8
            xrange = [mi-L,ma+L]
 
        x = np.arange(xrange[0],xrange[1],(xrange[1]-xrange[0])/10000)
        y = np.zeros_like(x)
        for i in range(self.num_eig):
            yi = self.Gau(x,ene[i],FWHM)
            yi = yi*self.os[i]
            y += yi
            plt.plot(x,yi)
            plt.bar(ene[i],self.os[i],width=0.01)

        plt.plot(x,y)
        plt.xlabel(label)
        plt.ylabel('Oscillator_Strength')
        plt.title('UV-Vis')
        plt.show()

    
    def Gau(self,x,x0,FWHM):
        c = FWHM/(2*np.sqrt(2*np.log(2)))
        z = np.exp(-(x-x0)**2/(2*(c**2)))
        return z/(c*np.sqrt(2*pi))
    
    def Lor(self,x,ene):
        y = 0
        for i in range(self.num_eig):
            y += self.os[i]/((x-ene[i])**2+0.25*self.FWHM**2)
        return y*self.FWHM/(2*pi)
    
    def PV(self,x,ene,weight=0.5):
        return weight*self.Gau(x,ene)+(1-weight)*self.Lor(x,ene)

    def kernel(self):
        print('开始对角化')
        val = self.Davidson()
        self.Hartree_to_eV = 27.211385050
        print('Excited energy(eV)')
        print(val*self.Hartree_to_eV)

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = '''
    C                  0.70961700    0.00001300    0.00280400
    C                  0.02236000    1.21539400    0.00395200
    C                 -1.36207000    1.21443500    0.00511500
    C                 -2.07956400    0.00001000    0.00585600
    C                 -1.36204100   -1.21441300    0.00526800
    C                  0.02237800   -1.21537900    0.00409700
    H                  0.58456400    2.14068200    0.00122300
    H                 -1.90417300    2.15610600    0.00960600
    H                 -1.90421700   -2.15604300    0.00981500
    H                  0.58455100   -2.14068800    0.00145800
    N                  2.16356300   -0.00000400   -0.00311100
    O                  2.74121800    1.09085300   -0.00539300
    O                  2.74116600   -1.09087600   -0.00578500
    N                 -3.45821100    0.00000000    0.05558900
    H                 -3.93071600    0.84590900   -0.23149700
    H                 -3.93063400   -0.84610300   -0.23107900
    '''
    mol.basis = 'sto-3g'
    mol.build()

    Func = 'PBE'

    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    mf_hf.kernel()

    td = TDA(mf_hf)
    td.num_eig =  6
    td.kernel()

    #td.FWHM=100
    os = td.oscillator_strength()
    td.plot('eV')
'''
    td = tdscf.TDA(mf_hf)
    td.nstates = 6
    td.kernel()
    td.analyze()
'''