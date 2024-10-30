import numpy as np
from pyscf import gto,dft,tdscf
import time

class TDA:
    
    def __init__(self,mf_hf,mol):

        td = tdscf.TDA(mf_hf)
        A,_ = td.get_ab()

        self.num_eig = 6
        self.nocc =  int(mol.nelectron)//2
        self.mo_ene = mf_hf.mo_energy
        self.A_size = A.shape[0]*A.shape[1]
        

        TD_matrix = A
        self.TD_matrix = TD_matrix.reshape(self.A_size,self.A_size)

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
        TD_matrix = self.TD_matrix

        A_size = TD_matrix.shape[0]

        eig = 2*k

        # 生成初猜矩阵
        V_old = np.zeros([A_size,eig])
        hdiag = hdiag.reshape(-1,)
        Dsort = hdiag.argsort()
        for j in range(k):
            V_old[Dsort[j], j] = 1.0

        max_iter = 50
        for i in range(max_iter):
            
            # 获得子空间下的A矩阵
            W_old = np.einsum("ki,il->kl",TD_matrix,V_old)
            sub_A = np.einsum("ik,il->kl",V_old,W_old)
            
            val,ket = np.linalg.eigh(sub_A)
            
            sub_val = val[:k]
            sub_ket = ket[:,:k]

            # 计算残差
            residual = np.einsum("ki,il->kl",W_old,sub_ket) - np.einsum("l,ki,il->kl",sub_val,V_old,sub_ket)
            
            r_norm = np.linalg.norm(residual,axis=0).tolist()
           
            max_norm = np.max(r_norm)
            if i>2 and max_norm < 1e-8:
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

        return sub_val
    
    def kernel(self):

        val = self.Davidson()
        Hartree_to_eV = 27.211385050
        print(val*Hartree_to_eV)

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = '''
    C                 -0.71205900   -0.00001100    0.00164200
    C                 -0.02242200    1.21633600    0.00195500
    C                  1.36411500    1.21608600    0.00308400
    C                  2.08781200   -0.00001200    0.00431800
    C                  1.36414600   -1.21610700    0.00338000
    C                 -0.02240200   -1.21632400    0.00226400
    H                 -0.59131100    2.14539300   -0.00013100
    H                  1.90773300    2.16404900    0.00493600
    H                  1.90769600   -2.16410100    0.00542100
    H                 -0.59129100   -2.14538500    0.00041900
    N                  3.45592600    0.00001600    0.03587500
    H                  3.95679500    0.85870800   -0.14709400
    H                  3.95687100   -0.85864300   -0.14710100
    N                 -2.17251000   -0.00000200   -0.00193800
    O                 -2.74286400   -1.08297000   -0.00375000
    O                 -2.74283000    1.08298000   -0.00298300
    '''
    mol.basis = '6-31G*'
    mol.build()

    Func = 'PBE'

    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    mf_hf.kernel()

    # PySCF
    td = tdscf.TDA(mf_hf)
    print('PySCF所得激发能')
    td.nstates=6
    td.kernel()
    
    # Davidson对角化
    td = TDA(mf_hf,mol)
    td.kernel()
    print()

    # numpy的eigh函数
    A = td.TD_matrix
    x = time.time()
    Hartree_to_eV = 27.211385050
    val,ket = np.linalg.eigh(A)
    y = time.time()

    print(f"eigh计算完成，耗时{y-x:.6f}s")
    print(f'{val[:6]*Hartree_to_eV}')