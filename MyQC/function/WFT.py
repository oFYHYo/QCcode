import numpy as np
from pyscf import gto
from scipy.linalg import eigh
import time

def solve_MP2(eri,mo,mo_ene,nocc):
    # 将原子轨道积分变换为分子轨道积分
    eri_mo =np.einsum("up,vq,uvkl,kr,ls->pqrs",mo,mo,eri,mo,mo,optimize=True) 

    # 产生D，T，t矩阵
    D_iajb = mo_ene[:nocc,None,None,None]+mo_ene[None,None,:nocc,None]-mo_ene[None,nocc:,None,None]-mo_ene[None,None,None,nocc:]
    t_iajb = eri_mo[:nocc,nocc:,:nocc,nocc:]/D_iajb
    T_iajb = 2*t_iajb-t_iajb.transpose(0,3,2,1)

    # 计算MP2相关能
    E_corr = (T_iajb*t_iajb*D_iajb).sum()
    print(f"MP2相关能E_corr: {E_corr:12.8f} Hartree")
    return E_corr

def solve_RHF(xyz,basis,charge=0):
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = basis
    mol.build()

    # 从PySCF中获取电子积分
    E_nuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp')
    V = mol.intor('int1e_nuc')
    T = mol.intor('int1e_kin')
    I = mol.intor('int2e')

    # 确定占据数
    nao = T.shape[0]
    nmo = nao
    nelec = mol.nelectron
    nocc = int(nelec/2)
    mo_occ = np.zeros(nmo, dtype=int)
    mo_occ[:nocc] = 2
    occ_list = np.where(mo_occ > 0)[0]

    # 密度矩阵初始猜测
    Dm = np.eye(T.shape[0])

    # 进行SCF迭代
    hcore = T+V
    E_err  = 1
    E_old  = 0
    E_occ  = 0
    item   = 1
    Dm_    = 1
    Dm_old = Dm
    noconv = E_err < (1e-8) or Dm_ < (1e-8)
    print("迭代次数    HF能量         能量误差           密度矩阵误差")
    while (E_err > (1e-8) or Dm_ > (1e-8)) and not item > 200:
        # 构造Fock矩阵
        Cul = np.einsum("pqrs,rs->pq",I, Dm_old)
        Exc = np.einsum("prqs,rs->pq",I, Dm_old)
        F = hcore+Cul-Exc*0.5
        E, C_ = eigh(F, S)
        C = C_[:,occ_list]
        Dm = 2*(C@(C.T))
        # 得到HF能量
        E_occ = 0.5 * np.einsum("ab,ab->", hcore + F, Dm)

        # 计算并打印迭代误差
        E_err = np.abs((E_occ-E_old))
        E_old = E_occ
        Dm_ = np.linalg.norm(Dm-Dm_old)
        Dm_old = Dm
        E_HF = E_occ+E_nuc
        print(f"{item:3d}     {E_HF: 12.6f}      {E_err: 12.8f}       {Dm_: 12.8f}")
        item += 1

    else:
        if item > 100:
            print("SCF不收敛，超过最大迭代次数")
        else:
            print("SCF 收敛")
            print(f"Hartree-Fock能量: {E_HF:.8f} Hartree")
    return E_HF,C_,E,I,nocc

E_HF,mo,mo_ene,eri,nocc = solve_RHF('''
 C                 -0.66295800    0.00000000   -0.00000000
 C                  0.66295800    0.00000000   -0.00000000
 H                 -1.25654334    0.92403753    0.00000000
 H                 -1.25654334   -0.92403753    0.00000000
 H                  1.25654334   -0.92403753    0.00000000
 H                  1.25654334    0.92403753   -0.00000000
''','def2-TZVP')

E_corr = solve_MP2(eri,mo,mo_ene,nocc)
print(f"总能量：{E_HF+E_corr:.8f} Hartree")

