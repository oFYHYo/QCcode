from pyscf import gto,dft,scf
import numpy as np
from scipy.linalg import eigh
import time
from functools import partial
# 使用贪心算法(greedy算法)，并限制内存为4GB
np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 4 / 8])

x = time.time()
# 定义原子坐标和基组
mol = gto.Mole()
mol.atom = '''
 C                 -0.66295800    0.00000000   -0.00000000
 C                  0.66295800    0.00000000   -0.00000000
 H                 -1.25654334    0.92403753    0.00000000
 H                 -1.25654334   -0.92403753    0.00000000
 H                  1.25654334   -0.92403753    0.00000000
 H                  1.25654334    0.92403753   -0.00000000
'''
mol.basis = '6-31G*'
mol.build()

# 获取电子积分与核排斥能
E_nuc = mol.energy_nuc()
S = mol.intor('int1e_ovlp')
V = mol.intor('int1e_nuc')
T = mol.intor('int1e_kin')
eri = mol.intor('int2e')

# 获取交换相关泛函及其杂化参数cx
Func = 'B3LYP'
mf_hf = dft.RKS(mol)
mf_hf.xc = Func
ni = dft.numint.NumInt()
cx = ni.hybrid_coeff(Func)

# 获取DFT积分格点
grids = dft.gen_grid.Grids(mol)
grids.build()

# 确定各占据数
nmo = mol.nao
mo_occ = np.zeros(nmo, dtype=int)
nocc = mol.nelec[0]
mo_occ[:nocc] = 2
occ_list = np.where(mo_occ > 0)[0]

# 初始化
Dm = np.eye(T.shape[0])
E_err = 1
Dm_err = 1
item = 0
Dm_old = Dm
E_old = 0
c = 0.5

# 进行SCF迭代
print("迭代次数    SCF能量         能量误差           密度矩阵误差")
while E_err > 1e-8 and Dm_err > 1e-8 :
    # 计算各类矩阵
    Fxc = ni.nr_rks(mol, grids, Func, Dm_old)[2]      # 交换相关势
    K = np.einsum("prqs,rs->pq",eri,Dm_old)           # 交换积分
    J = np.einsum("pqrs,rs->pq",eri,Dm_old)           # 库伦积分
    hcore = T+V                                       # 单电子哈密顿量
    F_KS = hcore + J - 0.5*cx*K +Fxc                  # KS矩阵

    # 对KS矩阵进行广义对角化并得到新的密度矩阵
    # FC = SCE
    mo_ene, mo_c = eigh(F_KS, S)
    mo_c = mo_c[:,occ_list]
    Dm = 2*(mo_c@(mo_c.T))
    
    # 计算能量与误差
    E_KS = ((hcore + 0.5*J - 0.25*cx*K)* Dm).sum() + ni.nr_rks(mol,grids,Func,Dm)[1]  + E_nuc
    E_err = np.abs(E_KS - E_old)
    Dm_err = np.linalg.norm(Dm-Dm_old)

    # 计算下一轮迭代所需量
    Dm_old = c*Dm + (1-c)*Dm_old                      # 密度矩阵混合
    E_old = E_KS
    item += 1
    print(f" {item:3d}     {E_KS:12.8f}      {E_err:12.8f}        {Dm_err:12.8f}")
    if item >= 500:
        print("SCF超过最大迭代次数")
        break
else: 
    print("SCF成功收敛")
    y = time.time()
    print(f"DFT能量 = {E_KS:12.8f} Hartree, 耗时{y-x:4.4f}s")





