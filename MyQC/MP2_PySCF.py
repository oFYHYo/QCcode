from pyscf import gto, scf
import numpy as np
import time
# 定义分子结构及基组
mol = gto.M(atom = 'H 0 0 0; Li 0 0 0.9', basis = 'sto-3g')
mol = mol.build()

# scf求解并获得分子轨道系数
print("开始进行RHF求解")
x = time.time()
HF = scf.RHF(mol).run()
mo = HF.mo_coeff
mo_ene = HF.mo_energy
eri = mol.intor("int2e")
y = time.time()
print(f"scf耗时{y-x:2.3f}s")

# 计算占据轨道数目
nelec = mol.nelectron
nocc = int(nelec/2)

# 将原子轨道积分变换为分子轨道积分
print("开始进行积分变换")
x = time.time()
eri_mo =np.einsum("up,vq,uvkl,kr,ls->pqrs",mo,mo,eri,mo,mo,optimize=True) 
y = time.time()
print(f"积分变换耗时{y-x:2.3f}s")

# 产生D，T，t矩阵
D_iajb = mo_ene[:nocc,None,None,None]+mo_ene[None,None,:nocc,None]-mo_ene[None,nocc:,None,None]-mo_ene[None,None,None,nocc:]
t_iajb = eri_mo[:nocc,nocc:,:nocc,nocc:]/D_iajb
T_iajb = 2*t_iajb-t_iajb.transpose(0,3,2,1)

# 计算MP2相关能
E_corr = (T_iajb*t_iajb*D_iajb).sum()
print(f"MP2相关能E_corr={E_corr:12.8f}")
