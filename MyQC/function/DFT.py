from pyscf import gto,dft,lib
import numpy as np
from scipy.linalg import eigh
from functools import partial
import time

def solve_GGA(mol,Func):
    if dft.libxc.xc_type(Func) != 'GGA':
        print(f"{Func}不是GGA泛函")
    
    # 使用贪心算法(greedy算法)，并限制内存为4GB
    np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 4 / 8])

    x = time.time()

    # 获取电子积分与核排斥能
    E_nuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp')
    V = mol.intor('int1e_nuc')
    T = mol.intor('int1e_kin')
    eri = mol.intor('int2e')

    # 获取交换相关泛函及其杂化参数cx
    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    ni = dft.numint.NumInt()
    cx = ni.hybrid_coeff(Func)

    # 获取DFT积分格点及其权重
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    weights = grids.weights

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

    # 获取原子轨道格点值
    ao_1 = ni.eval_ao(mol,grids.coords,deriv=1)
    ao_0 = ao_1[0]

    # 生成电子密度的格点值
    def gen_rho(ao,Dm):
        rho_0 = np.einsum("gu,gv,uv->g",ao[0],ao[0],Dm)
        rho_1 = 2*np.einsum("gu,rgv,uv->rg",ao[0],ao[1:],Dm)
        rho_01 = 2*np.einsum("gu,rgv,uv->rg",ao[0],ao,Dm)  # rho_01 = [rho_0,rho_1] 所以这里的第一项要乘以0.5
        rho_01[0] *= 0.5
        return rho_0, rho_1, rho_01

    # 进行SCF迭代
    print("迭代次数    SCF能量         能量误差           密度矩阵误差")
    while E_err > 1e-8 and Dm_err > 1e-8 :
            
        # 获取电子密度格点值
        rho_0, rho_1, rho_01 = gen_rho(ao_1,Dm_old)

        # 获取泛函核的格点值
        exc, fr = ni.eval_xc(Func,rho_01,deriv=3)[:2]
        exc *= weights
        fg = fr[1]*weights
        fr = fr[0]*weights

        # 获取交换相关势, 用到了交换相关势对uv指标的对称性
        Fxc = (
        + 0.5 * np.einsum("g, gu, gv -> uv",fr, ao_0, ao_0)
        + 2 * np.einsum("g, rg, rgu, gv -> uv", fg, rho_1, ao_1[1:], ao_0)
        )
        Fxc = Fxc+ Fxc.swapaxes(-1,-2)

        # 计算各类矩阵
        K = np.einsum("prqs,rs->pq",eri,Dm_old)            # 交换积分
        J = np.einsum("pqrs,rs->pq",eri,Dm_old)            # 库伦积分
        hcore = T+V                                        # 单电子哈密顿量
        F_KS = hcore + J - 0.5*cx*K + Fxc                  # KS矩阵

        # 对KS矩阵进行广义对角化并得到新的密度矩阵
        # FC = SCE
        mo_ene, mo_c = eigh(F_KS, S)
        mo_c = mo_c[:,occ_list]
        Dm = 2*(mo_c@(mo_c.T))

        # 计算能量与误差
        E_KS = ((hcore + 0.5*J - 0.25*cx*K)* Dm).sum() + (exc*rho_0).sum()+ E_nuc
        E_err = np.abs(E_KS - E_old)
        Dm_err = np.linalg.norm(Dm-Dm_old)

        # 计算下一轮迭代所需量
        Dm_old = c*Dm + (1-c)*Dm_old                      # 密度矩阵混合

        E_old = E_KS
        item += 1
        print(f" {item:3d}     {E_KS:12.8f}      {E_err:12.8f}        {Dm_err:12.8f}")
        if item >= 100:
            print("SCF超过最大迭代次数")
            break
    else: 
        y = time.time()
        print("SCF成功收敛")
        print(f"DFT能量 = {E_KS:12.8f} Hartree, 耗时{y-x:4.4f}s")

    return E_KS, mo_c, mo_ene

def solve_LDA(mol,Func):
    if dft.libxc.xc_type(Func) != 'LDA':
        print(f"{Func}不是LDA泛函")
    
    # 使用贪心算法(greedy算法)，并限制内存为4GB
    np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 4 / 8])

    x = time.time()

    # 获取电子积分与核排斥能
    E_nuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp')
    V = mol.intor('int1e_nuc')
    T = mol.intor('int1e_kin')
    eri = mol.intor('int2e')

    # 获取交换相关泛函及其杂化参数cx
    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    ni = dft.numint.NumInt()
    cx = ni.hybrid_coeff(Func)

    # 获取DFT积分格点及其权重
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    weights = grids.weights

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

    # 获取原子轨道格点值
    ao_1 = ni.eval_ao(mol,grids.coords,deriv=1)
    ao_0 = ao_1[0]

    # 生成电子密度的格点值
    def gen_rho(ao,Dm):
        rho_0 = np.einsum("gu,gv,uv->g",ao[0],ao[0],Dm)
        return rho_0

    # 进行SCF迭代
    print("迭代次数    SCF能量         能量误差           密度矩阵误差")
    while E_err > 1e-8 and Dm_err > 1e-8 :
            
        # 获取电子密度格点值
        rho_0 = gen_rho(ao_1,Dm_old)

        # 获取泛函核的格点值
        exc, fr = ni.eval_xc(Func,rho_0,deriv=1)[:2]
        exc *= weights
        fr = fr[0]*weights

        # 获取交换相关势
        Fxc =  np.einsum("g, gu, gv -> uv",fr, ao_0, ao_0)

        # 计算各类矩阵
        K = np.einsum("prqs,rs->pq",eri,Dm_old)            # 交换积分
        J = np.einsum("pqrs,rs->pq",eri,Dm_old)            # 库伦积分
        hcore = T+V                                        # 单电子哈密顿量
        F_KS = hcore + J - 0.5*cx*K + Fxc                  # KS矩阵

        # 对KS矩阵进行广义对角化并得到新的密度矩阵
        # FC = SCE
        mo_ene, mo_c = eigh(F_KS, S)
        mo_c = mo_c[:,occ_list]
        Dm = 2*(mo_c@(mo_c.T))

        # 计算能量与误差
        E_KS = ((hcore + 0.5*J - 0.25*cx*K)* Dm).sum() + (exc*rho_0).sum()+ E_nuc
        E_err = np.abs(E_KS - E_old)
        Dm_err = np.linalg.norm(Dm-Dm_old)

        # 计算下一轮迭代所需量
        Dm_old = c*Dm + (1-c)*Dm_old                      # 密度矩阵混合

        E_old = E_KS
        item += 1
        print(f" {item:3d}     {E_KS:12.8f}      {E_err:12.8f}        {Dm_err:12.8f}")
        if item >= 100:
            print("SCF超过最大迭代次数")
            break
    else: 
        y = time.time()
        print("SCF成功收敛")
        print(f"DFT能量 = {E_KS:12.8f} Hartree, 耗时{y-x:4.4f}s")

    return E_KS, mo_c, mo_ene

def solve_mGGA(mol,Func):
    if dft.libxc.xc_type(Func) != 'MGGA':
        print(f"{Func}不是MGGA泛函")
    
    # 使用贪心算法(greedy算法)，并限制内存为4GB
    np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 4 / 8])

    x = time.time()

    # 获取电子积分与核排斥能
    E_nuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp')
    V = mol.intor('int1e_nuc')
    T = mol.intor('int1e_kin')
    eri = mol.intor('int2e')

    # 获取交换相关泛函及其杂化参数cx
    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    ni = dft.numint.NumInt()
    cx = ni.hybrid_coeff(Func)

    # 获取DFT积分格点及其权重
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    weights = grids.weights

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

    # 获取原子轨道格点值
    ao = ni.eval_ao(mol,grids.coords,deriv=1)
    ao_1 = ao[1:]
    ao_0 = ao[0]

    # 生成电子密度的格点值
    def gen_rho(ao,Dm):
        rho_0 = np.einsum("gu,gv,uv->g",ao[0],ao[0],Dm)
        rho_1 = 2*np.einsum("gu,rgv,uv->rg",ao[0],ao[1:],Dm)
        rho_01 = 2*np.einsum("gu,rgv,uv->rg",ao[0],ao,Dm)  # rho_01 = [rho_0,rho_1] 所以这里的第一项要乘以0.5
        rho_01[0] *= 0.5
        
        # 动能密度的计算, tau = 0.5*/sum|\nambla\psi|^2
        tau = 0.5*np.einsum('uv,kgu,kgv->g',Dm,ao_1,ao_1)

        rho_01t = np.vstack([rho_01,tau])
        return rho_0, rho_1, rho_01, rho_01t

    # 进行SCF迭代
    print("迭代次数    SCF能量         能量误差           密度矩阵误差")
    while E_err > 1e-8 and Dm_err > 1e-8 :
            
        # 获取电子密度格点值
        rho_0, rho_1, rho_01, rho_01t = gen_rho(ao,Dm_old)

        # 获取泛函核的格点值
        exc, fr = ni.eval_xc(Func,rho_01t,deriv=2)[:2]

        exc *= weights
        ft = fr[3]*weights   # 对动能密度的一阶导
        fg = fr[1]*weights   # 对密度梯度的一阶导
        fr = fr[0]*weights   # 对密度的一阶导

        # 获取交换相关势, 用到了交换相关势对uv指标的对称性
        Fxc = (
        + 0.5 * np.einsum("g, gu, gv -> uv",fr, ao_0, ao_0)
        + 2 * np.einsum("g, rg, rgu, gv -> uv", fg, rho_1, ao_1, ao_0)
        )
        Fxc = Fxc+ Fxc.swapaxes(-1,-2)

        Fxc += 0.5*np.einsum('g,kgu,kgv->uv',ft,ao_1,ao_1)   # 动能密度相关项

        '''' 比较构建的xc势与PySCF的xc势的差异
        xc_n,xc_e,xc_v = ni.nr_rks(mol, grids, "TPSS", Dm_old)
        print(np.allclose(xc_v,Fxc))
        '''

        # 计算各类矩阵
        K = np.einsum("prqs,rs->pq",eri,Dm_old)            # 交换积分
        J = np.einsum("pqrs,rs->pq",eri,Dm_old)            # 库伦积分
        hcore = T+V                                        # 单电子哈密顿量
        F_KS = hcore + J - 0.5*cx*K + Fxc                  # KS矩阵

        # 对KS矩阵进行广义对角化并得到新的密度矩阵
        # FC = SCE
        mo_ene, mo_c = eigh(F_KS, S)
        mo_c = mo_c[:,occ_list]
        Dm = 2*(mo_c@(mo_c.T))

        # 计算能量与误差
        E_KS = ((hcore + 0.5*J - 0.25*cx*K)* Dm).sum() + (exc*rho_0).sum()+ E_nuc
        E_err = np.abs(E_KS - E_old)
        Dm_err = np.linalg.norm(Dm-Dm_old)

        # 计算下一轮迭代所需量
        Dm_old = c*Dm + (1-c)*Dm_old                      # 密度矩阵混合

        E_old = E_KS
        item += 1
        print(f" {item:3d}     {E_KS:12.8f}      {E_err:12.8f}        {Dm_err:12.8f}")
        if item >= 100:
            print("SCF超过最大迭代次数")
            break
    else: 
        y = time.time()
        print("SCF成功收敛")
        print(f"DFT能量 = {E_KS:12.8f} Hartree, 耗时{y-x:4.4f}s")

    return E_KS, mo_c, mo_ene

def solve_DFT(mol,Func):
    if dft.libxc.xc_type(Func) == 'LDA':
        E_KS,mo_c,mo_ene = solve_LDA(mol,Func)
    elif dft.libxc.xc_type(Func) == 'GGA':
        E_KS,mo_c,mo_ene = solve_GGA(mol,Func)
    elif dft.libxc.xc_type(Func) == 'MGGA':
        E_KS,mo_c,mo_ene = solve_mGGA(mol,Func)
    else:
        print('ERROR FUNCTION')

if __name__ == '__main__':
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

    Func = 'TPSS'
    print(dft.libxc.xc_type(Func))

    solve_DFT(mol,Func)
    mf_hf = dft.RKS(mol)
    mf_hf.xc = Func
    mf_hf.kernel()
