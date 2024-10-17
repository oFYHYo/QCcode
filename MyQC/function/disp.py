from pyscf import gto,dft
import numpy as np
from functools import partial
from math import pi
import time

# 使用贪心算法(greedy算法)，并限制内存为4GB
np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 4 / 8])

def E_VV10(rho_01,b,C,grids):
    weights = grids.weights
    rho_0 = rho_01[0]*weights
    rho_1 = rho_01[1:3]*weights

    def beta(b):
        print('开始计算beta')
        return ((3/b**2)**0.75)/32
    
    def omega(rho_01,C):
        print('开始计算omega')
        grad2 = (rho_1**2).sum(0)
        return (C*(grad2**2/rho_01[0]**4)+4*pi*rho_01[0]/3)**0.5
    
    def kapa(b,rho_0):
        print('开始计算kapa')
        kapa = b*1.5*pi*(rho_0/(9*pi))**(1/6)
        return kapa
    
    def R2(grids):
        print('开始计算R2')
        xyz = grids.coords
        r = ((xyz**2).sum(1))**0.5
        n = r.shape[0]

        print('开始计算R2')

        R2 = np.zeros([r.shape[0],r.shape[0]])
        x = time.time()
        R2 = (r[:,None]-r[None,:])**2
        y = time.time()
        print(y-x)
        return R2 + R2.transpose(1,0)
    
    def g(rho_01,C,b,grids):
        print('开始计算g')
        return omega(rho_01,C)*R2(grids)+kapa(b,rho_01[0])
    
    g = g(rho_01,C,b,grids)
    assert g.shape[0] == rho_0.shape[0]

    print('开始计算integ')
    phi = 1/((g*g.transpose(1,0)*(g[:,None]+g[None,:])))
    integ = 1

    print('开始计算E_VV10')
    return ((beta(b) + 0.5*integ)*rho_0*weights).sum()



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

# 获取交换相关泛函及其杂化参数cx
Func = 'TPSS'
mf_hf = dft.RKS(mol)
mf_hf.xc = Func
ni = dft.numint.NumInt()
cx = ni.hybrid_coeff(Func)

# 获取DFT积分格点及其权重
grids = dft.gen_grid.Grids(mol)
grids.build()
weights = grids.weights

# 获取原子轨道格点值
ao_1 = ni.eval_ao(mol,grids.coords,deriv=1)
ao_0 = ao_1[0]
print(ao_1.shape)

# 生成电子密度的格点值
def gen_rho(ao,Dm) :
    ao_0 = ao[0]
    ao_1 = ao[1:]

    rho_0 = np.einsum("gu,gv,uv->g",ao_0,ao_0,Dm)
    rho_1 = 2*np.einsum("gu,rgv,uv->rg",ao_0,ao_1,Dm)
    rho_01 = np.vstack([rho_0,rho_1])
    '''
    rho_01 = 2*np.einsum("gu,rgv,uv->rg",ao_0,ao,Dm)  # rho_01 = [rho_0,rho_1] 所以这里的第一项要乘以0.5
    rho_01[0] *= 0.5
    '''
    
    tau = 0.5*np.einsum('uv,kgu,kgv->g',Dm,ao_1,ao_1)
    rho_01t = np.vstack([rho_01,tau])
    return rho_0, rho_1, rho_01, rho_01t

mf_hf.kernel()
Dm = mf_hf.make_rdm1()
mo_c = mf_hf.mo_coeff
nocc = mol.nelec[0]
rho_01t = gen_rho(ao_1,Dm)[-1]
rho_0 = rho_01t[0]
rho = ni.eval_rho(mol,ao_1,Dm,xctype = 'mGGA',with_lapl=False)

print(rho_01t.shape)
print(np.allclose(rho[-1],rho_01t[-1]))
print(rho[-1]-rho_01t[-1])

exc, fr = ni.eval_xc(Func,rho_01t,deriv=3)[:2]
E_xc = (exc*rho_01t[0]*weights).sum()
xc_n, xc_e, xc_v = ni.nr_rks(mol, grids, "TPSS", Dm)
print(f'Exc_PySCF = {xc_e}')
print(f'Exc = {E_xc}')

# print(E_VV10(rho_01,5.9,0.0083,grids))

