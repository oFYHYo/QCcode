import numpy as np
from pyscf import gto,scf
import time
from functools import partial

np.einsum = partial(np.einsum,optimize=True)

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

nelec = mol.nelectron
nocc = int(nelec/2)

HF = scf.RHF(mol).run()
mo = HF.mo_coeff
mo_ene = HF.mo_energy
eri = mol.intor("int2e")

eri_mo =np.einsum("up,vq,uvkl,kr,ls->pqrs",mo,mo,eri,mo,mo)

ant_eri_mo = eri_mo.transpose(0,2,1,3) - eri_mo.transpose(0,2,3,1)

mo_oovv = ant_eri_mo[:nocc,:nocc,nocc:,nocc:]
mo_oooo = ant_eri_mo[:nocc,:nocc,:nocc,:nocc]
mo_vvvv = ant_eri_mo[nocc:,nocc:,nocc:,nocc:]
mo_ovvo = ant_eri_mo[:nocc,nocc:,nocc:,:nocc]
mo_vvoo = ant_eri_mo[nocc:,nocc:,:nocc,:nocc]

D_oovv = mo_ene[:nocc,None,None,None]-mo_ene[None,None,nocc:,None]+mo_ene[None,:nocc,None,None]-mo_ene[None,None,None,nocc:]
D_vvoo = -mo_ene[nocc:,None,None,None]+mo_ene[None,None,:nocc,None]-mo_ene[None,nocc:,None,None]+mo_ene[None,None,None,:nocc]
def gen_taub(t_oovv):
    return t_oovv

def gen_tau(t_oovv):
    return t_oovv

def gen_Fvv(mo_oovv,taub_oovv):
    return -0.5*np.einsum("mnaf,mnbf->ab",taub_oovv,mo_oovv)

def gen_Foo(mo_oovv,taub_oovv):
    return 0.5*np.einsum("inef,jnef->ij",taub_oovv,mo_oovv)

def gen_Woooo(mo_oooo,mo_oovv,tau_oovv):
    return mo_oooo + 0.25 * np.einsum("ijef,mnef->mnij",tau_oovv,mo_oovv)

def gen_Wvvvv(mo_vvvv,mo_oovv,tau_oovv):
    return mo_vvvv + 0.25 * np.einsum("mnab,mnef->abef",tau_oovv,mo_oovv)

def gen_Wovvo(mo_ovvo,mo_oovv,tau_oovv):
    return mo_ovvo - 0.25 * np.einsum("jnfb,mnef->mbej",tau_oovv,mo_oovv)

def gen_eqn(mo_vvoo,Foo,Fvv,Woooo,Wovvo,Wvvvv,t_oovv,tau_oovv):

    def P(A,a,b):
        sort = []
        
        for i in range(A.ndim):
            if i == a:
                i = b
            elif i == b:
                i = a
            sort.append(i)
        
        return A - np.transpose(A,sort)
    
    eqn = mo_vvoo.transpose(2,3,0,1)
    
    eqn += P(np.einsum("ijae,be->ijab",t_oovv,Fvv),2,3)
    
    eqn -= P(np.einsum("imab,mj->ijab",t_oovv,Foo),0,1)
    
    eqn += 0.5*np.einsum("mnab,mnij->ijab",tau_oovv,Woooo)
    
    eqn += 0.5*np.einsum("ijef,abef->ijab",tau_oovv,Wvvvv)
    
    A = P(np.einsum("imae,mbej->ijab",t_oovv,Wovvo),2,3)
    eqn += P(A,0,1)

    return eqn

max_iter = 5
iter = 0
'''
t2_old = np.zeros_like(mo_oovv)
'''
t2_old = mo_vvoo/D_vvoo
t2_old = t2_old.transpose(2,3,0,1)
print(0.25*np.einsum("ijab,ijab->",mo_oovv,t2_old))

D_iajb = mo_ene[:nocc,None,None,None]+mo_ene[None,None,:nocc,None]-mo_ene[None,nocc:,None,None]-mo_ene[None,None,None,nocc:]
t_iajb = eri_mo[:nocc,nocc:,:nocc,nocc:]/D_iajb
T_iajb = 2*t_iajb-t_iajb.transpose(0,3,2,1)

print((T_iajb*t_iajb*D_iajb).sum())

for i in range(max_iter):
    taub = gen_taub(t2_old)
    tau = gen_tau(t2_old)
    Fvv = gen_Fvv(mo_oovv,taub)
    Foo = gen_Foo(mo_oovv,taub)
    Woooo = gen_Woooo(mo_oooo,mo_oovv,tau)
    Wovvo = gen_Wovvo(mo_ovvo,mo_oovv,tau)
    Wvvvv = gen_Wvvvv(mo_vvvv,mo_oovv,tau)

    eqn = gen_eqn(mo_vvoo,Foo,Fvv,Woooo,Wovvo,Wvvvv,t2_old,tau)
    
    t2_new = eqn/D_oovv
    
    iter += 1
    E_CCD = 0.25*np.einsum("ijab,ijab->",mo_oovv,t2_new) 
    print(iter,np.linalg.norm(t2_new-t2_old),E_CCD)
    tol = 1e-8
    if np.linalg.norm(t2_new-t2_old) < tol:
        print("CCD计算完毕")
        break
    
    t2_old = t2_new.copy()
    



