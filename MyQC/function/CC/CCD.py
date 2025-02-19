import numpy as np
from pyscf import gto,scf
import time
from functools import partial

einsum = partial(np.einsum,optimize=True)

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

eri_mo =einsum("up,vq,uvkl,kr,ls->pqrs",mo,mo,eri,mo,mo)

ant_eri_mo = (eri_mo.transpose(0,2,1,3) - eri_mo.transpose(0,2,3,1))[nocc:,nocc:,:nocc,:nocc]
fock = HF.get_fock()
fock_mo = einsum('ip,pq,qj',mo.T,fock,mo)[nocc:,:nocc]

def CCSD_T1(old_t1,old_t2,fock,ant_eri_mo):
    t1 = old_t1
    t2 = old_t2
    f = fock
    I = ant_eri_mo

    T1 = 1.0*einsum('ai->ai', f)
    T1 += -1.0*einsum('ji,aj->ai', f, t1)
    T1 += 1.0*einsum('ab,bi->ai', f, t1)
    T1 += -1.0*einsum('jb,abji->ai', f, t2)
    T1 += -1.0*einsum('jaib,bj->ai', I, t1)
    T1 += 0.5*einsum('jkib,abkj->ai', I, t2)
    T1 += -0.5*einsum('jabc,cbji->ai', I, t2)
    T1 += -1.0*einsum('jb,bi,aj->ai', f, t1, t1)
    T1 += -1.0*einsum('jkib,aj,bk->ai', I, t1, t1)
    T1 += 1.0*einsum('jabc,ci,bj->ai', I, t1, t1)
    T1 += -0.5*einsum('jkbc,aj,cbki->ai', I, t1, t2)
    T1 += -0.5*einsum('jkbc,ci,abkj->ai', I, t1, t2)
    T1 += 1.0*einsum('jkbc,cj,abki->ai', I, t1, t2)
    T1 += 1.0*einsum('jkbc,ci,aj,bk->ai', I, t1, t1, t1)
    
    return T1

def CCSD_T2(old_t1,old_t2,fock,ant_eri_mo):
    f = fock
    I = ant_eri_mo
    t1 = old_t1
    t2 = old_t2

    T2  = 1.0*einsum('baji->abij', I)
    T2 += 1.0*einsum('ki,bakj->abij', f, t2)
    T2 += -1.0*einsum('kj,baki->abij', f, t2)
    T2 += 1.0*einsum('ac,bcji->abij', f, t2)
    T2 += -1.0*einsum('bc,acji->abij', f, t2)
    T2 += -1.0*einsum('kaji,bk->abij', I, t1)
    T2 += 1.0*einsum('kbji,ak->abij', I, t1)
    T2 += -1.0*einsum('baic,cj->abij', I, t1)
    T2 += 1.0*einsum('bajc,ci->abij', I, t1)
    T2 += -0.5*einsum('klji,balk->abij', I, t2)
    T2 += 1.0*einsum('kaic,bckj->abij', I, t2)
    T2 += -1.0*einsum('kajc,bcki->abij', I, t2)
    T2 += -1.0*einsum('kbic,ackj->abij', I, t2)
    T2 += 1.0*einsum('kbjc,acki->abij', I, t2)
    T2 += -0.5*einsum('bacd,dcji->abij', I, t2)
    T2 += -1.0*einsum('kc,ak,bcji->abij', f, t1, t2)
    T2 += 1.0*einsum('kc,bk,acji->abij', f, t1, t2)
    T2 += 1.0*einsum('kc,ci,bakj->abij', f, t1, t2)
    T2 += -1.0*einsum('kc,cj,baki->abij', f, t1, t2)
    T2 += 1.0*einsum('klji,bk,al->abij', I, t1, t1)
    T2 += 1.0*einsum('kaic,cj,bk->abij', I, t1, t1)
    T2 += -1.0*einsum('kajc,ci,bk->abij', I, t1, t1)
    T2 += -1.0*einsum('kbic,cj,ak->abij', I, t1, t1)
    T2 += 1.0*einsum('kbjc,ci,ak->abij', I, t1, t1)
    T2 += 1.0*einsum('bacd,di,cj->abij', I, t1, t1)
    T2 += 1.0*einsum('klic,ak,bclj->abij', I, t1, t2)
    T2 += -1.0*einsum('klic,bk,aclj->abij', I, t1, t2)
    T2 += 0.5*einsum('klic,cj,balk->abij', I, t1, t2)
    T2 += -1.0*einsum('klic,ck,balj->abij', I, t1, t2)
    T2 += -1.0*einsum('kljc,ak,bcli->abij', I, t1, t2)
    T2 += 1.0*einsum('kljc,bk,acli->abij', I, t1, t2)
    T2 += -0.5*einsum('kljc,ci,balk->abij', I, t1, t2)
    T2 += 1.0*einsum('kljc,ck,bali->abij', I, t1, t2)
    T2 += 0.5*einsum('kacd,bk,dcji->abij', I, t1, t2)
    T2 += -1.0*einsum('kacd,di,bckj->abij', I, t1, t2)
    T2 += 1.0*einsum('kacd,dj,bcki->abij', I, t1, t2)
    T2 += -1.0*einsum('kacd,dk,bcji->abij', I, t1, t2)
    T2 += -0.5*einsum('kbcd,ak,dcji->abij', I, t1, t2)
    T2 += 1.0*einsum('kbcd,di,ackj->abij', I, t1, t2)
    T2 += -1.0*einsum('kbcd,dj,acki->abij', I, t1, t2)
    T2 += 1.0*einsum('kbcd,dk,acji->abij', I, t1, t2)
    T2 += 0.5*einsum('klcd,adji,bclk->abij', I, t2, t2)
    T2 += -1.0*einsum('klcd,adki,bclj->abij', I, t2, t2)
    T2 += -0.5*einsum('klcd,baki,dclj->abij', I, t2, t2)
    T2 += -0.5*einsum('klcd,bdji,aclk->abij', I, t2, t2)
    T2 += 1.0*einsum('klcd,bdki,aclj->abij', I, t2, t2)
    T2 += 0.25*einsum('klcd,dcji,balk->abij', I, t2, t2)
    T2 += -0.5*einsum('klcd,dcki,balj->abij', I, t2, t2)
    T2 += -1.0*einsum('klic,cj,bk,al->abij', I, t1, t1, t1)
    T2 += 1.0*einsum('kljc,ci,bk,al->abij', I, t1, t1, t1)
    T2 += -1.0*einsum('kacd,di,cj,bk->abij', I, t1, t1, t1)
    T2 += 1.0*einsum('kbcd,di,cj,ak->abij', I, t1, t1, t1)
    T2 += -1.0*einsum('klcd,ak,dl,bcji->abij', I, t1, t1, t2)
    T2 += -0.5*einsum('klcd,bk,al,dcji->abij', I, t1, t1, t2)
    T2 += 1.0*einsum('klcd,bk,dl,acji->abij', I, t1, t1, t2)
    T2 += -1.0*einsum('klcd,di,ak,bclj->abij', I, t1, t1, t2)
    T2 += 1.0*einsum('klcd,di,bk,aclj->abij', I, t1, t1, t2)
    T2 += -0.5*einsum('klcd,di,cj,balk->abij', I, t1, t1, t2)
    T2 += 1.0*einsum('klcd,di,ck,balj->abij', I, t1, t1, t2)
    T2 += 1.0*einsum('klcd,dj,ak,bcli->abij', I, t1, t1, t2)
    T2 += -1.0*einsum('klcd,dj,bk,acli->abij', I, t1, t1, t2)
    T2 += -1.0*einsum('klcd,dj,ck,bali->abij', I, t1, t1, t2)
    T2 += 1.0*einsum('klcd,di,cj,bk,al->abij', I, t1, t1, t1, t1)
    
    return T2

def CCSD_E(t1,t2,fock,ant_eri_mo):
    f = fock
    I = ant_eri_mo

    E  = 1.0*einsum('ia,ai->', f, t1)
    E += 0.25*einsum('ijab,baji->', I, t2)
    E += -0.5*einsum('ijab,bi,aj->', I, t1, t1)

    return E

def update(old_t1,old_t2,fock,ant_eri_mo):

    E = CCSD_E(old_t1,old_t2,fock,ant_eri_mo)
    t1 = CCSD_T1(old_t1,old_t2,fock,ant_eri_mo)
    t2 = CCSD_T2(old_t1,old_t2,fock,ant_eri_mo)

    return t1,t2,E

old_t1 = np.zeros_like(fock_mo)
old_t2 = np.zeros_like(eri_mo)
old_E = 1
for i in range(100):
    t1,t2,E = update(old_t1,old_t2,fock_mo,ant_eri_mo)
    print(i,E)
    
    if abs(old_E-E) < 1.e-8:
        break
    
    old_E = E.copy()
    old_t1 = t1.copy()
    old_t2 = t2.copy()