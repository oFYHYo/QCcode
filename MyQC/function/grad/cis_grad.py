import numpy as np
from pyscf import gto,scf,tdscf,grad,ao2mo
import itertools
from hf_grad import *
from scipy.sparse.linalg import LinearOperator,gmres
from scipy.optimize import newton_krylov

einsum = np.einsum

def calc_cis_grad(cis,x):

    mf = cis._scf
    nocc = mf.mol.nelectron//2
    mo_c = mf.mo_coeff
    Co = mf.mo_coeff[:,:nocc]
    Cv = mf.mo_coeff[:,nocc:]
    nmo = mo_c.shape[1]
    mo_ene = mf.mo_energy
    dm = mf.make_rdm1()
    
    '''
    计算单电子密度矩阵
    '''

    # 计算弛豫密度矩阵的对角元
    P_delta_mo = np.zeros_like(dm)
    P_delta_mo[:nocc,:nocc] = -einsum('ia,ja->ij',x,x)*2
    P_delta_mo[nocc:,nocc:] = einsum('ia,ib->ab',x,x)*2

    # 计算弛豫密度矩阵的非对角元

    z = CPHF(mf,x,P_delta_mo)
    P_delta_mo[nocc:,:nocc] = z*2

    opdm = dm + einsum('pq,up,vq->uv',P_delta_mo,mo_c,mo_c)
    # print(np.allclose(opdm,dmz1doo),np.linalg.norm(opdm-dmz1doo))
    #opdm = dmz1doo
    '''
    计算双电子密度矩阵
    '''
    T = einsum('ia,ui,va->uv',x,Co,Cv)*2
    tmp = 2*opdm - dm
    tpdm = (einsum('uv,kl->uvkl',tmp,dm) + 2*einsum('uv,kl->uvkl',T,T)
            )*0.25
    
    '''
    计算权重密度矩阵
    '''
    eri = mf._eri
    eri_mo = ao2mo.kernel(eri,mo_c,compact=False)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    eri_mo = eri_mo.transpose(0,2,1,3)
    ant_eri_mo = 2*eri_mo - eri_mo.transpose(0,1,3,2)

    wdm = calc_wdm(mf)

    wdm_delta = np.zeros_like(wdm)
    wdm_delta[:nocc,:nocc] -= einsum('pq,plqk->kl',P_delta_mo,ant_eri_mo[:,:nocc,:,:nocc])
    wdm_delta[:nocc,:nocc] -= einsum('ia,lb,akib->kl',x,x,ant_eri_mo[nocc:,:nocc,:nocc,nocc:])*2
    wdm_delta[:nocc,:nocc] -= einsum('kl,l->kl',P_delta_mo[:nocc,:nocc],mo_ene[:nocc])

    wdm_delta[nocc:,nocc:] -= einsum('ac,c->ac',P_delta_mo[nocc:,nocc:],mo_ene[nocc:])
    wdm_delta[nocc:,nocc:] -= einsum('ia,jb,jcbi->ab',x,x,ant_eri_mo[:nocc,nocc:,nocc:,:nocc])*2
    
    wdm_delta[nocc:,:nocc] -= einsum('ak,k->ak',P_delta_mo[nocc:,:nocc],mo_ene[:nocc])*2
    wdm_delta[nocc:,:nocc] -= einsum('ia,jb,jkbi->ak',x,x,ant_eri_mo[:nocc,:nocc,nocc:,:nocc])*4


    # zeta = (mo_ene[:,None] + mo_ene[None,:]) * .5
    # zeta[nocc:,:nocc] = mo_ene[:nocc]
    # zeta[:nocc,nocc:] = mo_ene[nocc:]
    #dm1 = P_delta_mo.copy()
    #dm1[:nocc,:nocc] += np.eye(nocc)*2 # for ground state
    #wdm_delta -= dm1*zeta
    
    wdm = wdm + einsum('pq,up,vq->uv',wdm_delta,mo_c,mo_c)

    #print(np.allclose(tmp1,wdm1),np.linalg.norm(tmp1-wdm1),(wdm1)[nocc:,:nocc])
    #wdm = wdm1
    

    '''
    计算CIS梯度
    '''
    h1eaox = calc_hcore_grad_ao(mol)
    h2eaox = calc_eri_grad_ao(mol)
    h2eaox_sym = 2*h2eaox - h2eaox.swapaxes(-2,-3)
    ovlpx = calc_ovlp_grad_ao(mol)
    
    grad = ( einsum('uv,Axuv->Ax',opdm,h1eaox)
           + einsum('uvkl,Axuvkl->Ax',tpdm,h2eaox_sym)
           + einsum('uv,Axuv->Ax',wdm,ovlpx)
           + grad_nuc(mol)
            )


    return grad


def CPHF(mf,x,P_delta_mo):
    '''
    计算CPHF方程
    '''
    
    nocc = mf.mol.nelectron//2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    mo_c = mf.mo_coeff
    mo_ene = mf.mo_energy
    eai = mo_ene[nocc:,None] - mo_ene[None,:nocc]
    
    eri = mf._eri
    eri_mo = ao2mo.kernel(eri,mo_c,compact=False)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    eri_mo = eri_mo.transpose(0,2,1,3)
    ant_eri_mo = 2*eri_mo - eri_mo.transpose(0,1,3,2)

    L = ( einsum('ab,acbk->ck',P_delta_mo[nocc:,nocc:],ant_eri_mo[nocc:,nocc:,nocc:,:nocc])
         + einsum('ij,icjk->ck',P_delta_mo[:nocc,:nocc],ant_eri_mo[:nocc,nocc:,:nocc,:nocc])
         + einsum('ia,kb,cabi->ck',x,x,ant_eri_mo[nocc:,nocc:,nocc:,:nocc])*2
         - einsum('ic,jb,jkbi->ck',x,x,ant_eri_mo[:nocc,:nocc,nocc:,:nocc])*2
        )

    def vind(z):
        # 计算CPHF方程的右端项
        z = z.reshape(nvir,nocc)
        
        left = ( einsum('ai,ai->ai',eai,z)
               + einsum('bj,abij->ai',z,ant_eri_mo[nocc:,nocc:,:nocc,:nocc])
               + einsum('bj,ajib->ai',z,ant_eri_mo[nocc:,:nocc,:nocc,nocc:]))

        return left+L
    
    
    z = newton_krylov(vind,np.zeros((nvir, nocc)), verbose=False)
    z = z.reshape(nvir,nocc)

    return z


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = ''' 
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
    '''

    mol.basis = 'sto-3g'
    #mol.verbose = 6
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    
    cis = tdscf.rhf.CIS(mf)
    cis.nstates = 10
    cis.kernel()

    atom = []
    for i in range(mol.natm):
        atom.append(mol.atom_symbol(i))

    k = 1
    cis_grad = cis.Gradients()
    cis_grad.state = k
    cis_grad.kernel()

    X = []
    for i,x in enumerate(cis.xy):
        X.append(x[0])
    X = np.array(X)

    mf = cis._scf
    x = X[k-1]
    dm = mf.make_rdm1()
    nocc = mf.mol.nelectron//2
    
    # from cis_grad_test import grad_elec
    # dmz1doo,wdm1 = grad_elec(cis_grad, cis.xy[k-1], singlet=True)

    grad = calc_cis_grad(cis,x)
    print(f'--------- My CIS gradients for state {k} ----------')
    for i,g in enumerate(grad):
        print(f'{i:1} {atom[i]:3} {g[0]:12.8f} {g[1]:12.8f} {g[2]:12.8f}')


