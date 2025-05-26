import numpy as np
from pyscf import gto,scf,tdscf,grad,ao2mo
import itertools
from hf_grad import *
from scipy.sparse.linalg import LinearOperator,gmres

einsum = np.einsum

def calc_two_part_dm(cis,x,state=1):
    '''
    计算双电子密度矩阵
    '''
    k = state
    mf = cis._scf
    nocc = mf.mol.nelectron//2
    Co = mf.mo_coeff[:,:nocc]
    Cv = mf.mo_coeff[:,nocc:]
    
    dm = mf.make_rdm1()
    T = einsum('ia,ui,va->uv',x[k-1],Co,Cv)
    opdm = calc_one_part_dm(cis,x,state)
    tpdm = (einsum('uv,kl->uvkl',opdm,dm) - einsum('ul,kv->uvkl',opdm,dm)
            + 2*einsum('uv,kl->uvkl',T,T) - 2*einsum('ul,kv->uvkl',T,T))*.5
    
    return tpdm

def calc_one_part_dm(cis,x,state=1,calc_delta_dmmo=False):
    '''
    计算单电子密度矩阵
    '''
    k = state
    mf = cis._scf
    nocc = mf.mol.nelectron//2
    mo_coeff = mf.mo_coeff
    
    dm = mf.make_rdm1()

    # CIS单电子密度矩阵对角元的计算
    P_delta_mo = np.zeros_like(dm)
    P_delta_mo[:nocc,:nocc] = -einsum('ia,ja->ij',x,x)*2
    P_delta_mo[nocc:,nocc:] =  einsum('ia,ib->ab',x,x)*2

    # 非对角元的计算
    import cis_grad_new 
    z = cis_grad_new.CPHF(mf,x,P_delta_mo)
    return z
    P_delta_mo[nocc:,:nocc] = z
    P_delta_mo[:nocc,nocc:] = z.T
    
    
    if calc_delta_dmmo == True:
        return P_delta_mo

    opdm = dm + einsum('pq,up,vq->uv',P_delta_mo,mo_coeff,mo_coeff)
    
    return opdm

def calc_weight_dm(cis,x,state=1):
    '''
    计算权重密度矩阵
    '''
    k = state
    mf = cis._scf
    nocc = mf.mol.nelectron//2
    mo_c = mf.mo_coeff
    nmo = mo_c.shape[1]
    mo_ene = mf.mo_energy
    
    dm = mf.make_rdm1()

    wdm = calc_wdm(mf)
    wdm_delta = np.zeros_like(dm)
    Pdelta_mo = calc_one_part_dm(cis,x,state=k,calc_delta_dmmo=True)

    h2e = mf._eri
    h2e_mo = ao2mo.kernel(h2e, mo_c, compact=False)
    h2e_mo = h2e_mo.reshape(nmo, nmo, nmo, nmo)
    h2e_mo_sym = 2*h2e_mo - h2e_mo.swapaxes(-2,-3)

    b = -einsum('ia,aikb->kb',x[k-1],h2e_mo_sym[nocc:,:nocc,:nocc,nocc:])
    S1 = 2*einsum('jb,kb->jk',x[k-1],b)
    S2 = 2*einsum('ia,ic->ac',x[k-1],b)
    print(np.allclose(S2,S2.T))
 
    wdm_delta[:nocc,:nocc] -= einsum('ji,i->ij',Pdelta_mo[:nocc,:nocc],mo_ene[:nocc])
    wdm_delta[:nocc,:nocc] += S1.T
    wdm_delta[:nocc,:nocc] -= einsum('pq,pqji->ij',Pdelta_mo,h2e_mo_sym[:,:,:nocc,:nocc])

    wdm_delta[nocc:,nocc:] += einsum('ba,a->ab',Pdelta_mo[nocc:,nocc:],mo_ene[nocc:])
    wdm_delta[nocc:,nocc:] += S2.T

    wdm_delta[nocc:,:nocc] -= 2*einsum('ia,jb,kijb->ak',x[k-1],x[k-1],h2e_mo_sym[:nocc,:nocc,:nocc,nocc:])
    wdm_delta[nocc:,:nocc] -=  einsum('ai,i->ai',Pdelta_mo[nocc:,:nocc],mo_ene[:nocc])

    wdm_delta[:nocc,nocc:] += wdm_delta[nocc:,:nocc].T

    wdm += np.einsum('ui,ij,vj->uv',mo_c,wdm_delta,mo_c)

    return wdm

def calc_cis_grad(cis,x,state=1):
    '''
    计算CIS梯度
    '''
    k = state
    mf = cis._scf

    h1ao = calc_hcore_grad_ao(mf.mol)
    h2eao = calc_eri_grad_ao(mf.mol)
    ovlp = calc_ovlp_grad_ao(mf.mol)
    

    opdm = calc_one_part_dm(cis,x,state=k)
    tpdm = calc_two_part_dm(cis,x,state=k)
    wdm = calc_weight_dm(cis,x,state=k)
    grad = (
        einsum('Axpq,pq->Ax',h1ao,opdm)+
        einsum('Axpqrs,pqrs->Ax',h2eao,tpdm)+
        einsum('Axpq,pq->Ax',ovlp,wdm) + 
        grad_nuc(mf.mol))

    return grad


def CPHF(mf,x,P_delta_mo,state=1):
    '''
    计算CPHF方程
    '''
    k = state
    x = x[k-1]
    mol = mf.mol
    
    mo_ene = mf.mo_energy

    h2e = mf._eri
    mo_c = mf.mo_coeff
    nmo = mo_c.shape[1]
    nocc = mol.nelectron//2

    nvir = nmo - nocc
    
    h2e_mo = ao2mo.kernel(h2e, mo_c, compact=False)
    h2e_mo = h2e_mo.reshape(nmo, nmo, nmo, nmo)
    h2e_mo_sym = 2*h2e_mo - h2e_mo.swapaxes(-2,-3)

    

    def A_matvec(P):
        #assert P.shape == (nmo-nocc, nocc)
        P = P.reshape(nmo-nocc, nocc)

        eai = mo_ene[nocc:,None]-mo_ene[None,:nocc]

        proc = einsum('ai,ai->ai',eai,P)

        proc += ( einsum('iajb,bj->ai',h2e_mo_sym[:nocc,nocc:,:nocc,nocc:],P)
                 - einsum('ijba,bj->ai',h2e_mo_sym[:nocc,:nocc,nocc:,nocc:],P)
        )
        return proc
    
    L = (-2*einsum('ia,jb,cjba->ci',x,x,h2e_mo_sym[nocc:,:nocc,nocc:,nocc:])
         + 2*einsum('ia,jb,ijka->bk',x,x,h2e_mo_sym[:nocc,:nocc,:nocc,nocc:])
    )

    L += (einsum('kl,ailk->ai',P_delta_mo[:nocc,:nocc],h2e_mo_sym[nocc:,:nocc,:nocc,:nocc])
        + einsum('bc,aibc->ai',P_delta_mo[nocc:,nocc:],h2e_mo_sym[nocc:,:nocc,nocc:,nocc:])
    )

    A = LinearOperator((nvir*nocc,nvir*nocc), matvec=A_matvec)
    b = L.flatten()

    z = gmres(A, b, rtol=1e-8, maxiter=100)[0].reshape(nvir, nocc)

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
    cis.nstates = 6
    cis.kernel()

    atom = []
    for i in range(mol.natm):
        atom.append(mol.atom_symbol(i))

    k = 2
    cis_grad = cis.Gradients()
    cis_grad.state = k
    cis_grad.kernel()

    X = []
    for i,x in enumerate(cis.xy):
        X.append(x[0])
    X = np.array(X)

    grad = calc_cis_grad(cis,X,state=k)
    print(f'--------- My CIS gradients for state {k} ----------')
    for i,g in enumerate(grad):
        print(f'{i:1} {atom[i]:3} {g[0]:12.8f} {g[1]:12.8f} {g[2]:12.8f}')

