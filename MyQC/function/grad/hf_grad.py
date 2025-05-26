import numpy as np
from pyscf import gto,scf,tdscf,grad
import itertools

einsum = np.einsum

'''
ref: https://py-xdh.readthedocs.io/zh-cn/latest/derivonce/grad_rhf_skeleton.html
'''

def calc_hcore_grad_ao(mol,ao_slice=None):
    '''
    计算单电子积分梯度
    '''
    if ao_slice is None:
        # 第i个原子对应的基函数范围
        ao_slice = mol.aoslice_by_atom()[:,2:]
    '''
    ipkin  : <\partial_t u|\hat{T}|v>
    ipnuc  : <\partial_t u|\hat{V}|v>
    iprinv : <\partial_t u|r^{-1}|v>
    '''
    ipkin = mol.intor('int1e_ipkin')
    ipnuc = mol.intor('int1e_ipnuc')
    Z = mol.atom_charges()

    hcore_grad = np.zeros([mol.natm,3,mol.nao,mol.nao])
    for i in range(mol.natm):
        begin,end = ao_slice[i,:]
        hcore_grad[i,:,begin:end,:] -= ipkin[:,begin:end,:]
        hcore_grad[i,:,begin:end,:] -= ipnuc[:,begin:end,:]

        # 将积分原点设置为第i个原子核位置
        with mol.with_rinv_as_nucleus(i):
            hcore_grad[i] -= Z[i] * mol.intor("int1e_iprinv")
    

    hcore_grad += hcore_grad.swapaxes(-1, -2)
        
    return hcore_grad

def calc_ovlp_grad_ao(mol,ao_slice=None):
    '''
    计算重叠积分梯度
    '''
    if ao_slice is None:
        ao_slice = mol.aoslice_by_atom()[:,2:]
    ipovlp = mol.intor('int1e_ipovlp')
    ovlp_grad = np.zeros([mol.natm,3,mol.nao,mol.nao])
    for i in range(mol.natm):
        begin,end = ao_slice[i,:]
        ovlp_grad[i,:,begin:end,:] = -ipovlp[:,begin:end,:]
    ovlp_grad += ovlp_grad.swapaxes(-1, -2)
    return ovlp_grad


def calc_eri_grad_ao(mol,ao_slice=None):
    '''
    计算双电子积分梯度
    '''
    if ao_slice is None:
        ao_slice = mol.aoslice_by_atom()[:,2:]
    '''
    iperi = <\partial_t {u} v||kl>
    '''
    iperi = mol.intor('int2e_ip1')
    eri_grad = np.zeros([mol.natm,3,mol.nao,mol.nao,mol.nao,mol.nao])
    for i in range(mol.natm):
        begin,end = ao_slice[i,:]
        eri_grad[i,:,begin:end] -= iperi[:,begin:end]
    eri_grad += eri_grad.swapaxes(-3, -4)
    eri_grad += eri_grad.swapaxes(-1, -3).swapaxes(-2, -4)

    return eri_grad

def grad_nuc(mol):
    '''
    计算核排斥能梯度
    '''
    z = mol.atom_charges()
    r = mol.atom_coords()
    dr = r[:,None,:] - r
    dist = np.linalg.norm(dr, axis=2)
    
    diag_idx = np.diag_indices(z.size)
    dist[diag_idx] = 1e100
    rinv = 1./dist
    rinv[diag_idx] = 0.
    gs = np.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv**3)

    return gs

def calc_wdm(mf):

    mol = mf.mol
    nocc = mol.nelectron // 2
    mo_c = mf.mo_coeff
    mo_ene = mf.mo_energy[:nocc]

    return -2*(mo_c[:,:nocc] * mo_ene[:nocc]) @ mo_c[:,:nocc].T

def hf_grad(mf):
    mol = mf.mol

    wdm = calc_wdm(mf)
    
    dm = mf.make_rdm1()

    h1ex = calc_hcore_grad_ao(mol)
    h2ex = calc_eri_grad_ao(mol)
    ovlpx = calc_ovlp_grad_ao(mol)
    
    h2ex_sym = h2ex - 0.5*h2ex.swapaxes(-2,-3 )
    grad = (
        einsum('Axpq,pq->Ax',h1ex,dm)+
        0.5*einsum('Axpqrs,pq,rs->Ax',h2ex_sym,dm,dm)+
        einsum('Axpq,pq->Ax',ovlpx,wdm)+grad_nuc(mol))
    return grad


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = ''' 
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
    '''

    mol.basis = 'sto-3g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mf_grad = grad.RHF(mf).run()
    
    grad = hf_grad(mf)

    g = mf.Gradients().kernel()
    print(grad)

