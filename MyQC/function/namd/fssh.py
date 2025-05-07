import numpy as np
from pyscf import gto,scf,tdscf
import itertools

einsum = np.einsum

def calc_ovlp(Xm,Xn,Cm,Cn,ovlp):
    '计算不同波函数之间的重叠积分'
    nocc,nvir = Xm[0,:,:].shape
    ovlp_mo = einsum('mp,mn,nq->pq',Cm,ovlp,Cn)

    ovlp_00 = np.linalg.det(ovlp_mo[:nocc,:nocc])**2

    # ovlp_0_ia = np.linalg.solve(ovlp_mo[:nocc,:nocc],ovlp[:nocc,nocc:])**2*np.sqrt(ovlp_00)
    # ovlp_jb_0 = np.linalg.solve(ovlp_mo[:nocc,:nocc].T,ovlp[nocc:,:nocc].T)**2*np.sqrt(ovlp_00)

    # 不同行列式之间的重叠积分
    ovlp_jb_0 = np.zeros([nocc,nvir])
    ovlp_0_ia = np.zeros([nocc,nvir])
    for j in range(nocc):
        for b in range(nvir):
            S = ovlp_mo[:nocc,:nocc].copy()
            S[j,:] = ovlp_mo[nocc+b,:nocc]
            # alpha自旋的行列式与beta自旋行列式之积，*2代表ab两种激发
            ovlp_jb_0[j,b] = np.linalg.det(S)
    for i in range(nocc):
        for a in range(nvir):
            S = ovlp_mo[:nocc,:nocc].copy()
            S[:,i] = ovlp_mo[:nocc,nocc+a]
            ovlp_0_ia[i,a] = np.linalg.det(S)

    ovlp_0n = einsum('nia,ia->n',Xn,ovlp_0_ia)*2*np.sqrt(ovlp_00)
    ovlp_m0 = einsum('mjb,jb->m',Xm,ovlp_jb_0)*2*np.sqrt(ovlp_00)


    ovlp_jb_ia = np.zeros([nocc,nvir,nocc,nvir])
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    S = ovlp_mo[:nocc,:nocc].copy()
                    S[j,:] = ovlp_mo[nocc+b,:nocc]
                    S[:,i] = ovlp_mo[:nocc,nocc+a]
                    S[j,i] = ovlp_mo[nocc+b,nocc+a]

                    ovlp_jb_ia[j,b,i,a] = np.linalg.det(S)
    
    ovlp_mn = einsum('mjb,nia,jbia->mn',Xm,Xn,ovlp_jb_ia)*np.sqrt(ovlp_00)
    ovlp_mn += einsum('m,n->mn',ovlp_m0,ovlp_0n)
    ovlp_mn *= 2


    return np.block([[ovlp_00,ovlp_0n],[ovlp_m0.reshape(-1,1),ovlp_mn]])

def cal_wf_overlap_r(Xm, Ym, Xn, Yn, Cm, Cn, S):
    # restricted case has same orbitals for alpha and beta electrons
    has_m = True if isinstance(Xm, np.ndarray) else False
    has_n = True if isinstance(Xn, np.ndarray) else False
    has_y = True if (isinstance(Ym, np.ndarray) and isinstance(Yn, np.ndarray)) else False

    nroots, no, nv = Xm.shape

    smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)
    #print_matrix('smo:', smo, 5, 1)
    smo_oo = np.copy(smo[:no,:no])
    dot_0 = np.linalg.det(smo_oo)
    ovlp_00 = dot_0**2

    if not (has_m or has_n):
        return ovlp_00

    # Cramer's rule
    # Ax_j = b_j <=> x_j = det(A_j(b_j)) / det(A) where A_j(b_j) is replacing A's j-column with b_j
    vec1 = np.linalg.solve(smo_oo.T, smo[no:,:no].T) # replace rows
    vec2 = np.linalg.solve(smo_oo, smo[:no,no:]) # replace columns

    #vec3 = np.linalg.solve(smo_oo, Xm.transpose(1,0,2).reshape(no, -1))
    #vec3 = vec3.reshape(no, nroots, nv).transpose(1,0,2)
    #vec4 = smo[no:,no:] - np.einsum('ia,ib->ab', vec1, smo[:no,no:])
    #vec4 = np.einsum('ab,kib->kia', vec4, Xn)


    # excited-ground
    if has_m:
        ovlp1 = np.einsum('kia,ia->k', Xm, vec1)
        ovlp_m0 = np.copy(ovlp1)

        if has_y:
            ovlp2 = np.einsum('kia,ia->k', Ym, vec2)
            ovlp_m0 += ovlp2

        ovlp_m0 *= 2.*ovlp_00
        if not has_n:
            return np.array([ovlp_00, ovlp_m0])

    # ground-excited
    if has_n:
        ovlp3 = np.einsum('kia,ia->k', Xn, vec2)
        ovlp_0n = np.copy(ovlp3)

        if has_y:
            ovlp4 = np.einsum('kia,ia->k', Yn, vec1)
            ovlp_0n += ovlp4

        ovlp_0n *= 2.*ovlp_00
        if not has_m:
            return np.array([ovlp_00, ovlp_0n])

    # excited-excited
    if has_m and has_n:
        # e-g * g-e
        ovlp_mn = np.einsum('m,n->mn', ovlp1, ovlp3)
        if has_y:
            ovlp_mn -= np.einsum('m,n->mn', ovlp2, ovlp4)

        # e-e * g-g
        for a, i in itertools.product(range(nv), range(no)):
            ts0 = np.copy(smo[:no,:])
            ts0[i,:] = smo[no+a,:]
            vec3 = np.linalg.solve(ts0[:,:no], ts0[:,no:])
            ovlp_mn += np.einsum('m,njb,jb->mn', Xm[:,i,a], Xn, vec3) * vec1[i,a]

            if has_y:
                ovlp_mn -= np.einsum('mjb,n,jb->mn', Ym, Yn[:,i,a], vec3) *vec1[i,a]

        ovlp_mn *= 2.*ovlp_00
        return np.block([[ovlp_00, ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])
    
def run_td(mol,i,j,sign='+',delta=0.01):
    coords = mol.atom_coords(unit='Bohr')
    if sign == '+':
        coords[i,j] += delta
    elif sign == '-':
        coords[i,j] -= delta
    else:
        raise
    mol1 = mol.set_geom_(coords)
    mol1.unit='Bohr'
    mol1.build()

    mf1 = scf.RHF(mol1)
    mf1.kernel()
    td1 = tdscf.TDA(mf1)
    k=6
    td1.nstates=k
    td1.kernel()

    Xn = []
    for l in range(k):
        Xn.append(td1.xy[l][0])
    Xn = np.array(Xn)

    Cn = mf1.mo_coeff
    return Xn,Cn,mol1

def _calc_ovlp(S1_mo,Xm):
    nocc,nvir = Xm[0,:,:].shape
    ovlp_mo = S1_mo
    ovlp_ia_jb = np.zeros([nocc,nvir,nocc,nvir])
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    S = ovlp_mo[:nocc,:nocc].copy()
                    S[j,:] = ovlp_mo[nocc+b,:nocc]
                    S[:,i] = ovlp_mo[:nocc,nocc+a]
                    S[j,i] = ovlp_mo[nocc+b,nocc+a]

                    ovlp_ia_jb[j,b,i,a] = np.linalg.det(S)
    ovlp1 = np.einsum('kia,ljb,iajb->kl',Xm,Xm,ovlp_ia_jb)
    return ovlp1

def nac(mol,delta=0.01):

    mol.verbose = 0
    coords = mol.atom_coords(unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    td = tdscf.TDA(mf)
    k=6
    td.nstates=k
    td.kernel()

    Xm = []
    for i in range(k):
        Xm.append(td.xy[i][0])
    Xm = np.array(Xm)

    Cm = mf.mo_coeff
    nac_vector = np.zeros([k+1,k+1,coords.shape[0],coords.shape[1]])

    nocc,nvir = Xm[0,:,:].shape#

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            
            Xn1,Cn1,mol1 = run_td(mol,i,j,sign='+',delta=delta)
            S1 = gto.intor_cross('int1e_ovlp', mol, mol1)
            Xn1,Yn1,Cn1 = change_phase(x0=Xm,y0=None,x1=Xn1,y1=None,mo0=Cm,mo1=Cn1,ovlp=S1)
            
            ovlp1 = calc_ovlp(Xm,Xn1,Cm,Cn1,S1)
            ovlp1 = sign_fixing(ovlp1)

            # S1_mo = einsum('mp,mn,nq->pq',Cm,S1,Cn1)
            # ovlp1 = np.einsum('kia,lia->kl',Xm,Xn1)
            # ovlp1 += _calc_ovlp(S1_mo,Xm)
            
            Xn2,Cn2,mol2 = run_td(mol,i,j,sign='-',delta=delta)
            S2 = gto.intor_cross('int1e_ovlp', mol, mol2)
            Xn2,Yn2,Cn2 = change_phase(x0=Xm,y0=None,x1=Xn2,y1=None,mo0=Cm,mo1=Cn2,ovlp=S2)

            ovlp2 = calc_ovlp(Xm,Xn2,Cm,Cn2,S2)
            ovlp2 = sign_fixing(ovlp2)

            # S2_mo = einsum('mp,mn,nq->pq',Cm,S2,Cn2)
            # ovlp2 = np.einsum('kia,lia->kl',Xm,Xn2)
            # ovlp2 += _calc_ovlp(S2_mo,Xm)
            
            nac_vector[:,:,i,j] = (ovlp1-ovlp2)/(2*delta)
            # nac_vector[1:,1:,i,j] = (ovlp1-ovlp2)/(2*delta)

    return nac_vector

def sign_fixing(mat):
    """
    refer Zhou, Subotnik JCTC 2020, 10.1021/acs.jctc.9b00952
    """
    #U, s, Vt = np.linalg.svd(mat)
    #mat = np.einsum('ij,jk->ik', U, Vt)

    if np.linalg.det(mat) < 0.:
        mat[:,0] *= -1.

    nroots = mat.shape[0]

    # Jacobi sweeps
    converged = False
    while not converged:
        converged = True

        for i, j in itertools.product(range(nroots), range(nroots)):
            dot  = 3.* (mat[i,i]**2 + mat[j,j]**2)
            dot += 6.* (mat[i,j] * mat[j,i])
            dot += 8.* (mat[i,i] + mat[j,j])
            dot -= 3.* (np.dot(mat[i,:], mat[:,i]) + np.dot(mat[j,:], mat[:,j]))

            if dot < 0.:
                mat[:,i] *= -1.
                mat[:,j] *= -1.
                converged = False

    return mat

def change_phase(x0, y0, x1, y1, mo0, mo1, ovlp):
    nroots, no, nv = x1.shape
    ovlp = np.einsum('mp,mn,nq->pq', mo0, ovlp, mo1)
    idx = np.argmax(np.abs(ovlp), axis=0) # large index for each column
    #print('idx:', idx)

    for i, j in enumerate(idx):
        if ovlp[i,j] < 0.:
            #print(i, j)
            mo1[:,j] *= -1
            if j < no:
                x1[:,j,:] *= -1.
            else:
                x1[:,:,j-no] *= -1.
            if isinstance(y1, np.ndarray):
                if j < no:
                    y1[:,j,:] *= -1.
                else:
                    y1[:,:,j-no] *= -1.

    return x1, y1, mo1

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = ''' 
    O                 -3.01915195    0.11651239    0.00000000
    H                 -2.12897584   -0.24291284    0.00000000
    H                 -2.97700797    1.07558688    0.00000000
    '''

    mol.basis = 'sto-3g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    td = tdscf.TDA(mf)
    k=10
    td.nstates=k
    td.kernel()

    m = mol.atom_mass_list().reshape(-1,1)
    natm = m.shape[0]
    atom = []
    for i in range(natm):
        atom.append(mol.atom_symbol(i))
    
    nacv = nac(mol,delta=0.01)
    
    k = 0
    l = 1
    #nacv[k,l,:,:] = sign_fixing(nacv[k,l,:,:])
    #print(mol.atom_coords(unit='A'))
    print(f'        ********** < {k} | \\nabla_R | {l} > **********')

    for i in range(len(atom)):
        print(f'{atom[i]:4}{nacv[k,l,i,0]:16.8f}  {nacv[k,l,i,1]:16.8f}  {nacv[k,l,i,2]:16.8f}')
    print(np.linalg.norm(nacv[k,l,:,:]))

