import numpy as np
from pyscf import gto,dft,tdscf
import time

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

Func = 'PBE'

mf_hf = dft.RKS(mol)
mf_hf.xc = Func
mf_hf.kernel()

td = tdscf.TDA(mf_hf)
print('PySCF所得激发能')
td.nstates=6
td.kernel()

A,B = td.get_ab()

A_size = A.shape[0]*A.shape[1]
x = time.time()
# TD_matrix = np.einsum('iakl,kljb->iajb',A+B,A-B)
TD_matrix = A
TD_matrix = TD_matrix.reshape(A_size,A_size)

mo_ene = mf_hf.mo_energy
nocc = int(mol.nelectron)//2
e_ia = (mo_ene[nocc:].reshape(-1,1) - mo_ene[:nocc]).T
hdiag = e_ia.ravel()

# vind,hdiag = td.gen_vind()

# V = np.eye(A_size,A_size)
# TD_matrix = np.array(vind(V))

k = 6
eig = 2*k
V_old = np.zeros([A_size,eig])
hdiag = hdiag.reshape(-1,)
Dsort = hdiag.argsort()
for j in range(k):
    V_old[Dsort[j], j] = 1.0

max_iter = 50
for i in range(max_iter):
    
    W_old = np.einsum("ki,il->kl",TD_matrix,V_old)
    
    sub_A = np.einsum("ik,il->kl",V_old,W_old)
    
    val,ket = np.linalg.eigh(sub_A)
    
    sub_val = val[:k]
    sub_ket = ket[:,:k]

    residual = np.einsum("ki,il->kl",W_old,sub_ket) - np.einsum("l,ki,il->kl",sub_val,V_old,sub_ket)
    #print(np.linalg.norm(W_old))
    r_norm = np.linalg.norm(residual,axis=0).tolist()
    #print(r_norm)
    max_norm = np.max(r_norm)
    if i>2 and max_norm < 1e-8:
        y = time.time()
        break

    t = 1e-8
    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_val
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    new_guess = residual/D

    V_new = np.hstack([V_old,new_guess])
    V_old,_ = np.linalg.qr(V_new)

Hartree_to_eV = 27.211385050

print(f'Davidson:{sub_val*Hartree_to_eV}，耗时{y-x:.6f}')

x1 = time.time()
val,_ = np.linalg.eigh(TD_matrix)
y1 = time.time()
print(f'numpy.eigh:{val[:k]*Hartree_to_eV}，耗时{y1-x1:.6f}')

X_ia = np.zeros([A_size,k])
for i in range(k):
    eqn = TD_matrix - sub_val[i]*np.eye(A_size)
    r = np.zeros(A_size)
    X_ia[:,i] = np.linalg.solve(eqn,r)
print(r.shape)
X_ia = X_ia.reshape(A.shape[0],A.shape[1],-1)
print(np.linalg.solve(eqn,r))