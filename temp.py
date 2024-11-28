import numpy as np
from scipy import special
import numpy
from pyscf import gto,dft,tdscf,lib,scf
import time
from functools import partial
np.einsum = partial(np.einsum,optimize=["greedy", 1024 ** 3 * 40 / 8])
mol = gto.Mole()
mol.atom = '''
 C                  0.69091000   -0.00000900    0.00005600
 C                  0.02087600    1.23401200    0.11547300
 C                 -1.35141400    1.24115600    0.12060300
 C                 -2.06641700    0.00002400   -0.00004200
 C                 -1.35141700   -1.24115400   -0.12043100
 C                  0.02086700   -1.23402500   -0.11525900
 H                  0.59903700    2.14730700    0.19073200
 H                 -1.90322200    2.17148500    0.21249700
 H                 -1.90326700   -2.17146900   -0.21223300
 H                  0.59900600   -2.14734500   -0.19037800
 N                  2.16510300    0.00002700   -0.00006700
 O                  2.70857600   -1.04975800    0.32016200
 O                  2.70858700    1.04973800   -0.32029700
 N                 -3.39959000    0.00001100   -0.00024900
 H                 -3.93397400    0.86093900    0.07829100
 H                 -3.93390500   -0.86104600   -0.07801600
'''
mol.basis = 'sto-3g'
mol.build()

mf = dft.RKS(mol)
mf.kernel()

td = tdscf.TDA(mf)
A = td.get_ab()[0]
_,hdiag = td.gen_vind()
A_size =A.shape[0]*A.shape[1]
A = A.reshape(A_size,A_size)
print(np.sort(np.diagonal(A))-np.sort(hdiag))

'''
int1e_r = mol.intor_symmetric('int1e_r', comp=3)



dm1 = mf.make_rdm1()[0]
dipole1 = np.einsum('xuv,uv',int1e_r1,dm)
'''
#print((dipole-dipole1))
#print(xy1[1]-xy[1])
#print(np.allclose(dm,dm1))


#print(np.allclose(int1e_r,int1e_r1))

'''
Func = 'PBE'

mf_hf = dft.RKS(mol)
mf_hf.xc = Func
mf_hf.kernel()

td = tdscf.TDA(mf_hf)
print('PySCF所得激发能')
td.nstates=6
td.kernel()
os = td.oscillator_strength()
e = td.e*27.211385050
#print(e)
#print(os)

dipole = td.transition_dipole()
print(dipole)
print(mol.atom_coords)
for i in range(mol.atom_coords):
    print(i)



from math import pi
import matplotlib.pyplot as plt
def Gau(x,x0,FWHM):
    c = FWHM/(2*np.sqrt(2*np.log(2)))
    y = np.exp(-(x-x0)**2/(2*(c**2)))
    return y/(c*np.sqrt(2*pi))
x = np.arange(2,5,3/1000)
y = np.zeros_like(x)
for i in range(len(td.e)):
    yi = Gau(x,e[i],0.6667)*os[i]
    y += yi
    plt.plot(x,yi)
    plt.bar(e[i],os[i],width=0.01)
plt.plot(x,y)
plt.show()


mo_ene = mf_hf.mo_energy
nocc = int(mol.nelectron)//2
e_ia = (mo_ene[nocc:].reshape(-1,1) - mo_ene[:nocc]).T
hdiag = e_ia.ravel()

os = td.oscillator_strength()
print('PySCF_os')
print(os)
X = td.xy

mo_c = mf_hf.mo_coeff

grids = dft.gen_grid.Grids(mol)
grids.build()
#weights = grids.weights
#ni = dft.numint.NumInt()
#ao_1 = ni.eval_ao(mol,grids.coords,deriv=1)
#ao_0 = ao_1[0]
#coords = grids.coords

u = np.zeros([len(X),3])
Dm_0 = np.diag(mf_hf.mo_occ)
transdip = td.transition_dipole()
-0.00578500
int1e_r = mol.intor_symmetric('int1e_r', comp=3)

#Dm_0_ao = np.einsum('pi,ij,qj->pq',mo_c,Dm_0,mo_c.conj())

#Dm = np.diag(mf_hf.mo_occ)

#Dm_ao = np.einsum('pi,ij,qj->pq',mo_c,Dm,mo_c.conj())

#rho_0 = np.einsum("gu,gv,uv->g",ao_0,ao_0,Dm_0_ao)

#r_e = np.einsum("kx,k,k->x",coords,rho_0,weights)
#r_e /= np.einsum('k,k',rho_0,weights)

#u_ao = np.einsum("gu,gv,gx,g->xuv",ao_0,ao_0,coords,weights)
#print(np.linalg.norm(u_ao-int1e_r))

for i in range(len(X)):

    u_mo = lib.einsum("ua,xuv,vi->xia",mo_c[:,nocc:],int1e_r,mo_c[:,:nocc])
    #u_mo -= lib.einsum('k,k->',rho_0**2,weights)
    u_k = lib.einsum('ia,xia->x',X[i][0],u_mo)*2
    u[i] = u_k

u2 = (u**2).sum(1)
print('my_os')
f = 2*td.e*(u2)/3

#print(transdip)
print(f)


vind,_ = td.gen_vind()
A = np.ones([8*28,4])
print(vind(A.T).T.shape)'''

