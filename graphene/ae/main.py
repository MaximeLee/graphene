import os
import sys
import numpy as np
import scipy as scp
import gc
import csv
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from graphene.cell import get_cell
from graphene.basis.gaussian import PrimitiveGaussian, ContractedGaussian
from graphene.basis import gaussian
from graphene.operators import *
from graphene.ae.molecule import Molecule

from graphene.energy import *

C_position = np.zeros((1,3), dtype=np.float64)

# build Gaussian basis set
# read the contracted gaussian from the coefficient file
path = os.path.join(os.path.dirname(gaussian.__file__), "coefficients", "sto-3g", "C.txt")
with open(path,'r') as coeff_file:
    data = coeff_file.readlines()

# loop over Carbon atom
CG_list = []

# read 1s contracted gaussians
contraction_coefficients = []
PG_list_1s = []
for i in range(1,4):
    row = data[i].strip().split()
    gaussian_exponent = float(row[0])

    PG_list_1s.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=C_position))

    contraction_coefficients.append(float(row[1]))

CG_1s = ContractedGaussian(contraction_coefficients, PG_list_1s)

# read 2s and 2p contracted gaussians
PG_list_2s = []
PG_list_2px = []
PG_list_2py = []
PG_list_2pz = []
contraction_coefficients_2s = []
contraction_coefficients_2px = []
contraction_coefficients_2py = []
contraction_coefficients_2pz = []
for i in range(3):
    
    row = data[i+5].strip().split()
    gaussian_exponent = float(row[0])

    PG_list_2s.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=C_position))
    PG_list_2px.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=C_position, ex=1, ey=0, ez=0))
    PG_list_2py.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=C_position, ex=0, ey=1, ez=0))
    PG_list_2pz.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=C_position, ex=0, ey=0, ez=1))

    contraction_coefficients_2s.append(float(row[1]))
    contraction_coefficients_2px.append(float(row[2]))
    contraction_coefficients_2py.append(float(row[2]))
    contraction_coefficients_2pz.append(float(row[2]))

CG_2s = ContractedGaussian(contraction_coefficients_2s, PG_list_2s)
CG_2px = ContractedGaussian(contraction_coefficients_2s, PG_list_2px)
CG_2py = ContractedGaussian(contraction_coefficients_2s, PG_list_2py)
CG_2pz = ContractedGaussian(contraction_coefficients_2s, PG_list_2pz)

# add to list
CG_list.extend([CG_1s, CG_2s, CG_2px, CG_2py, CG_2pz])

del(data)

# compute operators for each spin
print("Computing operators...")
if os.path.exists('S.npy'):
    S = np.load('S.npy')
else:
    S = overlap_operator_gaussian(CG_list)
    np.save('S.npy', S)

if os.path.exists('T.npy'):
    T = np.load('T.npy')
else:
    T = kinetic_operator_gaussian(CG_list)
    np.save('T.npy', T)
print('kinetic and overlap done!')

M = Molecule()
M.add_atom('C', C_position)
if os.path.exists('Vep.npy'):
    Vep = np.load('Vep.npy')
else:
    Vep = nuclear_attraction_operator_gaussian(CG_list, M)
    np.save('Vep.npy', Vep)
print('nuclear attraction done!')

if os.path.exists('Vee.npy'):
    Vee_ = np.load('Vee.npy')
else:
    Vee_ = electron_repulsion_operator_gaussian(CG_list)
    np.save('Vee.npy', Vee_)
print('electronic repulsion done!')

Epp = proton_proton_energy(M)

# density matrices initial guesses
# and other useful matrices
n_basis = len(CG_list)

n_alpha = 3
P_alpha = np.zeros((n_basis, n_basis))

n_beta = 3
P_beta = np.zeros((n_basis, n_basis)) 

P_total = P_beta + P_alpha

# S^(-1/2)
S_12  = scp.linalg.sqrtm(np.linalg.inv(S))

# SCF loop
n_max = 100
E_threshold = 1e-6
E_old = np.inf

H_core = T + Vep

for i in range(n_max):
    
    # evaluation the coulomb and exchange operators
    #J_alpha = coulomb_operator(Vee_, P_alpha)
    #J_beta = coulomb_operator(Vee_, P_beta)
    J = coulomb_operator(Vee_, P_total)

    K_alpha = exchange_operator(Vee_, P_alpha) 
    K_beta = exchange_operator(Vee_, P_beta) 

    F_alpha = H_core + J - K_alpha
    F_beta  = H_core + J - K_beta

    # orthogonalisation of the Pople-Nesbet equations
    Fu_alpha = S_12 @ F_alpha @ S_12
    Fu_beta  = S_12 @ F_beta  @ S_12

    # solving Pople-Nesbet equations
    e_alpha, Cu_alpha = np.linalg.eigh(Fu_alpha)
    e_beta , Cu_beta  = np.linalg.eigh(Fu_beta )
    Cu_alpha = Cu_alpha[:, :n_alpha]
    Cu_beta  = Cu_beta [:, :n_beta]

    # reverse basis change
    C_alpha = S_12 @ Cu_alpha
    C_beta  = S_12 @ Cu_beta 

    # new density matrices
    P_alpha = C_alpha @ C_alpha.T
    P_beta = C_beta @ C_beta.T

    P_total = P_alpha + P_beta

    # computing energy of the wf
    E_core = np.sum(H_core * P_total)
    E_coulomb = 0.5 * np.sum(J * P_total)
    E_exchange = 0.5 * (np.sum(K_alpha * P_alpha) + np.sum(K_beta * P_beta))

    E = E_core + E_coulomb - E_exchange

    # check convergence
    converged = abs(E-E_old)<E_threshold

    # update E_old
    E_old = E

    if converged:
        print(f"Converged in {i+1} iterations!")
        break
print("Finished!")

print('Saving wavefunction coefficients and energies...')
np.save('P_alpha.npy', P_alpha)
np.save('P_beta.npy', P_beta)

np.save('C_alpha.npy', C_alpha)
np.save('C_beta.npy', C_beta)

np.save('e_alpha.npy', e_alpha)
np.save('e_beta.npy', e_beta)

print("Saving wavefunction values...")
"""
n_r = 100
n_theta = 50
n_phi = 50

r_vec = np.linspace(0.0, r_max, n_r, endpoint=False)
phi_vec = np.linspace(0.0, np.pi, n_phi, endpoint=False)
theta_vec = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)

r_mesh, phi_mesh, theta_mesh = np.meshgrid(r_vec, phi_vec, theta_vec)

r_mesh = r_mesh.flatten()
phi_mesh = phi_mesh.flatten()
theta_mesh = theta_mesh.flatten()
"""
npts = 200000
r_mesh = np.random.uniform(size=npts, low=0.0, high=5.0)
phi_mesh = np.random.uniform(size=npts, low=0.0, high=np.pi)
theta_mesh = np.random.uniform(size=npts, low=0.0, high=2*np.pi)

x_mesh = r_mesh * np.sin(phi_mesh) * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(phi_mesh) * np.sin(theta_mesh)
z_mesh = r_mesh * np.cos(phi_mesh)

X = np.stack((x_mesh, y_mesh, z_mesh), axis=1)
X = np.unique(X, axis=0)
n = len(X)

#del r_vec, phi_vec, theta_vec
del r_mesh, phi_mesh, theta_mesh
del x_mesh, y_mesh, z_mesh

# evaluate CG at each mesh point
CG_xyz_eval = []
for CG in CG_list:
    CG_xyz_eval.append(CG(X).flatten())
CG_xyz_eval = np.array(CG_xyz_eval)
#CG_xyz_product = np.einsum("ni,nj->nij", CG_xyz_eval, CG_xyz_eval)

density_xyz = [] #np.einsum("ij,nij->n", P_total, CG_xyz_product)
for ii in range(n):
    density_ = 0.0
    for i in range(n_basis):
        for j in range(n_basis):
            density_ += P_total[i, j] * CG_xyz_eval[i, ii] * CG_xyz_eval[j, ii]

    density_xyz.append(density_)

# save total wavefunction
data = {
    'x': X[:,0],
    'y': X[:,1],
    'z': X[:,2],
    'rho': density_xyz
}
df = pd.DataFrame(data)
df.to_csv('density_xyz.csv', index=False)

# save each electron wavefunction
for spin in ['alpha', 'beta']:
    if spin=='alpha':
        n_spin = n_alpha
        P_spin = P_alpha
        C_spin = C_alpha
    else:
        n_spin = n_beta
        P_spin = P_beta
        C_spin = C_beta

    for nn in range(n_spin):
        density_spin_nn_xyz = []
        
        for ii in range(n):
            density_ = 0.0
            for i in range(n_basis):
                for j in range(n_basis):
                    density_ +=  C_spin[i, nn] * C_spin[j, nn] * CG_xyz_eval[i, ii] * CG_xyz_eval[j, ii]

            density_spin_nn_xyz.append(density_)

        data = {
            'x': X[:,0],
            'y': X[:,1],
            'z': X[:,2],
            'rho': density_spin_nn_xyz
        }
        df = pd.DataFrame(data)
        df.to_csv(f'density_{spin}_{nn+1}_xyz.csv', index=False)
        
print("Saved!")

