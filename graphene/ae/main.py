import os
import sys
import numpy as np

from graphene.cell import get_cell
from graphene.basis.gaussian import PrimitiveGaussian, ContractedGaussian
from graphene.basis import gaussian
from graphene.operators import *
from graphene.ae.molecule import Molecule

from graphene.energy import *

# get position of 6 atoms of unit cell
X_atoms = get_cell()[0]

# build Gaussian basis set
# read the contracted gaussian from the coefficient file
path = os.path.join(os.path.dirname(gaussian.__file__), "coefficients", "sto-3g", "C.txt")
with open(path,'r') as coeff_file:
    data = coeff_file.readlines()

# loop over Carbon atom
CG_list = []
for i in range(6):
    X_i = X_atoms[i:i+1]

    # read 1s contracted gaussians
    contraction_coefficients = []
    PG_list_1s = []
    for i in range(1,4):
        row = data[i].strip().split()
        gaussian_exponent = float(row[0])
    
        PG_list_1s.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=X_i))
    
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
    
        PG_list_2s.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=X_i))
        PG_list_2px.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=X_i, ex=1, ey=0, ez=0))
        PG_list_2py.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=X_i, ex=0, ey=1, ez=0))
        PG_list_2pz.append(PrimitiveGaussian(alpha=gaussian_exponent, atom_position=X_i, ex=0, ey=0, ez=1))
    
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
S = overlap_operator_gaussian(CG_list)
T = kinetic_operator_gaussian(CG_list)
print('kinetic and overlap done!')

unit_cell = Molecule()
for xyz in X_atoms:
    unit_cell.add_atom('C', xyz.reshape(1,-1))
Vep = nuclear_attraction_operator_gaussian(CG_list, unit_cell)
print('nuclear attraction done!')
Vee = electron_repulsion_operator_gaussian(CG_list)
print('electronic repulsion done!')
Epp = proton_proton_energy(unit_cell)
