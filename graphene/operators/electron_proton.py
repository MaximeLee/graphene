"""compute electronic-proton integrals"""
import numpy as np
from graphene.basis.gaussian import PrimitiveGaussian

def nuclear_attraction_operator_gaussian(CPG_list, molecule, Chebyshev_quadrature_points_01):

    n = len(CPG_list)
    Vep = np.zeros([n,n])

    for atom in range(molecule.natoms):
        Rp = molecule.atomic_positions[atom]
        Z = molecule.atomic_numbers[atom]

        for i in range(n):  
            CPG_i = CPG_list[i]
            n_i = len(CPG_i)
    
            for j in range(n):
                CPG_j = CPG_list[j]
                n_j = len(CPG_j)
    
                for k in range(n_i):
                    PG_i = CPG_i.PG_list[k]
                    c_i = CPG_i.coeff_list[k]
    
                    for l in range(n_j):
                        PG_j = CPG_j.PG_list[l]
                        c_j = CPG_j.coeff_list[l]
    
                        Vep[i,j] += c_i * c_j * PrimitiveGaussian.electron_proton_int(PG_i, PG_j, Rp, Z, Chebyshev_quadrature_points_01)

    return Vep