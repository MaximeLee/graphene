import os
from scipy.optimize import minimize_scalar
from graphene.basis import gaussian
from graphene.basis.gaussian import PrimitiveGaussian, ContractedGaussian
from pseudo_config import *

def find_r_ml(C_ae, CG_list, l):
    """find outermost peak of r * phi_l(r) with optimization algorithm"""
    n = len(CG_list)
    C_ae_l = C_ae[:, l]

    def objective_function(r):
        """valence wave function"""
        y = 0
        for i in range(n):
            xyz = np.array([[r, 0, 0]]) 
            y += CG_list[i](xyz) 
        y = -r * y[0,0]
        return y

    r_ml = minimize_scalar(objective_function, bounds=(0.0, 10.0), method='bounded')
    return r_ml

def function_in_basis(func, CG_list, R3_quadrature_points, subs):
    """computing each elements via 3D quadrature integration"""

    n = len(CG_list)
    O = np.zeros((n, n))

    weights = R3_quadrature_points[:, 0:1]
    points = R3_quadrature_points[:, 1:]

    # computing off-diagonal elements
    for i in range(n):
        CG_i = CG_list[i]
        PG_i_list = CG_i.PG_list
        coeff_i_list = CG_i.coeff
        n_i = len(CG_i)

        for j in range(i):
            CG_j = CG_list[j]
            PG_j_list = CG_j.PG_list
            coeff_j_list = CG_j.coeff
            n_j = len(CG_j)

            for ii in range(n_i):
                PG_ii = PG_i_list[ii]
                c_ii = coeff_i_list[ii]
                for jj in range(n_j):
                    PG_jj = PG_i_list[jj]
                    c_jj = coeff_i_list[jj]

                    int_ij = np.sum(weights * subs * func(points-C_position) * PG_ii(points)*PG_jj(points))
                    O[i, j] += c_ii * c_jj * int_ij
                    O[j, i] += c_ii * c_jj * int_ij

    # computing diagonal elements
    for i in range(n):
        CG_i = CG_list[i]
        PG_i_list = CG_i.PG_list
        coeff_i_list = CG_i.coeff
        n_i = len(CG_i)

        for ii in range(n_i):
            PG_ii = PG_i_list[ii]
            c_ii = coeff_i_list[ii]

            int_ii = np.sum(weights * subs * func(points-C_position) * PG_ii(points)**2)
            O[i, i] += c_ii**2 * int_ii

    return O

def get_CG_list():

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
    
    return CG_list

