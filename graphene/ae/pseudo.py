# Paper: Norm-Conserving Pseudopotentials 1979
import numpy as np

from graphene.integral.quadrature import R3_quadrature_points, subs
from graphene.operators import *
from pseudo_utils import *
from pseudo_config import *

# list of contracted gaussians
CG_list = get_CG_list()

# load density matrices
C_alpha = np.load('C_alpha.npy')
C_beta = np.load('C_beta.npy')

C_1s_alpha = C_alpha[:, 0:1]
C_1s_beta = C_beta[:, 0:1]

P_alpha = np.load('P_alpha.npy')
P_beta = np.load('P_beta.npy')
P_total = P_alpha + P_beta

P_1s_alpha = C_1s_alpha @ C_1s_alpha.T
P_1s_beta = C_1s_beta @ C_1s_beta.T

# loading integrals
S = np.load('S.npy')
T = np.load('T.npy')
Vep = np.load('Vep.npy')
Vee_ = np.load('Vee.npy')

# computing operators
H_core = T + Vep
J = coulomb_operator(Vee_, P_total)
K_alpha = exchange_operator(Vee_, P_alpha) 
K_beta = exchange_operator(Vee_, P_beta) 

# alpha/beta portentials (Vep + J - K)
V_alpha = Vep + J - K_alpha
V_beta  = Vep + J - K_beta

# r_cutoff
r_ml_alpha = find_r_ml(C_alpha, CG_list, l=0).x
r_ml_beta  = find_r_ml(C_beta, CG_list, l=0).x

r_cl_alpha = 0.5 * r_ml_alpha
r_cl_beta  = 0.5 * r_ml_beta

# V^{PS}_1_ij = <chi_i| (1-f(r/r_cl))V(r) +  c_l * f(r/r_cl)|chi_j>
# V^{PS}_1_ij = FV_PS_ij + c_l * F_PS_ij
FV_PS_alpha = function_in_basis(lambda xyz: (1-f(xyz/r_cl_alpha))*V_eval(xyz, V_alpha, C_alpha, CG_list), CG_list, R3_quadrature_points, subs)
FV_PS_beta = function_in_basis(lambda xyz: (1-f(xyz/r_cl_beta))*V_eval(xyz, V_beta, C_beta, CG_list), CG_list, R3_quadrature_points,subs)
F_PS_alpha = function_in_basis(lambda xyz: f(xyz/r_cl_alpha), CG_list, R3_quadrature_points, subs)
F_PS_beta  = function_in_basis(lambda xyz: f(xyz/r_cl_beta ), CG_list, R3_quadrature_points, subs)

F_PS_alpha_12 = scp.linalg.sqrt(np.linalg.inv(F_PS_alpha)) 
F_PS_beta_12  = scp.linalg.sqrt(np.linalg.inv(F_PS_beta)) 

# compute cl constant with scf loop for each spin
# operator to diagonalize to get the constant c_l
e_1s_alpha = np.load('e_alpha.npy')[0]
e_1s_beta  = np.load('e_beta.npy')[0]
O_alpha = T + FV_PS_alpha - e_1s_alpha * S
O_beta  = T + FV_PS_beta - e_1s_beta * S

for i in range(n_max):

    H_PS_alpha = -(T + FV_PS_alpha - e_1s_alpha * S)
    Hu_PS_alpha = F_PS_alpha_12 @ H_PS_alpha @ F_PS_alpha_12
    c_l_alpha, C_PS_1s_alpha = np.linalg.eigh(Hu_PS_alpha)

    H_PS_beta = -(T + FV_PS_beta - e_1s_beta * S)
    Hu_PS_beta = F_PS_beta_12 @ H_PS_beta @ F_PS_beta_12
    c_l_beta, C_PS_1s_beta = np.linalg.eigh(Hu_PS_beta)

