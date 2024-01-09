import numpy as np
from graphene.basis.gaussian import PrimitiveGaussian

def kinetic_operators_pw(basis_set):
    T_k = []
    for pw_k_list in basis_set:
        T_k_ = [pw_k.kinetic_integral(pw_k) for pw_k in pw_k_list]
        T_k_ = np.diag(T_k_)
        T_k.append(T_k_)
    return T_k

def kinetic_operator_gaussian(CPG_list):
    """kinetic matrix"""

    n = len(CPG_list)

    kinetic = np.zeros([n, n])

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

                    kinetic[i,j] += c_i * c_j * PrimitiveGaussian.kinetic_int(PG_i, PG_j)

    return kinetic
