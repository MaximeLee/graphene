import numpy as np

def compute_kinetic_operators(basis_set):
    T_k = []
    for pw_k_list in basis_set:
        T_k_ = [pw_k.kinetic_integral(pw_k) for pw_k in pw_k_list]
        T_k_ = np.diag(T_k_)
        T_k.append(T_k_)
    return T_k
