import numpy as np

from pseudo_config import *

C_position = np.zeros((1,3), dtype=np.float64)

# cutoff functions
def f(x):
    return np.exp(-np.linalg.norm(x, axis=1, keepdims=True)**4)

def g(x, l=0):
    return x**l * f(x)

def V_eval(xyz, V_ae, C_ae, CG_list):
    V_xyz = 0.0
    n = len(CG_list)

    for i in range(C_ae.shape[1]):
        for k in range(n):
            CG_k = CG_list[k]
            V_xyz += V_ae[i,k] * C_ae[k,i] * CG_k(xyz)
        
    return V_xyz

# scf parameters
n_max = 100
E_threshold = 1e-6
E_old = np.inf

