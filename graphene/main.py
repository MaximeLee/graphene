import numpy as np
import matplotlib.pyplot as plt

from constants import pi
from cell import get_cell, get_brillouin_zone
from k_points.sampling import get_k_points, get_G_vectors
from basis.plane_wave import get_plane_wave_basis_set
from operators.kinetic import compute_kinetic_operators

# generate the carbon lattice 
X, R, X_reciprocal, G = get_cell()

# get Brillouin zone
X_BZ = get_brillouin_zone(X_reciprocal)

# get k-points
k_vectors = get_k_points(X_BZ, sampling_name="baldereschi", n=1)

# reciprocal lattice vectors in the basis set
E_cutoff = 50
G_vectors = get_G_vectors(G, E_cutoff)

# generate basis set
basis_set = get_plane_wave_basis_set(G_vectors, k_vectors)

# compute operators for each value of k

# kinetic operator
T_k = compute_kinetic_operators(basis_set)
