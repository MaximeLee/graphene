import numpy as np
import matplotlib.pyplot as plt

from constants import pi
from generate_mesh import mesh

# generate the carbon lattice with nx x ny hexagon cells
nx = 10
ny = 10

X, a1, a2 = mesh(nx, ny, centered = True)

plt.scatter(X[:, 0], X[:, 1])
plt.savefig('slab.png')
plt.close()

# compute reciprocal lattice vectors
A = np.vstack([a1,a2])
A_1 = np.linalg.inv(A)
b1 = A_1 @ np.array([2*pi, 0])
b2 = A_1 @ np.array([0, 2*pi])

del(A, A_1)

# compute reciprocal hexagon containing origin
# neighboring cell barycenters
O = np.zeros(3)
cell1 = b1
cell2 = b1 + b2
cell3 = b2
cell4 = -b1
cell5 = -b1 - b2
cell6 = -b2

# origin neighbors
neighbor1 = (O + cell1 + cell2)/3
neighbor2 = (O + cell2 + cell3)/3
neighbor3 = (O + cell3 + cell4)/3
neighbor4 = (O + cell4 + cell5)/3
neighbor5 = (O + cell5 + cell6)/3
neighbor6 = (O + cell6 + cell1)/3

# Bragg plane normals
normal1 = O + neighbor1
normal2 = O + neighbor2
normal3 = O + neighbor3
normal4 = O + neighbor4
normal5 = O + neighbor5
normal6 = O + neighbor6

# barycenter of each segment origin neighbor/pt on the Brag plane
barycenter1 = normal1/2
barycenter2 = normal2/2
barycenter3 = normal3/2
barycenter4 = normal4/2
barycenter5 = normal5/2
barycenter6 = normal6/2

# compute the first Brillon zone

# compute operators in Hamiltonian
