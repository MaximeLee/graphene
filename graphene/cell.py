import numpy as np
from copy import deepcopy as dcp
import math
from graphene.constants import pi

theta = pi/3.0 # pi*180/60

# nanometers
#l_eq = 0.142
# A
#l_eq = 1.42
# au
l_eq = 1.42 * 1.88973

def get_cell(l = l_eq):
    """
    compute:
    - unit cell
    - lattice vectors
    - reciprocal lattice vectors
    """
    X = np.vstack(
        [[math.cos(theta*i + pi/2), np.sin(theta*i+pi/2)] for i in range(6)]
    ) * l

    X_reciprocal = X/l * 2 * pi

    # real space vector basis
    a1 = (X[5] + X[0])
    a2 = (X[3] + X[4])
    R = np.vstack([a1,a2])

    # reciprocal vector basis
    R_1 = np.linalg.inv(R)
    b1 = R_1 @ np.array([2*pi, 0])
    b2 = R_1 @ np.array([0, 2*pi])
    G = np.vstack([b1,b2])

    # padding to have 3D vectors
    X = np.pad(X, ((0, 0), (0, 1)), 'constant')
    R = np.pad(R, ((0, 0), (0, 1)), 'constant')
    X_reciprocal = np.pad(X_reciprocal, ((0, 0), (0, 1)), 'constant')
    G = np.pad(G, ((0, 0), (0, 1)), 'constant')
    return X, R, X_reciprocal, G

def get_brillouin_zone(X):
    """return points of the First Brillouin zone"""

    plane_point_vector = []
    for i in range(6):
        nx, ny = X[i]
        pt = np.array([nx, ny])/2
        v = np.array([[ny, -nx]])

        plane_point_vector.append([pt, v])

    X_BZ = []
    for i in range(6):
        X1, V1 = plane_point_vector[i]
        X2, V2 = plane_point_vector[(i+1)%6]
        
        # use parametric equations to solve intersection problem
        M = np.hstack([-V1.T, V2.T])
        B = X1 - X2
        t1, t2 = np.linalg.solve(M, B.flatten())

        X_BZ.append(X1 + t1 * V1)

    X_BZ = np.concatenate(X_BZ, axis = 0)
    return X_BZ
