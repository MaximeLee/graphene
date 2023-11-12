import numpy as np
from copy import deepcopy as dcp

from constants import pi

theta = pi/3.0 # pi*180/60

l_eq = 0.142

def mesh(nx, ny, l = l_eq, centered = False):
    """compute :
    - barycenter of every hexagon cell of the lattice on its reciprocal
    - Brillouin zone of on the hexagon of unit length
    - real and reciprocal lattice vectors and their lengths
    """
    ny = ny + 1 if ny%2==0 else ny
    O = np.zeros(2)

    # distance between atoms on the same row
    dx = l * np.sin(theta) * 2.0 
    # distance between atoms in diagonal
    dy = l * np.cos(theta) 

    # real space vector basis
    a1 = np.array([[dx/2,  dy]])
    a2 = np.array([[dx/2, -dy]])
    # x-translation
    ex = a1 + a2

    # reciprocal space vector basis
    A = np.vstack([a1,a2])
    A_1 = np.linalg.inv(A)
    b1 = A_1 @ np.array([2*pi, 0])
    b2 = A_1 @ np.array([0, 2*pi])
    
    # Real space vectors R
    FirstCell = np.zeros(shape=(1,2))
    
    # define the first row
    tmp_list = [dcp(FirstCell)]
    for i in range(nx-1):
        tmp = tmp_list[-1] + ex
        tmp_list.append(dcp(tmp))
    
    R_real = np.concatenate(tmp_list, axis = 0)
    LastRow = np.asarray(R_real)
    
    for i in range(ny-1):
        
        if i%2==0:
            R_tmp = LastRow[:-1] + a2
        else:
            R_tmp = LastRow - a1
            R_tmp = np.concatenate([R_tmp, R_tmp[-1] + ex], axis=0)
    
        R_real = np.concatenate([R_real, R_tmp], axis=0)
    
        LastRow = np.asarray(R_tmp)
        
    if centered:
        m = np.mean(R_real, axis = 0, keepdims = True)
        R_real -= m
        O -= m

    R_reciprocal = R_real/a1 * b1

    # computing the first Brillouin zone polygon over Hexagon of unit length
    # save point on the plane and vector in it
    Planes= []
    for i in range(6):
        # normal vector
        nx, ny = np.cos(theta*i + pi/2), np.sin(theta*i + pi /2)
        # pts on the plane
        x, y = nx/2, ny/2

        Planes.append([np.array([[x, y]]), -np.array([[ny, -nx]])])
        
    BZ_polygon = []
    for i in range(6):
        X1, V1 = Planes[i]
        X2, V2 = Planes[(i+1)%6]
        
        # use parametric equations to solve intersection problem
        M = np.hstack([-V1.T, V2.T])
        B = X1 - X2
        t1, t2 = np.linalg.solve(M, B.flatten())

        BZ_polygon.append(X1 + t1 * V1)

    BZ_polygon = np.concatenate(BZ_polygon, axis = 0)

    return O, R_real, (a1, a2, l), R_reciprocal, (b1, b2, 2*pi/l), BZ_polygon
