import numpy as np
from copy import deepcopy as dcp

theta = np.pi/3.0

l_eq = 0.142
def mesh(nx, ny, l = l_eq, centered = True):
    """generate hexagonal mesh with nx x ny cells"""

    ny = ny + 1 if ny%2==0 else ny

    # distance between atoms on the same row
    dx = l * np.sin(theta) * 2.0 
    # distance between atoms in diagonal
    dy = l * np.cos(theta) 


    a1 = np.array([dx/2,  dy])
    a2 = np.array([dx/2, -dy])

    # first row
    x_row = dx * np.arange(nx).reshape(-1, 1)
    y_row = np.zeros_like(x_row)
    X = np.hstack([x_row, y_row])

    # x-axis position possibilities
    x_row_1 = dx * np.arange(nx).reshape(-1, 1)
    x_row_2 = dx * (np.arange(nx+1) - 0.5).reshape(-1,1)

    y_row_1 = np.ones_like(x_row_1)
    y_row_2 = np.ones_like(x_row_2)

    # current y pos
    y = 0.0

    for i in range(ny):
        if i%2==0:
            X_tmp_1 = np.hstack([x_row_2, y_row_2*(y + dy) ])
            y += dy

            X_tmp_2 = np.hstack([x_row_2, y_row_2*(y + l)])
            y += l

            X_tmp_3 = np.hstack([x_row_1, y_row_1*(y +dy)])
            y += dy

            X_tmp = np.concatenate([X_tmp_1, X_tmp_2, X_tmp_3], axis=0)
        else:
            X_tmp_1 = np.hstack([x_row_1, y_row_1*(y + l) ])
            y += l

            X_tmp = X_tmp_1

        X = np.concatenate([X, X_tmp], axis = 0)

    if centered:
        X -= np.mean(X, axis = 0, keepdims=True)
    return X, a1, a2

