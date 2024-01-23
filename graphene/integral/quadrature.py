import copy as cp
from copy import deepcopy as dcp
import os
import warnings
import numpy as np

from graphene.constants import pi


def tensor_prod(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Cartesian/tensor product of two arrays"""
    n1 = len(X1)
    n2 = len(X2)

    Xprod = np.concatenate([np.tile(X1, (n2, 1)), np.repeat(X2, n1, axis=0)], axis=1)

    return Xprod

def get_quadrature_points(n=None, precision=11, quadrature_type="gauss_chebyshev_2"):
    """quadrature points :
    - first column : weights
    - second and next columns : abscissa on the interval/domain

    Supported quadrature types :
    - gauss_chebyshev_2 : Gauss Chebyshev quadrature of the second kind
    - lebedev : https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
    """
    if quadrature_type == "gauss_chebyshev_2":
        i = np.arange(1.0, n + 1.0)
        sin_i = np.sin(i * pi / (n + 1.0))
        cos_i = np.cos(i * pi / (n + 1.0))

        abscissa = (n + 1.0 - 2.0 * i) / (n + 1.0) + 2.0 / pi * (
            1.0 + 2.0 / 3.0 * sin_i**2
        ) * cos_i * sin_i
        abscissa = abscissa.reshape(-1, 1)

        weights = 16.0 / 3.0 / (n + 1.0) * sin_i**4.0
        weights = weights.reshape(-1, 1)

    elif quadrature_type == "lebedev":
        precision = str(precision)
        precision = '0'*(3-len(precision)) + precision
        data = np.loadtxt(f"{os.path.dirname(__file__)}/lebedev_quadrature/lebedev_{precision}.txt")

        abscissa = data[:, :2] * pi / 180.0
        weights = data[:, 2:3]

    else:
        raise ValueError("Quadrature type not recognized/supported.")
    return np.concatenate([weights, abscissa], axis=1)

#############################
# Chebyshev quadrature points
#############################
nquad = 80

# pts on [-1,1]
Chebyshev_quadrature_points = get_quadrature_points(
    n=nquad, quadrature_type="gauss_chebyshev_2"
)
Chebyshev_weights = Chebyshev_quadrature_points[:, 0:1]
Chebyshev_abscissa = Chebyshev_quadrature_points[:, 1:2]

# pts on [0,1]
Chebyshev_quadrature_points_01 = cp.deepcopy(Chebyshev_quadrature_points)
Chebyshev_quadrature_points_01[:, 1] = (
    Chebyshev_quadrature_points_01[:, 1] + 1.0
) / 2.0
# sum weights = 1
Chebyshev_quadrature_points_01[:, 0] /= 2.0

#############################
# Lebedev quadrature points
#############################
Lebedev_quadrature = get_quadrature_points(precision=13, quadrature_type = "lebedev")
Lebedev_weights = Lebedev_quadrature[:, 0:1]
Lebedev_abscissa = Lebedev_quadrature[:, 1:]

#############################
# R3 integration pts
#############################

# variable substitution from [-1,1] to [0, +infty[
# Briggs-slater radius for Carbon
rm = 0.7/2.0
# quadrature radius
R_quadrature = rm* (1.0 + Chebyshev_abscissa) / (1.0 - Chebyshev_abscissa)

R3_weights_quadrature = np.prod(
    tensor_prod(Chebyshev_weights, Lebedev_weights), axis=1, keepdims=True
)

R3_points_quadrature = tensor_prod(R_quadrature, Lebedev_abscissa)
R = dcp(R3_points_quadrature[:, 0])
Theta = dcp(R3_points_quadrature[:, 1])
Phi = dcp(R3_points_quadrature[:, 2])

X = R * np.cos(Theta) * np.sin(Phi)
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Phi)

R3_points_quadrature[:, 0] = X
R3_points_quadrature[:, 1] = Y
R3_points_quadrature[:, 2] = Z

R3_quadrature_points = np.hstack([R3_weights_quadrature, R3_points_quadrature])

Mu_int = (R - rm) / (R + rm)

# to multiply the integrand with because of the variable substitution
subs = (R**2 * 2.0 * rm / (1.0 - Mu_int) ** 2).reshape(-1, 1)

del (R3_points_quadrature, R3_weights_quadrature)
del (Chebyshev_weights, Lebedev_weights)
del (Chebyshev_abscissa, Lebedev_abscissa)
del (R, Theta, Phi)
del (X, Y, Z)
