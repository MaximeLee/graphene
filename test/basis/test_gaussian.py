import time
from graphene.basis.gaussian import PrimitiveGaussian
import numpy as np
import math as m
from constants import *
from utils import *
from graphene.integral.quadrature import *

pi = np.pi

class TestPrimitiveGaussian:

    def test_overlap_int(self):
        # 1s - 1s
        alpha = pi/2
        x1 = 5.0
        x2 = 0.0
        X1 = np.array([[x1, 0, 0]])
        X2 = np.array([[x2, 0, 0]])
        PG1 = PrimitiveGaussian(alpha=alpha, atom_position = X1.flatten(), ex=0, ey=0, ez=0)
        PG2 = PrimitiveGaussian(alpha=alpha, atom_position = X2.flatten(), ex=0, ey=0, ez=0)
        integral = PrimitiveGaussian.overlap_int(PG1,PG2)
        assert np.isclose(integral,m.exp(-pi/4*(x1-x2)**2))

        x3 = 54.0
        PG3 = PrimitiveGaussian(1.0, np.array([x3, 0, 0]), 0, 0, 0)
        integral = PG2.overlap_int(PG3)
        assert np.isclose(integral,(2/pi)**(3/4)*m.exp(-1/(1+2/pi)*(x2-x3)**2))

    def test_kinetic_int_1s(self):
        overlap = PG1_1s.overlap_int(PG2_1s)
        I_int = overlap * a2 * (-3.0 * a2 / a1p2 - 2.0 * a2 * np.linalg.norm(X_bar-X2)**2.0 + 3.0)
        assert m.isclose(PG1_1s.kinetic_int(PG2_1s), I_int)

    def test_electron_proton_int_1s(self):
        """electron attraction integrals on hydrogen type orbitals"""
        I_int = - 2.0 * A1 * A2 * (pi / a1p2) * e12 * boys(a1p2 * np.linalg.norm(R-X_bar)**2.0)
        assert m.isclose(PG1_1s.electron_proton_int(PG2_1s, R.flatten(), 1., Chebyshev_quadrature_points_01), I_int)

    def test_electron_electron_int(self):
        """
        alpha = pi/2
        x1 = np.array([[5.0, 0.0, 0.0]])
        x2 = np.array([[0.0, 0.0, 0.0]])
        PG1 = PrimitiveGaussian(alpha=alpha, atom_position = x1)
        PG2 = PrimitiveGaussian(alpha=alpha, atom_position = x2)
        """
        t1 = time.time()
        integral = PG1_1s.electron_electron_int(PG2_1s,PG3_1s,PG4_1s,Chebyshev_quadrature_points_01, R3_quadrature_points, subs)
        t2 = time.time()
        print(t2-t1)

        I_int = 2 * A1 * A2 * A3 * A4 * e12 * e34 * m.sqrt(p/pi) * (pi**2/a1p2/a3p4)**(3/2) * boys(p*Q2)
        assert np.isclose(integral, I_int)
