import math as m
import numpy as np

from graphene.basis.gaussian import PrimitiveGaussian

np.random.seed(12)

pi = m.pi

# for single PG integration tests
alpha = 5.0
G_alpha = m.sqrt(pi / alpha)

# for multiple PG integration tests
X1 = np.zeros([1, 3])
a1 = 5.0

X2 = np.zeros([1, 3]) + 0.1
a2 = 6.0

a1p2 = a1 + a2

X_bar = X12 = (X1 * a1 + X2 * a2) / (a1 + a2)

e12 = m.exp(-a1 * a2 / a1p2 * np.linalg.norm(X1 - X2) ** 2.0)

G12 = m.sqrt(pi / a1p2)

# proton position
R = np.random.uniform(size=[1,3])

# for electron-electron integrals
X3 = np.random.uniform(size=[1, 3]) + 0.1
a3 = 2.0

X4 = np.random.uniform(size=[1, 3]) + 0.3
a4 = 1.0

a3p4 = a3 + a4

X34 = (X3*a3 + X4*a4)/(a3p4)

e34 = m.exp(-a3 * a4 / a3p4 * np.linalg.norm(X3 - X4) ** 2.0)

p = a1p2 * a3p4 / (a1p2 + a3p4)

Q2 = np.linalg.norm(X34-X12)**2.0

### PG
PG1_1s = PrimitiveGaussian(a1, X1.flatten(), 0, 0, 0)
A1 = PG1_1s.A
E1 = PG1_1s.angular_exponents

PG2_1s = PrimitiveGaussian(a2, X2.flatten(), 0, 0, 0)
A2 = PG2_1s.A
exponents2 = E2 = PG2_1s.angular_exponents

PG3_1s = PrimitiveGaussian(a3, X3.flatten(), 0, 0, 0)
A3 = PG3_1s.A
exponents3 = E3 = PG3_1s.angular_exponents

PG4_1s = PrimitiveGaussian(a4, X4.flatten(), 0, 0, 0)
A4 = PG4_1s.A
exponents4 = E4 = PG4_1s.angular_exponents
