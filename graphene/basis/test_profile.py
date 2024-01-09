import time
import pstats, cProfile
from graphene.basis.gaussian import PrimitiveGaussian
from graphene.integral.quadrature import *
import numpy as np

n = 10

# Defining PG
a1 = np.random.uniform(size=1)[0]
X1 = np.random.uniform(size=3)
PG1 = PrimitiveGaussian(a1, X1, 0, 0, 0)

a2 = np.random.uniform(size=1)[0]
X2 = np.random.uniform(size=3)
PG2 = PrimitiveGaussian(a2, X2, 0, 0, 0)

a3 = np.random.uniform(size=1)[0]
X3 = np.random.uniform(size=3)
PG3 = PrimitiveGaussian(a3, X3, 0, 0, 0)

a4 = np.random.uniform(size=1)[0]
X4 = np.random.uniform(size=3)
PG4 = PrimitiveGaussian(a4, X4, 0, 0, 0)

# test repulsion integrals
t1 = time.time()
with cProfile.Profile() as profile:
    for _ in range(n):
        PG1.electron_electron_int(PG2, PG3, PG4, Chebyshev_quadrature_points_01, R3_quadrature_points, subs)

t2 = time.time()
print(f"time = {t2-t1} s")
results = pstats.Stats(profile)
results.sort_stats(pstats.SortKey.TIME)
results.print_stats()
results.dump_stats("results2.prof")

