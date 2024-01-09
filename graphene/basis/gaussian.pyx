# cython: profile=False
# cython: np_pythran=False
# cython: cdivision=True

from libc.math cimport sqrt, pow, exp, abs
from graphene.c_ops cimport dot, prod, add, elementwise_multiply, constant_multiply, elementwise_pow
import numpy as np
cimport numpy as cnp
cnp.import_array()

#from graphene.integral.quadrature import Chebyshev_quadrature_points_01, R3_quadrature_points, subs
from graphene.constants cimport pi

ctypedef cnp.float64_t DTYPE_t

cdef class ContractedGaussianList:
    def __cinit__(self, CPG_list):
        self.CPG_list = CPG_list

    cdef ContractedGaussian get(self, int i):
        return self.CPG_list[i]

cdef class ContractedGaussian:
    
#    cdef readonly PrimitiveGaussian[:] PG_list
#    cdef readonly double[:] coeff_list

    def __cinit__(self, double[:] coeff_list, PrimitiveGaussian[:] PG_list):
        self.PG_list = PG_list
        self.coeff_list = coeff_list
        self.length = coeff_list.shape[0]

    def __len__(self):
        return self.coeff_list.shape[0]

    cdef double get_coeff(self, int i):
        return self.coeff_list[i]

    cdef PrimitiveGaussian get_PG(self, int i):
        return self.PG_list[i]

cdef class PrimitiveGaussian:

#    cdef readonly double[:] atom_position
#    cdef readonly short[3] angular_exponents
#
#    cdef readonly double alpha
#    cdef readonly double A

    def __cinit__(self, double alpha, double[:] atom_position, ex, ey, ez):
        """constructor function runned before __init__ -> self is not defined"""
        self.alpha = alpha
        self.atom_position = atom_position
        self.angular_exponents = np.array([ex, ey, ez])

    def __init__(self, double alpha, double[:] atom_position, ex, ey, ez):
        """constructor function runned after __cinit__ -> self is defined"""
        self.A = self.normalization_constant()

    cpdef tuple get_attributes(self):
        cdef double out1 = self.alpha
        cdef double[:] out2 = self.atom_position
        cdef short[:] out3 = self.angular_exponents
        cdef double out4 = self.A
        return out1, out2, out3, out4

    cdef double normalization_constant(self):
        cdef double alpha_2 = 2.0*self.alpha
        cdef short ex = self.angular_exponents[0]
        cdef short ey = self.angular_exponents[1]
        cdef short ez = self.angular_exponents[2]

        Ix = gaussian_integral(alpha_2, 2*ex)
        Iy = gaussian_integral(alpha_2, 2*ey)
        Iz = gaussian_integral(alpha_2, 2*ez)

        #return 1.0 / sqrt(Ix * Iy * Iz)
        return pow(Ix * Iy * Iz, -0.5)

    
    cpdef double overlap_int(self, PrimitiveGaussian basis2):
        """overlap integral over each coordinate"""
        cdef double[:] X1, X2, dX12
        cdef short[:] E1, E2
        cdef double Ea, A1, A2, a1, a2
        cdef double dX12_2
        cdef double I = 1.0
        cdef short xyz

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = np.subtract(X1, X2)
        dX12_2 = np.dot(dX12, dX12)

        Ea = exp(-a1 * a2 / a1p2 * dX12_2)

        for xyz in range(3):
            I *= self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, xyz)
        return A1 * A2 * Ea * I

    cdef inline double overlap_int_xyz(self, double[:] X1, double[:] X2, double a1, double a2, short[:] E1, short[:] E2, short xyz):
        """overlap integral over a coordinate"""
        cdef double a1p2 = a1 + a2
        cdef double x1 = X1[xyz]
        cdef double x2 = X2[xyz]
        cdef double x_bar = (x1 * a1 + x2 * a2) / a1p2
        cdef double I = 0.0
        cdef short e1 = E1[xyz]
        cdef short e2 = E2[xyz]
    
        cdef short i, j
    
        for i in range(e1 + 1):
            for j in range(e2 + 1):
                I += (
                    custom_comb(e1, i)
                    * custom_comb(e2, j)
                    * (x_bar - x1) ** (e1 - i)
                    * (x_bar - x2) ** (e2 - j)
                    * gaussian_integral(a1 + a2, i + j)
                )
        return I

    cpdef double kinetic_int(self, PrimitiveGaussian basis2):
        cdef double[:] X1, X2, dX12
        cdef short[:] E1, E2
        cdef double dX12_2
        cdef double A1, A2, a1, a2
        cdef short xyz, e2

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = np.subtract(X1, X2)
        dX12_2 = np.dot(dX12, dX12)

        Ea = exp(-a1 * a2 / a1p2 * dX12_2)

        I = 0.0
        for xyz in range(3):
            e2 = E2[xyz]
            if e2 == 0:
                I1 = -a2 * (
                    2.0 * a2 * self.overlap_int_xyz(X1, X2, a1, a2, E1, np.add(E2, 2), xyz)
                    - self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, xyz)
                )
        
            elif e2 == 1:
                I1 = -a2 * (
                    -3.0 * self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, xyz)
                    + 2.0 * a2 * self.overlap_int_xyz(X1, X2, a1, a2, E1, np.add(E2, 2), xyz)
                )
            else:
                Fkp2 = 4.0 * a2 ** 2.0 * self.overlap_int_xyz(X1, X2, a1, a2, E1, np.add(E2,  2), xyz)
                Fkm2 = e2 * (e2 - 1.0) * self.overlap_int_xyz(X1, X2, a1, a2, E1, np.add(E2, -2), xyz)
                Fk = -2.0 * a2 * (2.0 * e2 + 1.0) * self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, xyz)
        
                I1 = -0.5 * (Fkp2 + Fkm2 + Fk)
        
            I2 = self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, (xyz + 1) % 3)
            I3 = self.overlap_int_xyz(X1, X2, a1, a2, E1, E2, (xyz + 2) % 3)
            I += I1 * I2 * I3

        return A1 * A2 * Ea * I


    cpdef double electron_proton_int(self, basis2, double[:] R, double Z, double[:,:] Chebyshev_quadrature_points_01):
        """electron-proton integral via Chebyshev-Gauss Quadrature
        quadrature points are scaled to the interval [0,1]
        """
        cdef double[:] dX12, P
        cdef short[:] E1, E2
        cdef double A1, A2, a1, a2, a1p2, Ea
        cdef double I, I_tmp
        cdef double tk, wk
        cdef double dX12_2, PR2

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()

        a1p2 = a1 + a2

        dX12 = np.subtract(X1, X2)
        dX12_2 = np.dot(dX12, dX12)

        Ea = exp(-a1 * a2 / a1p2 * dX12_2)

        # barycenter of the two Gaussians
        #P = (a1 * X1 + a2 * X2) / (a1p2)
        #P = linear_combination(X1, a1/a1p2, X2, a2/a1p2)
        P = np.add(np.multiply(X1, a1/a1p2), np.multiply(X2, a2/a1p2))

        # squared distance between P and R (proton position)
        PR = np.subtract(P, R)
        PR2 = np.dot(PR, PR)

        # quadrature loop
        I = 0.0
        for i in range(Chebyshev_quadrature_points_01.shape[0]):
            wk = Chebyshev_quadrature_points_01[i, 0]
            tk = Chebyshev_quadrature_points_01[i, 1]
            tk2 = pow(tk,2.0)
            I_tmp = 1.0
            for xyz in range(3):
                I_tmp *= self.overlap_int_xyz(
                    P, R, a1=a1p2, a2=tk2 * a1p2 / (1.0 - tk2), E1=E1, E2=E2, xyz=xyz
                )
            I += wk / pow(1.0 - tk2, (3.0 / 2.0)) * exp(-a1p2 * tk2 * PR2) * I_tmp

        return -2.0 * A1 * A2 * Ea * Z * sqrt(a1p2 / pi) * I

    #cpdef double electron_electron_int(self, PrimitiveGaussian basis2, PrimitiveGaussian basis3, PrimitiveGaussian basis4, double[:,:] Chebyshev_quadrature_points_01, double[:,:] R3_quadrature_points, double[:,:] subs):
    cpdef double electron_electron_int(self, PrimitiveGaussian basis2, PrimitiveGaussian basis3, PrimitiveGaussian basis4, double[:,::1] Chebyshev_quadrature_points_01, double[:,::1] R3_quadrature_points, double[:] subs):
        """electron-electron integral in chemist notation (ab|cd) = integral a(R1) b(R1) |R1-R2|^(-1) c(R2) d(R2)"""
      
        cdef double[:] X1, X2, X3, X4
        cdef short[:] E1, E2, E3, E4
        cdef double a1, a2, a3, a4
        cdef double A1, A2, A3, A4

        cdef double[:] X12, X34
        cdef double[:] dX12, dX34, dX12_34, dR2
        cdef double X12_34
        cdef double E12, E34
        cdef double AA, a1p2, a3p4, p

        cdef double[:] R2_bar, R2
        cdef double I, I_tmp_t, I_tmp_R1
        cdef double wkt, tk, tk2
        cdef double angular_tmp, E2_tmp

        cdef int nk, num_R3, k2, k, R1_mu

        a1, X1, E1, A1 = self.get_attributes()
        a2, X2, E2, A2 = basis2.get_attributes()
        a3, X3, E3, A3 = basis3.get_attributes()
        a4, X4, E4, A4 = basis4.get_attributes()

        AA = A1 * A2 * A3 * A4

        a1p2 = a1 + a2
        a3p4 = a3 + a4
        p = a1p2 * a3p4 / (a1p2 + a3p4)

        dX12 = np.subtract(X1, X2)
        dX34 = np.subtract(X3, X4)
        E12 = exp(-a1 * a2 / a1p2 * dot(dX12,dX12))
        E34 = exp(-a3 * a4 / a3p4 * dot(dX34,dX34))

        X12 = add(constant_multiply(a1/a1p2, X1), constant_multiply(a2/a1p2, X2))
        X34 = add(constant_multiply(a3/a3p4, X3), constant_multiply(a4/a3p4, X4))
        dX12_34 = np.subtract(X12, X34)
        X12_34 = dot(dX12_34, dX12_34)

        I = 0.0

        # quadrature loop over Gauss transformation integration variable: t
        num_R3 = R3_quadrature_points.shape[0]
        nk = Chebyshev_quadrature_points_01.shape[0]

        for k in range(nk):
            wkt = Chebyshev_quadrature_points_01[k,0]
            tk = Chebyshev_quadrature_points_01[k,1]

            tk2 = tk**2.0
            I_tmp_t = 0.0
#            R2_bar = (a3p4 * X34 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)) * X12) / (
#                a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))
#            )
            R2_bar = add(
                constant_multiply(a3p4/(a3p4 + (a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2))), X34),
                constant_multiply((a1p2 * tk2 * p) / (a1p2 + tk2 * (p - a1p2)) / (a1p2 + tk2 * (p - a1p2)), X12)
            )

            # integrating over R2
            for k2 in range(num_R3):
                wk_xyz2 = R3_quadrature_points[k2, 0]
                R2 = np.array(R3_quadrature_points[k2, 1:4])
                subs2 = subs[k2]

                R2 = add(R2, R2_bar) # centering quadrature pts
                I_tmp_R1 = 1.0

                # integrating over R1
                for R1_mu in range(3):
                    I_tmp_R1 *= self.overlap_int_xyz(
                        X12, R2, a1p2, tk2*p/(1-tk2), E1, E2, R1_mu
                    )

                dR2 = np.subtract(R2,R2_bar)
                angular_tmp = prod(elementwise_multiply(
                    elementwise_pow(np.subtract(R2,X3), E3), 
                    elementwise_pow(np.subtract(R2,X4), E4))
                )
                E2_tmp = exp(-a1p2 * a3p4 / (a1p2 + tk2 * (p - a1p2)) * dot(dR2, dR2))
                I_tmp_t += 4.0 * pi * wk_xyz2 * subs2 * angular_tmp * E2_tmp * exp(-p * tk2 * X12_34) * I_tmp_R1

            #I += wkt / (1.0 - tk2) ** (3.0 / 2.0) * I_tmp_t
            I += wkt / pow(1.0 - tk2, 3.0 / 2.0) * I_tmp_t
        return 2.0 * AA * E12 * E34 * sqrt(p / pi) * I

###############
# HELPER FUNCTIONS
###############
cdef inline double custom_comb(short n, short k):
    """custom combinatory coefficient"""
    cdef short result = 1
    cdef short i
    
    if k > n - k:
        k = n - k
    for i in range(k):
        result *= n - i
        result //= i + 1
    return <double>result

cpdef inline double gaussian_integral(double alpha, short n):
    """integral over R of x^n exp(-alpha x^2)"""
    cdef short nn
    cdef double ii
    cdef short i
    cdef double prod = 1.0
    
    if n % 2 == 1:
        return 0.0
    nn = n // 2 
    for i in range(nn):
        prod = prod * (2*i+1)
    return sqrt(pi/alpha) * pow(2.0 * alpha, -nn) * prod

