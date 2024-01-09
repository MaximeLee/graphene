"""compute coulomb matrix from list of Contracted Gaussians"""
from cython.parallel import prange
import numpy as np
from graphene.basis.gaussian cimport PrimitiveGaussian, ContractedGaussian

cpdef double[:,:,:,:] electron_repulsion_operator_gaussian(ContractedGaussian[:] CPG_list, double[:,::1] Chebyshev_quadrature_points_01, double[:,::1] R3_quadrature_points, double[:] subs):
    """coulomb matrix"""

    cdef PrimitiveGaussian[:] CPG_i, CPG_j, CPG_k, CPG_l
    cdef PrimitiveGaussian PG_i, PG_j, PG_k, PG_l
    cdef double[:] CPG_i_coeff_list, CPG_j_coeff_list, CPG_k_coeff_list, CPG_l_coeff_list
    cdef double[:, :, :, :] coulomb
    cdef double c_i, c_j, c_k, c_l
    cdef double tmp
    cdef int n, n_i, n_j, n_k, n_l
    cdef int i, j, k, l, ii, jj, kk, ll

    #n = len(CPG_list)
    n = CPG_list.shape[0]

    coulomb = np.zeros([n, n, n, n])

    for i in range(n):
        CPG_i = CPG_list[i].PG_list
        CPG_i_coeff_list = CPG_list[i].coeff_list
        n_i = len(CPG_i)

        for j in range(n):
            CPG_j = CPG_list[j].PG_list
            CPG_j_coeff_list = CPG_list[j].coeff_list
            n_j = len(CPG_j)

            for k in range(n):
                CPG_k = CPG_list[k].PG_list
                CPG_k_coeff_list = CPG_list[k].coeff_list
                n_k = len(CPG_k)

                for l in range(n):
                    CPG_l = CPG_list[l].PG_list
                    CPG_l_coeff_list = CPG_list[l].coeff_list
                    n_l = len(CPG_l)

                    for ii in prange(n_i, nogil=True):
                        with gil:
                            PG_i = CPG_i[ii]
                            c_i = CPG_i_coeff_list[ii]

                        for jj in range(n_j):
                            with gil:
                                PG_j = CPG_j[jj]
                                c_j = CPG_j_coeff_list[jj]

                            for kk in range(n_k):
                                with gil:
                                    PG_k = CPG_k[kk]
                                    c_k = CPG_k_coeff_list[kk]

                                for ll in range(n_l):
                                    with gil:
                                        PG_l = CPG_l[ll]
                                        c_l = CPG_l_coeff_list[ll]

                                        tmp = PrimitiveGaussian.electron_electron_int(
                                            PG_i, PG_j, PG_k, PG_l, Chebyshev_quadrature_points_01, R3_quadrature_points, subs
                                        )
                                    coulomb[i, j, k, l] += (
                                        c_i
                                        * c_j
                                        * c_k
                                        * c_l
                                        * tmp
                                    )
    return coulomb
