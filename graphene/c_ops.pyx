from libc.math cimport pow
cimport cython

cdef inline double dot(double[:] X1, double[:] X2):
    cdef double d = 0.0
    cdef int n = X1.shape[0]
    cdef int i

    for i in range(n):
        d += X1[i] * X2[i]    
    return d

cdef inline double prod(double[:] X):
    cdef double p = 1.0
    cdef int n = X.shape[0]
    cdef int i

    for i in range(n):
        p = p * X[i]
    return p

cdef inline double[:] add(double[:] X1, double[:] X2):
    cdef double[:] a
    cdef int n = X1.shape[0]
    cdef int i

    a = X1
    for i in range(n):
        a[i] += X2[i]
    return a

cdef inline double[:] elementwise_multiply(double[:] X1, double[:] X2):
    cdef double[:] res
    cdef int n = X1.shape[0]
    cdef int i

    res = X1
    for i in range(n):
        res[i] *= X2[i]
    return res

cdef inline double[:] constant_multiply(double c, double[:] X):
    cdef double[:] res
    cdef int n = X.shape[0]
    cdef int i

    res = X
    for i in range(n):
        res[i] *= c
    return res

cdef inline double[:] elementwise_pow(double[:] X1, short[:] X2):
    cdef double[:] res
    cdef int n = X1.shape[0]
    cdef int i

    res = X1
    for i in range(n):
        res[i] = pow(res[i], X2[i])
    return res
