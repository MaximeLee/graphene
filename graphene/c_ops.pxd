cdef double dot(double[:] X1, double[:] X2)
cdef double prod(double[:] X)

cdef double[:] add(double[:] X1, double[:] X2)
cdef double[:] elementwise_multiply(double[:] X1, double[:] X2)
cdef double[:] constant_multiply(double c, double[:] X)
cdef double[:] elementwise_pow(double[:] X1, short[:] X2)
