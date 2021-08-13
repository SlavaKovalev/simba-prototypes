import sys
import lgq
from numba import njit

@njit
def x2_multiply_3(x):
    return 3*x*x

print(lgq.legendre_gauss_quadrature(1, 2, int(sys.argv[1]), x2_multiply_3))
