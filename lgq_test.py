import sys
import lgq
from numba import njit

@njit
def x2_multiply_3(x):
    return 3*x*x

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: 'python lgq_test.py <n>' where n is from [2, 10]")
        exit(1)
    print(lgq.legendre_gauss_quadrature(1, 2, int(sys.argv[1]), x2_multiply_3))
