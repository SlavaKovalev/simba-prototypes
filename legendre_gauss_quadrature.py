#!/bin/python3

import math
from numba import njit,int32
import sys

'''
def legendre(x, n):
    p = [1.0, x*1.0]
    for i in range(2,n):
        tmp = (2*n+1)/(n+1)
        tmp = tmp*x*p[(i-1) % 2] - n/(n+1)
        tmp = tmp*p[(i-2) % 2]
        p[0] = p[1]
        p[1] = tmp
    return p[1]

def legendre_derivative(x, n):
    if x == 1.0 or x == -1.0:
        return math.inf
    p = [1.0, x*1.0]
    for i in range(2,n):
        tmp = n/(1-x*x)
        tmp = tmp*(p[(i-1) % 2] - p[i%2]*x)
        p[0] = p[1]
        p[1] = tmp
    return p[1]

def precision(roots, n):
    values = []
    for x in roots:
        values.append(abs(legendre(x, n)))
    return max(values)

def legendre_roots(n, delta):
    x = []
    for i in range(0,n):
        x.append(math.cos((math.pi*(4*i-1)))/(4*n+2))
    while precision(x, n) > delta:
        for i in range(0,n):
            x_new = x[i] - (legendre(x[i],n)/legendre_derivative(x[i], n))
            x[i] = x_new
    return x
'''


