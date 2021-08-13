  
import constants
from numba import njit,int32,float32
import numpy as np

@njit
def lgq_internal(a, b, w, x, fn, args = None):
    bpa2 = (b + a) / 2.0
    bma2 = (b - a) / 2.0
    res = 0.0

    for i in range(len(x)):
        res = res + w[i] * fn(bma2 * x[i] + bpa2, args)
    return res * bma2;

@njit
def legendre_gauss_quadrature(a, b, n, fn, args = None):
    if 2 == n:
        return lgq_internal(a, b, constants.w_2, constants.x_2, fn, args) 
    if 3 == n:
        return lgq_internal(a, b, constants.w_3, constants.x_3, fn, args)
    if 4 == n:
        return lgq_internal(a, b, constants.w_4, constants.x_4, fn, args)
    if 5 == n:
        return lgq_internal(a, b, constants.w_5, constants.x_5, fn, args)
    if 6 == n:
        return lgq_internal(a, b, constants.w_6, constants.x_6, fn, args)
    if 7 == n:
        return lgq_internal(a, b, constants.w_7, constants.x_7, fn, args)
    if 8 == n:
        return lgq_internal(a, b, constants.w_8, constants.x_8, fn, args)
    if 9 == n:
        return lgq_internal(a, b, constants.w_9, constants.x_9, fn, args)
    if 10 == n:
        return lgq_internal(a, b, constants.w_10, constants.x_10, fn, args)
    print(n," unsupported")
    return np.Inf
