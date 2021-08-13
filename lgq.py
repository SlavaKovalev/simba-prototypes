import constants
from numba import njit,int32,float32
import numpy as np

@njit
def lgq_internal(a, b, w, x, fn):
    bpa2 = (b + a) / 2.0
    bma2 = (b - a) / 2.0
    res = 0.0

    for i in range(len(x)):
        res = res + w[i] * fn(bma2 * x[i] + bpa2)
    return res * bma2;

@njit
def legendre_gauss_quadrature(a, b, n, fn):
    if 2 == n:
        return lgq_internal(a, b, constants.w_2, constants.x_2, fn) 
    if 3 == n:
        return lgq_internal(a, b, constants.w_3, constants.x_3, fn)
    if 4 == n:
        return lgq_internal(a, b, constants.w_4, constants.x_4, fn)
    if 5 == n:
        return lgq_internal(a, b, constants.w_5, constants.x_5, fn)
    if 6 == n:
        return lgq_internal(a, b, constants.w_6, constants.x_6, fn)
    if 7 == n:
        return lgq_internal(a, b, constants.w_7, constants.x_7, fn)
    if 8 == n:
        return lgq_internal(a, b, constants.w_8, constants.x_8, fn)
    if 9 == n:
        return lgq_internal(a, b, constants.w_9, constants.x_9, fn)
    if 10 == n:
        return lgq_internal(a, b, constants.w_10, constants.x_10, fn)
    print(n," unsupported")
    return np.Inf
