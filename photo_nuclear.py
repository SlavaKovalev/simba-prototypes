import constants
import lgq
import math
from numba import njit

# ALLM97 parameterisation of the proton structure function, F2.
#  
# @param x       The fractional kinetic energy lost to the photon.
# @param Q2      The negative four momentum squared.
# @return The corresponding value of the proton structure function, F2.
#  
# References:
#
# DESY 97-251 [arXiv:hep-ph/9712415].
# https://github.com/grinisrit/noa/blob/7245bb446deb2415c3ecf92c4561c065625072bc/include/noa/pms/dcs.hh#L289
@njit
def dcs_photonuclear_f2_allm(x, Q2):
    m02 = 0.31985
    mP2 = 49.457
    mR2 = 0.15052
    Q02 = 0.52544
    Lambda2 = 0.06527

    cP1 = 0.28067
    cP2 = 0.22291
    cP3 = 2.1979
    aP1 = -0.0808
    aP2 = -0.44812
    aP3 = 1.1709
    bP1 = 0.36292
    bP2 = 1.8917
    bP3 = 1.8439

    cR1 = 0.80107
    cR2 = 0.97307
    cR3 = 3.4942
    aR1 = 0.58400
    aR2 = 0.37888
    aR3 = 2.6063
    bR1 = 0.01147
    bR2 = 3.7582
    bR3 = 0.49338

    M2 = 0.8803505929
    W2 = M2 + Q2 * (1.0 / x - 1.0)
    t = math.log(math.log((Q2 + Q02) / Lambda2) / math.log(Q02 / Lambda2))
    xP = (Q2 + mP2) / (Q2 + mP2 + W2 - M2)
    xR = (Q2 + mR2) / (Q2 + mR2 + W2 - M2)
    lnt = math.log(t)
    cP = cP1 + (cP1 - cP2) * (1.0 / (1.0 + math.exp(cP3 * lnt)) - 1.0)
    aP = aP1 + (aP1 - aP2) * (1.0 / (1.0 + math.exp(aP3 * lnt)) - 1.0)
    bP = bP1 + bP2 * math.exp(bP3 * lnt)
    cR = cR1 + cR2 * math.exp(cR3 * lnt)
    aR = aR1 + aR2 * math.exp(aR3 * lnt)
    bR = bR1 + bR2 * math.exp(bR3 * lnt)

    F2P = cP * math.exp(aP * math.log(xP) + bP * math.log(1 - x))
    F2R = cR * math.exp(aR * math.log(xR) + bR * math.log(1 - x))

    return Q2 / (Q2 + m02) * (F2P + F2R)

# The F2 structure function for atomic weight A.
#
# @param x       The fractional kinetic energy lost to the photon.
# @param F2p     The proton structure function, F2.
# @param A       The atomic weight.
# @return The corresponding value of the structure function, F2.
#
# The F2 structure function for a nucleus of atomic weight A is computed
# according to DRSS, including a Shadowing factor.
#
# References:
# Dutta et al., Phys.Rev. D63 (2001) 094020 [arXiv:hep-ph/0012350].
@njit
def dcs_photonuclear_f2a_drss(x, F2p, A):
    a = 1.0
    if (x < 0.0014):
        a = math.exp(-0.1 * math.log(A))
    elif (x < 0.04):
        a = math.exp((0.069 * math.log10(x) + 0.097) * math.log(A))

    return (0.5 * A * a * (2.0 + x * (-1.85 + x * (2.45 + x * (-2.35 + x)))) * F2p)


# The R ratio of longitudinal to transverse structure functions.
# 
# @param x       The fractional kinetic energy lost to the photon.
# @param Q2      The negative four momentum squared.
# 
# References:
# Whitlow, SLAC-PUB-5284.
#
def dcs_photonuclear_r_whitlow(x, Q2):
    q2 = Q2
    if (Q2 < 0.3): 
        q2 = 0.3

    theta = 1 + 12.0 * q2 / (1.0 + q2) * 0.015625 / (0.015625 + x * x)

    return (0.635 / math.log(q2 / 0.04) * theta + 0.5747 / q2 -
            0.3534 / (0.09 + q2 * q2))


# The doubly differential cross sections d^2S/(dq*dQ2) for photonuclear
# interactions.
#  
# @param ml      The projectile mass.
# @param A       The target atomic weight.
# @param K       The projectile initial kinetic energy.
# @param q       The kinetic energy lost to the photon.
# @param Q2      The negative four momentum squared.
# @return The doubly differential cross section in m^2/kg/GeV^3.
#  
# References:
# Dutta et al., Phys.Rev. D63 (2001) 094020 [arXiv:hep-ph/0012350].
#
@njit
def dcs_photonuclear_d2(A, ml, K, q, Q2):
    cf = 2.603096E-35
    M = 0.931494
    E = K + ml

    y = q / E
    x = 0.5 * Q2 / (M * q)
    F2p = dcs_photonuclear_f2_allm(x, Q2)
    F2A = dcs_photonuclear_f2a_drss(x, F2p, A)
    R = dcs_photonuclear_r_whitlow(x, Q2)

    dds = (1 - y + 0.5 * (1 - 2 * ml * ml / Q2) *(y * y + Q2 / (E * E)) / (1 + R)) /(Q2 * Q2) - 0.25 / (E * E * Q2)

    return cf * F2A * dds / q


# Utility function for checking the consistency of the Photonuclear model.
# @param K The kinetic energy.
# @param q The kinetic energy lost to the photon.
# @return `0` if the model is valid.
#
# Check for the limit of the PDG model. Below this kinetic transfer a
# tabulation is used, which we don't do. Therefore we set the cross-section to
# zero below 1 GeV where it deviates from the model. In addition, for x < 2E-03
# the model is unstable and can lead to osccilations and negative cross-section
# values.
@njit
def dcs_photonuclear_check(double K, double q):
    return (q < 1.) || (q < 2E-03 * K)


@njit
def integrand(t, args = None):
    Q2 = math.math.exp(args['pQ2c'] + 0.5 * args['dpQ2'] * t)
    return dcs_photonuclear_d2(args['A'], args['mu'], args['K'], args['q'], Q2) * Q2


#
# Calculation the photonuclear differential cross section.
#
# @param Z       The charge number of the target atom.
# @param A       The mass number of the target atom.
# @param mu      The projectile rest mass, in GeV
# @param K       The projectile initial kinetic energy.
# @param q       The kinetic energy lost to the photon.
# @return The corresponding value of the atomic DCS, in m^2 / GeV.

# The photonuclear differential cross-section is computed following DRSS,
# with ALLM97 parameterisation of the structure function F2.
#
# References:
# Dutta et al., Phys.Rev. D63 (2001) 094020 [arXiv:hep-ph/0012350].
# https://github.com/SlavaKovalev/noa/blob/master/include/noa/pms/dcs.hh#L404
@njit
def photonuclear(Z, A, mu, K, q):
    if (dcs_photonuclear_check(K, q))
        return 0.
    
    M = 0.931494
    mpi = 0.134977
    E = K + mu

    ds = 0.
    if (q >= (E - mu)) or (q <= (mpi * (1.0 + 0.5 * mpi / M))):
        return ds

     y = q / E
     Q2min = mu * mu * y * y / (1 - y)
     Q2max = 2.0 * M * (q - mpi) - mpi * mpi
    if (Q2max < Q2min) | (Q2min < 0):
        return ds

    # Set the binning.
    pQ2min = math.log(Q2min)
    pQ2max = math.log(Q2max)
    dpQ2 = pQ2max - pQ2min
    pQ2c = 0.5 * (pQ2max + pQ2min)

    # Integrate the doubly differential cross-section over Q2 using
    # a Gaussian quadrature. Note that 9 points are enough to get a
    # better than 0.1 % accuracy.
    args = Dict.empty(key_type = types.unicode_type, value_type = types.float64)
    args['A']    = A
    args['mu']   = mu
    args['K']    = K
    args['q']    = q
    args['pQ2c'] = pQ2c
    args['dpQ2'] = dpQ2
    ds = lgq.legendre_gauss_quadrature(0., 1., 9, integrand, args)
    return 0. if ds < 0 .else 0.5 * ds * dpQ2 * 1E+03 * constants.AVOGADRO_NUMBER * (mu + K) / A

