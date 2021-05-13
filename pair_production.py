#!/bin/python3
from constants import ELECTRON_MASS
import math
'''
The default Bremsstrahlung differential cross section.
  @param Z       The charge number of the target atom.
  @param A       The mass number of the target atom.
  @param mu      The projectile rest mass, in GeV
  @param K       The projectile initial kinetic energy.
  @param q       The kinetic energy lost to the photon.
  @return The corresponding value of the atomic DCS, in m^2 / GeV.
The differential cross section is computed following R.P. Kokoulin's formulae taken from the Geant4 Physics Reference Manual. '''
def pair_production(Z, A_, mass, K, q):
    '''
    Coefficients for the Gaussian quadrature from:
    https://pomax.github.io/bezierinfo/legendre-gauss.html.
    '''
    N_GQ = 8
    xGQ = { 0.01985507, 0.10166676, 0.2372338,
                0.40828268, 0.59171732, 0.7627662, 0.89833324, 0.98014493 }
    wGQ = { 0.05061427, 0.11119052, 0.15685332,
                0.18134189, 0.18134189, 0.15685332, 0.11119052, 0.05061427 }
    # Check the bounds of the energy transfer.
    if q <= 4.0 * ELECTRON_MASS:
        return 0.0
    sqrte = 1.6487212707
    Z13 = math.pow(Z, 1. / 3.)
    if q >= K + mass * (1.0 - 0.75 * sqrte * Z13):
        return 0.
    # Precompute some constant factors for the integration.
    nu = q / (K + mass)
    r = mass / ELECTRON_MASS
    beta = 0.5 * nu * nu / (1.0 - nu)
    xi_factor = 0.5 * r * r * beta
    A = 202.4 if Z == 1.0 else 183.0
    AZ13 = A / Z13
    cL = 2. * sqrte * ELECTRON_MASS * AZ13
    cLe = 2.25 * Z13 * Z13 / (r * r)
    # Compute the bound for the integral.
    gamma = 1. + K / mass
    x0 = 4.0 * ELECTRON_MASS / q
    x1 = 6.0 / (gamma * (gamma - q / mass))
    argmin = (x0 + 2. * (1. - x0) * x1) / (1. + (1. - x1) * sqrt(1. - x0))
    if (argmin >= 1.) or (argmin <= 0.):
        return 0.0
    tmin = math.log(argmin)
    # Compute the integral over t = ln(1-rho).
    I = 0.0
    i = 0
    for i in range(8):
        eps = math.exp(xGQ[i] * tmin)
        rho = 1. - eps
        rho2 = rho * rho
        rho21 = eps * (2. - eps)
        xi = xi_factor * rho21
        xi_i = 1. / xi
        #Compute the e-term. 
        Be = 0.
        if xi >= 1E+03:
            Be = 0.5 * xi_i * ((3 - rho2) + 2. * beta * (1. + rho2))
        else:
            Be = ((2. + rho2) * (1. + beta) + xi * (3. + rho2)) * math.log(1. + xi_i) + (rho21 - beta) / (1. + xi) - 3. - rho2
        Ye = (5. - rho2 + 4. * beta * (1. + rho2)) / (2. * (1. + 3. * beta) * log(3. + xi_i) - rho2 - 2. * beta * (2. - rho2))
        xe = (1. + xi) * (1. + Ye)
        cLi = cL / rho21
        Le = math.log(AZ13 * sqrt(xe) * q / (q + cLi * xe)) - 0.5 * math.log(1. + cLe * xe)
        Phi_e = Be * Le
        if Phi_e < 0.:
            Phi_e = 0.
        # Compute the mu-term.
        Bmu = 0.
        if xi <= 1E-03:
            Bmu = 0.5 * xi * (5. - rho2 + beta * (3. + rho2))
        else:
            Bmu = ((1. + rho2) * (1. + 1.5 * beta) - xi_i * (1. + 2. * beta) * rho21) * math.log(1. + xi) + xi * (rho21 - beta) / (1. + xi) + (1. + 2. * beta) * rho21
        Ymu = (4. + rho2 + 3. * beta * (1. + rho2)) / ((1. + rho2) * (1.5 + 2. * beta) * math.log(3. + xi) + 1. - 1.5 * rho2)
        xmu = (1. + xi) * (1. + Ymu)
        Lmu = math.log(r * AZ13 * q / (1.5 * Z13 * (q + cLi * xmu)))
        Phi_mu = Bmu * Lmu
        if Phi_mu < 0.:
            Phi_mu = 0.
        # Update the t-integral.
        I -= (Phi_e + Phi_mu / (r * r)) * (1. - rho) * wGQ[i] * tmin
    # Atomic electrons form factor.
    zeta = 0.
    if gamma <= 35.:
        zeta = 0.
    else:
        gamma1 = 0.
        gamma2 = 0.
        if Z == 1.:
            gamma1 = 4.4E-05
            gamma2 = 4.8E-05
        else:
            gamma1 = 1.95E-05
            gamma2 = 5.30E-05
        zeta = 0.073 * math.log(gamma / (1. + gamma1 * gamma * Z13 * Z13)) - 0.26
        if zeta <= 0.:
            zeta = 0.
        else:
            zeta /= 0.058 * math.log(gamma / (1. + gamma2 * gamma * Z13)) - 0.14
    # Gather the results and return the macroscopic DCS.
    E = K + mass
    dcs = 1.794664E-34 * Z * (Z + zeta) * (E - q) * I / (q * E)
    return 0 if dcs < 0. else dcs

