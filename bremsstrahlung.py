import constants
import math
from numba import njit, double, int32
import torch


# TODO: Danila, this calculation is not accurate enough
# see https://github.com/grinisrit/noa/blob/7245bb446deb2415c3ecf92c4561c065625072bc/include/noa/pms/dcs.hh#L122
@njit(double(double, double, double, int32, double))
def bremsstrahlung(K, q, A, Z, mu):
    me = constants.ELECTRON_MASS
    sqrte = 1.648721271
    phie_factor = mu / (me * me * sqrte)
    rem = 5.63588E-13 * me / mu

    BZ_n = (202.4 if Z == 1 else 182.7) * pow(Z, -1. / 3.)
    BZ_e = (446.0 if Z == 1 else 1429.0) * pow(Z, -2. / 3.)
    D_n = 1.54 * pow(A, 0.27)
    E = K + mu
    dcs_factor = 7.297182E-07 * rem * rem * Z / E

    delta_factor = 0.5 * mu * mu / E
    qe_max = E / (1. + 0.5 * mu * mu / (me * E))

    nu = q / E
    delta = delta_factor * nu / (1. - nu)
    Phi_e = 0.0
    Phi_n = math.log(BZ_n * (mu + delta * (D_n * sqrte - 2.)) / (D_n * (me + delta * sqrte * BZ_n)))
    if Phi_n < 0.0:
        Phi_n = 0.0
    if q < qe_max:
        Phi_e = math.log(BZ_e * mu / ((1. + delta * phie_factor) * (me + delta * sqrte * BZ_e)))
        if (Phi_e < 0.):
            Phi_e = 0.0
    else:
        Phi_e = 0.0

    dcs = dcs_factor * (Z * Phi_n + Phi_e) * (4. / 3. * (1. / nu - 1.) + nu)
    return 0.0 if dcs < 0.0 else dcs * 1E+3 * double(constants.AVOGADRO_NUMBER) * (mu + K) / A


@njit('(float64[:], float64[:], float64[:], float64, int32, float64)')
def _vmap_bremsstrahlung(
        result,
        kinetic_energies,
        recoil_energies,
        atomic_mass,
        atomic_number,
        particle_mass):
    n = result.shape[0]
    for i in range(n):
        result[i] = bremsstrahlung(
            kinetic_energies[i],
            recoil_energies[i],
            atomic_mass,
            atomic_number,
            particle_mass)
    return result


def vmap_bremsstrahlung(
        kinetic_energies,
        recoil_energies,
        atomic_mass,
        atomic_number,
        particle_mass):
    result = torch.zeros_like(kinetic_energies)
    _vmap_bremsstrahlung(
        result.numpy(),
        kinetic_energies.numpy(),
        recoil_energies.numpy(),
        atomic_mass,
        atomic_number,
        particle_mass)
    return result
