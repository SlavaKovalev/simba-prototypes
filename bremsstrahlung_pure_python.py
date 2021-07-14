from math import log
import torch
import constants 

def dcs_bremsstrahlung(Z, A, mu, K, q):
    #print("kinetic energy ", K)
    #print("recoil energy ", q)
    me = constants.ELECTRON_MASS
    #print("ELECTRON_MASS ", constants.ELECTRON_MASS)
    sqrte = 1.648721271
    #print("sqrte ", sqrte)
    #print("muon mass ", mu)
    phie_factor = mu / (me * me * sqrte)
    #print("phie_factor ", format(phie_factor, '.15f'))

    rem = 5.63588E-13 * me / mu
    #print("rem ", format(rem, '.15f'))

    BZ_n = (202.4 if Z == 1 else 182.7) * pow(Z, -1./3.)
    #print("BZ_n ", format(BZ_n));
    BZ_e = (446.0 if Z == 1.0 else 1429.0) * pow(Z, -2. / 3.)
    #print("BZ_e ", format(BZ_e, '.15f'))
    D_n = 1.54 * pow(A, 0.27)
    #print("D_n ", format(D_n, '.15f'))
    E = K + mu
    #print("E ", format(E, '.15f'))
    dcs_factor = 7.297182E-07 * rem * rem * Z / E
    #print("dcs_factor ", format(dcs_factor, '.15f'))

    delta_factor = 0.5 * mu * mu / E
    #print("delta_factor ", format(delta_factor, '.15f'))
    qe_max = E / 1. + 0.5 * mu * mu / (me * E)
    #print("qe_max ", format(qe_max, '.15f'))

    nu = q / E
    #print("nu ", format(nu, '.15f'))
    delta = delta_factor * nu / (1. - nu)
    #print("delta ", format(delta, '.15f'))
    Phi_n = 0.
    Phi_e = 0.
    Phi_n = log(BZ_n * (mu + delta * (D_n * sqrte - 2.)) / (D_n * (me + delta * sqrte * BZ_n)))
    #print("Phi_n ", format(Phi_n, '.15f'))
    if Phi_n < 0.:
        Phi_n = 0.
    if q < qe_max:
        Phi_e = log(BZ_e * mu / ((1. + delta * phie_factor) * (me + delta * sqrte * BZ_e)))
    if Phi_e < 0.:
        Phi_e = 0.
    #print("Phi_e ", format(Phi_e, '.15f'))

    dcs = dcs_factor * (Z * Phi_n + Phi_e) * (4. / 3. * (1. / nu - 1.) + nu)
    #print("dcs ", format(dcs, '.15f'))
    if dcs < 0.:
        dcs = 0.
    else:
        dcs = dcs * 1E+03 * constants.AVOGADRO_NUMBER * (mu + K) / A
    #print(format(dcs, '.15f'))
    return  dcs


def _vmap_bremsstrahlung(
        result,
        kinetic_energies,
        recoil_energies,
        atomic_mass,
        atomic_number,
        particle_mass):
    n = result.shape[0]
    for i in range(n):
        result[i] = dcs_bremsstrahlung(
            atomic_number,
            atomic_mass,
            particle_mass,
            kinetic_energies[i],
            recoil_energies[i])
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

