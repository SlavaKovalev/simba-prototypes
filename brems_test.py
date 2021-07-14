import torch
import constants
import utils
#import bremsstrahlung_pure_python
import bremsstrahlung
import os
import sys
import test_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        splited_path = sys.argv[0].split('/')
        print(f'Usage: python {splited_path[len(splited_path)-1]} <path_to_noa-test-data>')
        exit(0)
    data = test_data.load_brems_test_data(sys.argv[1])

    brems = bremsstrahlung.vmap_bremsstrahlung(
            data.kinetic_energies,
            data.recoil_energies,
            constants.STANDARD_ROCK.A,
            constants.STANDARD_ROCK.Z,
            constants.MUON_MASS)

    print(utils.relative_error(brems, data.pumas_brems))
