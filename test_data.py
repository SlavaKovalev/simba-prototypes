import os
import torch

class brems_test_data:
    def __init__(self, kinetic_energies, recoil_energies, pumas_brems):
        self.kinetic_energies = kinetic_energies
        self.recoil_energies = recoil_energies
        self.pumas_brems = pumas_brems

def convert_brems_data_to_csv(path_to_noa_test_data):
    data = load_bems_test_data(path_to_noa_test_data)
    with open('brems.csv', 'w') as f:
        f.write('kinetic_energies,recoil_energies,pumas_brems\n')
        for i in range(len(data.kinetic_energies)):
            f.write(f'{data.kinetic_energies[i]},{data.recoil_energies[i]},{data.pumas_brems[i]}\n')

def load_brems_test_data(path_to_noa_test_data):
    return brems_test_data(
            list(torch.jit.load(os.path.join(path_to_noa_test_data, 'pms/kinetic_energies.pt')).parameters())[0],
            list(torch.jit.load(os.path.join(path_to_noa_test_data,'pms/recoil_energies.pt')).parameters())[0],
            list(torch.jit.load(os.path.join(path_to_noa_test_data, 'pms/pumas_brems.pt')).parameters())[0])
