ELECTRON_MASS = 0.510998910E-03 # GeV/c^2
AVOGADRO_NUMBER = 6.02214076E+23
# optimizing Avogadro Number special for bremsstrahlung
# see bremsstrahlung.py
AVOGADRO_NUMBER_brems = 6.02214076
MUON_MASS = 0.10565839 # GeV/c^2


class AtomicElement:
    def __init__(self, A, I, Z):
        self.A = A  # Atomic mass
        self.I = I  # Mean Excitation
        self.Z = Z  # Atomic Number


STANDARD_ROCK = AtomicElement(22.0, 0.1364E-6, 11)
