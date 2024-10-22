import numpy as np

from molecular_system import MolecularSystem


if __name__ == "__main__":
    positions = np.array([[0, 0, 0.0],
                          [1.0, 0.0, 0.0],])
    bonds = [(0,1)]
    md = MolecularSystem(positions, bonds)
    md.harmonic_bond_force()
    1+1