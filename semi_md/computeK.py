from openff.toolkit import Topology
from openmm.unit import kelvin, pico, meter, kilo, joule, mole, dalton, angstrom, calorie, nano, second, femto, \
    dimensionless
from openmm.unit.quantity import Quantity
from openmm.unit.unit import UnitSystem
import numpy as np


def compute_K(positions: np.array, bonds: dict, openff_topology: Topology, units: UnitSystem):
    '''The function computes the Hessian of the harmonic potential. This 
    is also the Jacobian of the force and usually denoted $K$ in our formulas.
    In particular the potential used here uses the euclidean distance with 
    square root $r_{ij}=||r_i - r_j||_2$.
    '''
    n = positions.size
    K = np.zeros((n, n))
    for i in range(openff_topology.n_atoms):
        for j in range(i + 1, openff_topology.n_atoms):
            if bonds.get((i, j)):
                k = bonds.get((i, j)) \
                    .parameter_type.k.to_openmm() \
                    .value_in_unit_system(units)
                r0 = bonds.get((i, j)) \
                    .parameter_type.length.to_openmm() \
                    .value_in_unit_system(units)
                difference = positions[i] - positions[j]
                distance = np.linalg.norm(difference)
                K_small = k * distance ** (-2) \
                          * (-1 + (distance - r0) * distance ** (-1)) \
                          * np.outer(difference, difference)
                add_for_diag = k * -(distance - r0) * distance ** (-1)
                K_small += np.diag(np.ones(3) * add_for_diag)
                K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = K_small
                # import pdb; pdb.set_trace()
                K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= K_small / 2
                K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= K_small / 2  # / 2 because of later + transpose
    K = K + K.transpose()
    return K


def compute_K_v2(r_ini, bonds, topology):
    '''The function computes the Hessian of the harmonic potential.
    This is also the Jacobian of the spring force.
    In particular the potential here uses the squared Euclidean distance
    $r_{ij}=||r_i - r_j||_2^2$.
    Source: https://victoriacity.github.io/hessian/'''
    n = np.size(r_ini)
    K = Quantity(np.zeros((n, n)), kilo * calorie / (mole))
    for i in range(topology.n_atoms):
        for j in range(i + 1, topology.n_atoms):
            RiRj = r_ini[i] - r_ini[j]
            rij = Quantity(np.linalg.norm(RiRj), RiRj.unit) ** 2
            if bonds.get((i, j)):
                k = bonds.get((i, j)) \
                    .parameter_type.k.to_openmm()
                r0 = bonds.get((i, j)) \
                         .parameter_type.length \
                         .to_openmm() ** 2
                l = -4 * k * Quantity(np.outer(RiRj, RiRj), RiRj.unit ** 2)
                r = -2 * k * (rij - r0) * np.eye(3)
                K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = l + r
                K[j * 3: (j + 1) * 3, i * 3: (i + 1) * 3] = l + r
                K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= l + r
                K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= l + r
    return K
