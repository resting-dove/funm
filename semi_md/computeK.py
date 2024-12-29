import openmm
import scipy.sparse
from openff.toolkit import Topology
from openmm.unit import kelvin, pico, meter, kilo, joule, mole, dalton, angstrom, calorie, nano, second, femto, \
    dimensionless
from openmm.unit.quantity import Quantity
from openmm.unit.unit import UnitSystem
import numpy as np


def compute_K(positions: np.array, bonds: dict, openff_topology: Topology, units: UnitSystem):
    '''The function computes the Hessian of the harmonic potential. This 
    is also the Jacobian of the force and usually denoted $K$ in our formulas.
    In particular the potential used here uses the Euclidean distance with
    square root $r_{ij}=||r_i - r_j||_2$.
    '''
    n = positions.size
    K = scipy.sparse.lil_array((n, n), dtype=np.float64)
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
                K_small = k * distance ** (-3) * (- r0) \
                          * np.outer(difference, difference)
                add_for_diag = k * -(distance - r0) * distance ** (-1)
                K_small += np.diag(np.ones(3) * add_for_diag)
                K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = K_small
                # import pdb; pdb.set_trace()
                K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= K_small / 2
                K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= K_small / 2  # / 2 because of later + transpose
    K = K + K.transpose()
    return K


def compute_K_with_dict(positions: np.array, bonds_dict: dict, units: UnitSystem):
    '''The function computes the Hessian of the harmonic potential. This
    is also the Jacobian of the force and usually denoted $K$ in our formulas.
    In particular the potential used here uses the Euclidean distance with
    square root $r_{ij}=||r_i - r_j||_2$.
    '''
    index_helper = np.array([0] * 3 + [1] * 3 + [2] * 3).reshape((3, 3))
    n = positions.size
    row = np.empty(4 * 9 * len(bonds_dict))
    col = np.empty(4 * 9 * len(bonds_dict))
    data = np.empty(4 * 9 * len(bonds_dict))
    index = 0
    for (i, j), bond in bonds_dict.items():
        k = bond.parameter_type.k.to_openmm() \
            .value_in_unit_system(units)
        r0 = bond.parameter_type.length.to_openmm() \
            .value_in_unit_system(units)
        difference = positions[i] - positions[j]
        distance = np.linalg.norm(difference)
        K_small = k * distance ** (-3) * (- r0) \
                  * np.outer(difference, difference)
        add_for_diag = k * -(distance - r0) * distance ** (-1)
        K_small += np.diag(np.ones(3) * add_for_diag)

        # K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = K_small
        row[index: index + 9] = (index_helper + i * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + j * 3).flatten(order='F')
        data[index: index + 9] = K_small.flatten(order='C')
        index += 9
        # K[j * 3: (j + 1) * 3, i * 3: (i + 1) * 3] = K_small
        row[index: index + 9] = (index_helper + j * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + i * 3).flatten(order='F')
        data[index: index + 9] = K_small.flatten(order='C')
        index += 9
        # K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= K_small
        row[index: index + 9] = (index_helper + j * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + j * 3).flatten(order='F')
        data[index: index + 9] = -1 * K_small.flatten(order='C')
        index += 9
        # K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= K_small
        row[index: index + 9] = (index_helper + i * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + i * 3).flatten(order='F')
        data[index: index + 9] = -1 * K_small.flatten(order='C')
        index += 9
    K = scipy.sparse.coo_array((np.array(data), (np.array(row), np.array(col))), shape=(n, n)).tocsr()
    return K


def compute_K_with_force(positions: np.array, force: openmm.HarmonicBondForce, units: UnitSystem):
    '''The function computes the Hessian of the harmonic potential. This
    is also the Jacobian of the force and usually denoted $K$ in our formulas.
    In particular the potential used here uses the Euclidean distance with
    square root $r_{ij}=||r_i - r_j||_2$.
    '''
    index_helper = np.array([0] * 3 + [1] * 3 + [2] * 3).reshape((3, 3))
    n = positions.size
    num_bonds = force.getNumBonds()
    row = np.empty(4 * 9 * num_bonds)
    col = np.empty(4 * 9 * num_bonds)
    data = np.empty(4 * 9 * num_bonds)
    index = 0
    for bond_index in range(num_bonds):
        i, j, r0, k = force.getBondParameters(bond_index)
        k = k.value_in_unit_system(units)
        r0 = r0.value_in_unit_system(units)
        difference = positions[i] - positions[j]
        distance = np.linalg.norm(difference)
        K_small = k * distance ** (-3) * (- r0) \
                  * np.outer(difference, difference)
        add_for_diag = k * -(distance - r0) * distance ** (-1)
        K_small += np.diag(np.ones(3) * add_for_diag)

        # K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = K_small
        row[index: index + 9] = (index_helper + i * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + j * 3).flatten(order='F')
        data[index: index + 9] = K_small.flatten(order='C')
        index += 9
        # K[j * 3: (j + 1) * 3, i * 3: (i + 1) * 3] = K_small
        row[index: index + 9] = (index_helper + j * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + i * 3).flatten(order='F')
        data[index: index + 9] = K_small.flatten(order='C')
        index += 9
        # K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= K_small
        row[index: index + 9] = (index_helper + j * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + j * 3).flatten(order='F')
        data[index: index + 9] = -1 * K_small.flatten(order='C')
        index += 9
        # K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= K_small
        row[index: index + 9] = (index_helper + i * 3).flatten(order='C')
        col[index: index + 9] = (index_helper + i * 3).flatten(order='F')
        data[index: index + 9] = -1 * K_small.flatten(order='C')
        index += 9
    K = scipy.sparse.coo_array((np.array(data), (np.array(row), np.array(col))), shape=(n, n)).tocsr()
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
                K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] += l + r
                K[j * 3: (j + 1) * 3, i * 3: (i + 1) * 3] += l + r
                K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= l + r
                K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= l + r
    return K


def compute_K_v2_with_force(r_ini, force: openmm.HarmonicBondForce, unit_system: openmm.unit.UnitSystem):
    '''The function computes the Hessian of the harmonic potential.
    This is also the Jacobian of the spring force.
    In particular the potential here uses the squared Euclidean distance
    $r_{ij}=||r_i - r_j||_2^2$.
    Source: https://victoriacity.github.io/hessian/'''
    n = np.size(r_ini)
    num_bonds = force.getNumBonds()
    K = Quantity(np.zeros((n, n)), kilo * calorie / (mole)).in_unit_system(unit_system)
    for bond_index in range(num_bonds):
        i, j, r0, k = force.getBondParameters(bond_index)
        RiRj = r_ini[i] - r_ini[j]
        rij = Quantity(np.linalg.norm(RiRj), RiRj.unit) ** 2
        k = k.in_unit_system(unit_system)
        r0 = r0.in_unit_system(unit_system) ** 2
        l = -4 * k * Quantity(np.outer(RiRj, RiRj), RiRj.unit ** 2)
        r = -2 * k * (rij - r0) * np.eye(3)
        K[i * 3: (i + 1) * 3, j * 3: (j + 1) * 3] = +l + r
        K[j * 3: (j + 1) * 3, i * 3: (i + 1) * 3] = +l + r
        K[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] -= l + r
        K[j * 3: (j + 1) * 3, j * 3: (j + 1) * 3] -= l + r
    return K
