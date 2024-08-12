import numpy as np
import openff.toolkit
import scipy.linalg
import openmm
from openmm.app import Topology, Simulation, simulation
from openmm.unit.quantity import Quantity

from molecular_forces import harmonic_bond_force_, harmonic_bond_potential_, harmonic_angle_potential_, \
    harmonic_angle_force_


class MolecularSystem():
    """My own wrapper around openmm so that I have exact control over what is
    being done during molecular dynamics simulation."""

    def __init__(self, simulation: Simulation,
                 forcefield: openff.toolkit.ForceField,
                 openff_topology: openff.toolkit.Topology,
                 units: openmm.unit.unit.UnitSystem):
        self.simulation = simulation
        self.units = units
        self.forcefield = forcefield
        self.openff_topology = openff_topology

    def get_positions(self):
        return (self.simulation.context.getState(getPositions=True)
                .getPositions(True)
                .value_in_unit_system(self.units))

    def get_velocities(self):
        return (self.simulation.context.getState(getVelocities=True)
                .getVelocities(True)
                .value_in_unit_system(self.units))

    def set_positions(self, positions: np.array):
        self.simulation.context.setPositions(
            Quantity(positions,
                     self.units.express_unit(openmm.unit.meter)))

    def set_velocities(self, velocities: np.array):
        self.simulation.context.setVelocities(
            Quantity(velocities,
                     self.units.express_unit(openmm.unit.meter) /
                     self.units.express_unit(openmm.unit.second)))

    def get_masses(self):
        m = np.empty(self.simulation.system.getNumParticles())
        for i in range(self.simulation.system.getNumParticles()):
            m[i] = (self.simulation.system.getParticleMass(i)
                    .value_in_unit_system(self.units))
        return m

    def get_harmonic_bond_parameters(self, particle1: int, particle2: int):
        bonds = self.forcefield.get_parameter_handler("Bonds").find_matches(self.openff_topology)
        bond = bonds.get((particle1, particle2))
        k = bond.parameter_type.k.to_openmm().value_in_unit_system(self.units)
        length = bond.parameter_type.length.to_openmm().value_in_unit_system(self.units)
        return length, k

    def harmonic_bond_force(self):
        positions = self.get_positions()
        forces = np.zeros_like(positions)
        for bond in self.simulation.topology.bonds():
            p1, p2 = bond.atom1.index, bond.atom2.index
            length, k = self.get_harmonic_bond_parameters(p1, p2)
            force = harmonic_bond_force_(positions[p1, :],
                                         positions[p2, :],
                                         length,
                                         k)
            forces[p1, :] += force
            forces[p2, :] += -force
        return forces

    def harmonic_bond_engergy(self):
        positions = self.get_positions()
        pot = 0
        for bond in self.simulation.topology.bonds():
            p1, p2 = bond.atom1.index, bond.atom2.index
            length, k = self.get_harmonic_bond_parameters(p1, p2)
            r = scipy.linalg.norm(positions[p1, :] - positions[p2, :])
            pot += harmonic_bond_potential_(r,
                                            length,
                                            k)
        return pot

    def get_harmonic_angle_parameters(self, particle1: int, particle2: int, particle3: int):
        angles = self.forcefield.get_parameter_handler("Angles").find_matches(self.openff_topology)
        angle = angles.get((particle1, particle2, particle3))
        k = angle.parameter_type.k.to_openmm().value_in_unit_system(self.units)
        theta0 = angle.parameter_type.angle.to_openmm().value_in_unit_system(self.units)
        return theta0, k

    def harmonic_angle_energy(self):
        positions = self.get_positions()
        pot = 0
        for angle in self.forcefield.get_parameter_handler("Angles").find_matches(self.openff_topology).keys():
            p1, p2, p3 = angle
            theta0, k = self.get_harmonic_angle_parameters(p1, p2, p3)
            pot += harmonic_angle_potential_(positions[p1, :],
                                             positions[p2, :],
                                             positions[p3, :],
                                             theta0,
                                             k)
        return pot

    def harmonic_angle_force(self):
        positions = self.get_positions()
        forces = np.zeros_like(positions)
        for angle in self.forcefield.get_parameter_handler("Angles").find_matches(self.openff_topology).keys():
            p1, p2, p3 = angle
            theta0, k = self.get_harmonic_angle_parameters(p1, p2, p3)
            force = harmonic_angle_force_(positions[p1, :],
                                          positions[p2, :],
                                          positions[p3, :],
                                          theta0,
                                          k)
            forces[[p1, p2, p3], :] += force
        return forces

    def get_forces(self):
        return self.harmonic_angle_force() + self.harmonic_bond_force()

    def potential_energy(self):
        return self.harmonic_bond_engergy() + self.harmonic_angle_energy()

    def kinetic_energy(self):
        return np.sum(0.5 * self.get_masses()[:, None] * self.get_velocities() ** 2)

    def total_energy(self):
        return self.potential_energy() + self.kinetic_energy()
