import numpy as np
import openmm.unit.unit_definitions
from matplotlib import pyplot as plt
from openff.toolkit import Molecule, ForceField, Topology
from openff.interchange import Interchange
from openmm.unit import md_unit_system, Quantity
from openmm import VerletIntegrator
from openmm.app import Simulation
from molecular_system import MolecularSystem
from semi_md.exponential_integrator import ExplicitGautschi, OneStepGautschi, ScipyExponential
from verlet_integrator import MyVerletIntegrator
from animation import animate_md
from utility import get_rest_positions
import scipy
import numpy as np
from rotations import get_Rx
from computeK import compute_K


def phi(A):
    A_inv = scipy.linalg.inv(A)
    return (scipy.linalg.expm(A) - np.eye(A.shape[0])) @ A_inv


if __name__ == "__main__":
    # SETUP
    smiles_explicit_h = Molecule.from_smiles(
        # "[H][C]([H])([H])[C@@]([H])([C](=[O])[O-])[N+]([H])([H])[H]",
        # "N1N=NN=N1",
        # "[H]O[H]",
        "O=C=O",
        # "N#N",
        hydrogens_are_explicit=False,
    )
    smiles_explicit_h.generate_conformers(n_conformers=1)

    openff_forcefield = ForceField("openff_unconstrained-2.1.0.offxml")
    interchange = Interchange.from_smirnoff(openff_forcefield, [smiles_explicit_h])

    positions = interchange.positions.to_openmm()
    system = interchange.to_openmm_system()
    for _ in range(system.getNumConstraints()):
        system.removeConstraint(0)
    openff_topology = Topology.from_molecules([smiles_explicit_h])
    time_step = Quantity(0.1, openmm.unit.femto * openmm.unit.second)
    integrator = VerletIntegrator(time_step)
    simulation = Simulation(openff_topology.to_openmm(), system, integrator)
    simulation.context.setPositions(positions)

    # Get rest positions
    original_positions, original_positions_numpy = get_rest_positions(openff_topology, system, positions,
                                                                      print_energies=False)
    simulation.context.setPositions(original_positions)
    print(f"pot.E: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")

    # Insert into my class
    r = np.empty((2000, *original_positions_numpy.shape))
    # md = MolecularSystem(simulation, openff_forcefield, openff_topology, md_unit_system)
    # assert np.all(md.get_positions() == original_positions.value_in_unit_system(md.units))
    # verlet = MyVerletIntegrator(time_step.value_in_unit_system(md.units))
    # r[0] = md.get_positions()
    # print(f"0: kinetic energy: {md.kinetic_energy()}, potential energy: {md.potential_energy()}")
    # for i in range(1, len(r)):
    #     p = verlet.advance_step(md)
    #     r[i] = p
    #     print(
    #         f"{i}: kin. energy: {md.kinetic_energy()}, pot. energy: {md.potential_energy():.2f} ({md.harmonic_bond_engergy():.1f}, {md.harmonic_angle_energy():.1f}),\t total energy: {md.total_energy()}")
    #     if i == 50:
    #         1 + 1
    #
    # ani = animate_md(r, topology=openff_topology, step=1, return_as="")
    # ani.save(filename="html_example.html", writer="html")

    # Comparison with scipy exponential integrator
    r = np.empty((len(r), openff_topology.n_atoms, 3))
    md = MolecularSystem(simulation, openff_forcefield, openff_topology, md_unit_system)
    md.set_positions(original_positions_numpy.value_in_unit_system(md.units))
    scex = ScipyExponential(md,
                            original_positions_numpy.value_in_unit_system(md.units),
                            time_step.value_in_unit_system(md.units))
    r[0] = md.get_positions()
    print(f"SciExp: 0: kinetic energy: {md.kinetic_energy()}, potential energy: {md.potential_energy()}")

    for i in range(len(r)):
        if i == 160 or i == 161:
            1 + 1
        p = scex.advance_step()
        r[i] = p
        if i % 1 == 0:
            print(
                f"ScipExp: {i}: kin. energy: {scex.md.kinetic_energy()}, pot. energy: {scex.md.potential_energy():.2f} ({scex.md.harmonic_bond_engergy():.1f}, {scex.md.harmonic_angle_energy():.1f}),\t total energy: {scex.md.total_energy()}")
    ani = animate_md(r, topology=openff_topology, step=1, return_as="")
    ani.save(filename="html_scipexp.html", writer="html")
