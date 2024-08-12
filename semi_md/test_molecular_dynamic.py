import numpy as np
import openmm.unit.unit_definitions
from openff.interchange import Interchange
from openff.toolkit import Molecule, ForceField, Topology
from openmm import VerletIntegrator
from openmm.app import Simulation
from openmm.unit import md_unit_system, Quantity
import copy

from animation import animate_md
from molecular_system import MolecularSystem
from semi_md.exponential_integrator import OneStepGautschi, ScipyExponential
from utility import get_rest_positions
from verlet_integrator import MyVerletIntegrator

if __name__ == "__main__":
    # SETUP
    smiles_explicit_h = Molecule.from_smiles(
        # "[H][C]([H])([H])[C@@]([H])([C](=[O])[O-])[N+]([H])([H])[H]",
        # "N1N=NN=N1",
        "C",
        #"[H]O[H]",
        # "O=C=O",
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
    r = np.empty((200, *original_positions_numpy.shape))
    md = MolecularSystem(simulation, openff_forcefield, openff_topology, md_unit_system)
    md.set_velocities(md.get_velocities() * 0)
    assert np.all(md.get_positions() == original_positions.value_in_unit_system(md.units))
    verlet = MyVerletIntegrator(time_step.value_in_unit_system(md.units))
    #verlet.recenter_positions(md)
    r[0] = md.get_positions()
    print(f"0: kinetic energy: {md.kinetic_energy()}, potential energy: {md.potential_energy()}")
    for i in range(1, len(r)):
        p = verlet.advance_step(md)
        r[i] = p
        if i % 1 == 0:
            print(
                f"{i}: kin. energy: {md.kinetic_energy()}, "
                f"pot. energy: {md.potential_energy():.2f} "
                f"({md.harmonic_bond_engergy():.1f}, {md.harmonic_angle_energy():.1f}),"
                f"\t total energy: {md.total_energy()}")

    ani = animate_md(r, topology=openff_topology, step=1, return_as="")
    ani.save(filename="complicated_example.html", writer="html")

    # Comparison with scipy exponential integrator
    rscex = np.empty((len(r), openff_topology.n_atoms, 3))
    md = MolecularSystem(simulation, openff_forcefield, openff_topology, md_unit_system)
    md.set_positions(original_positions_numpy.value_in_unit_system(md.units))
    md.set_velocities(md.get_velocities() * 0)
    scex = ScipyExponential(md,
                            original_positions_numpy.value_in_unit_system(md.units),
                            time_step.value_in_unit_system(md.units),
                            recenter=False)
    assert np.all(scex.md.get_positions() == original_positions.value_in_unit_system(scex.md.units))
    #scex.md.set_positions(scex.recenter_positions(scex.md.get_positions()))
    rscex[0] = md.get_positions()
    print(f"SciExp: 0: kinetic energy: {md.kinetic_energy()}, potential energy: {md.potential_energy()}")

    for i in range(1, len(r)):
        p = scex.advance_step()
        rscex[i] = p
        if i % 1 == 0:
            print(
                f"ScipExp: {i}: kin. energy: {scex.md.kinetic_energy()}, "
                f"pot. energy: {scex.md.potential_energy():.2f} "
                f"({scex.md.harmonic_bond_engergy():.1f}, {scex.md.harmonic_angle_energy():.1f}),"
                f"\t total energy: {scex.md.total_energy()}")
    ani = animate_md(rscex, topology=openff_topology, step=1, return_as="")
    ani.save(filename="complicated_scipexp.html", writer="html")

    # Comparison with one step exponential integrator
    rgau = np.empty((len(r), openff_topology.n_atoms, 3))
    md = MolecularSystem(simulation, openff_forcefield, openff_topology, md_unit_system)
    md.set_positions(original_positions_numpy.value_in_unit_system(md.units))
    md.set_velocities(md.get_velocities() * 0)
    gautschi = OneStepGautschi(md,
                               original_positions_numpy.value_in_unit_system(md.units),
                               time_step.value_in_unit_system(md.units))
    assert np.all(gautschi.md.get_positions() == original_positions.value_in_unit_system(gautschi.md.units))
    #gautschi.recenter_positions()
    rgau[0] = md.get_positions()
    print(f"Gautschi: 0: kinetic energy: {md.kinetic_energy()}, potential energy: {md.potential_energy()}")

    for i in range(1, len(r)):
        p = gautschi.advance_step()
        rgau[i] = p
        if i % 1 == 0:
            print(
                f"Gautschi: {i}: kin. energy: {gautschi.md.kinetic_energy()},"
                f" pot. energy: {gautschi.md.potential_energy():.2f} "
                f"({gautschi.md.harmonic_bond_engergy():.1f}, {gautschi.md.harmonic_angle_energy():.1f}),"
                f"\t total energy: {gautschi.md.total_energy()}")
    ani = animate_md(rgau, topology=openff_topology, step=1, return_as="")
    ani.save(filename="complicated_gautschi.html", writer="html")

    for i in range(0, len(r), 50):
        e_sci = np.linalg.norm((r[i] - rscex[i]) / (r[i] + np.finfo(np.float64).eps))
        e_g = np.linalg.norm((r[i] - rgau[i]) / (r[i] + np.finfo(np.float64).eps))
        print(f"{i * gautschi.time_step} {gautschi.md.units.express_unit(openmm.unit.second)}:\t"
              f"relative error Scipy Exponential integrator {100 * e_sci:.3f}%,\t"
              f"relative error my Exponential integrator {100 * e_g:.3f}%.")
    if i != len(r) - 1:
        e_sci = np.linalg.norm((r[-1] - rscex[-1]) / (r[-1] + np.finfo(np.float64).eps))
        e_g = np.linalg.norm((r[-1] - rgau[-1]) / (r[-1] + np.finfo(np.float64).eps))
        print(f"{i * gautschi.time_step} {gautschi.md.units.express_unit(openmm.unit.second)}:\t"
              f"relative error Scipy Exponential integrator {100 * e_sci:.3f}%,\t"
              f"relative error my Exponential integrator {100 * e_g:.3f}%.")
