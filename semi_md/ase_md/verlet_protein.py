import os
import subprocess

import ase.md
import ase.units as ase_units
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from nanover.ase.omm_calculator import OpenMMCalculator
from nanover.omni.ase_omm import ASEOpenMMSimulation

from semi_md.ase_md.unit_helpers import eV2kJ_mol, ase_unit_system
from semi_md.utilities.md_plotting_helpers import make_plots
from semi_md.utilities.read_protein_simulation import read_protein_simulation

root_path = os.getcwd()
if __name__ == "__main__":
    n_steps = 4000
    time_step = 0.01 * unit.femtosecond
    ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    simulation: app.Simulation
    simulation, openff_forcefield, openff_topology = read_protein_simulation(time_step)

    # Set up the simulation and define the time step
    ase_omm_sim = ASEOpenMMSimulation.from_simulation(simulation)
    ase_omm_sim.time_step = ase_time_step

    openmm_calculator = OpenMMCalculator(simulation)
    atoms = openmm_calculator.generate_atoms()
    atoms.calc = openmm_calculator
    atoms.set_velocities(
        simulation.context.getState(getVelocities=True).getVelocities().value_in_unit_system(ase_unit_system))

    vv = ase.md.VelocityVerlet(
        atoms=atoms,
        timestep=ase_time_step,
        trajectory="protein.traj")

    epots, ekins, etots, temps = [], [], [], []


    def printenergy(a=atoms):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        state: mm.State = a._calc.context.getState(getEnergy=True, getForces=True, getVelocities=True)
        epot = state.getPotentialEnergy().value_in_unit_system(
            unit.md_unit_system)  # a.get_potential_energy() * eV2kJ_mol
        ekin = state.getKineticEnergy().value_in_unit_system(unit.md_unit_system)  # a.get_kinetic_energy() * eV2kJ_mol
        etot = epot + ekin  # a.get_total_energy() * eV2kJ_mol
        temp = (2 * state.getKineticEnergy() / (
                len(a) * 3 * unit.BOLTZMANN_CONSTANT_kB) / unit.AVOGADRO_CONSTANT_NA).value_in_unit_system(
            unit.md_unit_system)  # a.get_temperature()
        epots.append(epot)
        ekins.append(ekin)
        etots.append(etot)
        temps.append(temp)
        if len(epots) % 10 == 0:
            print(f'Epot = {epot:.3f}kJ/mol  Ekin = {ekin:.3f}kJ/mol (T={temp:3.0f}K) '
                  f'Etot = {etot:.3f}kJ/mol')


    # Now run the dynamics
    vv.attach(printenergy, interval=1)
    printenergy()
    vv.run(n_steps)

    steps = range(len(temps))
    plot_store = {
        "epots": epots,
        "ekins": ekins,
        "etots": etots,
        "temps": temps,
        "filename": os.path.basename(__file__),
        "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip(),
    }
    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"VV_protein"), **plot_store)
    make_plots(steps, epots, ekins, etots, temps, "ASE VV: ")
