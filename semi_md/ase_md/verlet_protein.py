import ase.md
import ase.units as ase_units
import openmm.app as app
import openmm.unit as unit
from nanover.ase.omm_calculator import OpenMMCalculator
from nanover.omni.ase_omm import ASEOpenMMSimulation

from semi_md.ase_md.unit_helpers import eV2kJ_mol
from semi_md.utilities.md_plotting_helpers import make_plots
from semi_md.utilities.read_protein_simulation import read_protein_simulation

if __name__ == "__main__":
    n_steps = 400
    time_step = 0.1 * unit.femtosecond
    ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    simulation: app.Simulation
    simulation, openff_forcefield, openff_topology = read_protein_simulation(time_step)

    # Set up the simulation and define the time step
    ase_omm_sim = ASEOpenMMSimulation.from_simulation(simulation)
    ase_omm_sim.time_step = ase_time_step

    openmm_calculator = OpenMMCalculator(simulation)
    atoms = openmm_calculator.generate_atoms()
    atoms.calc = openmm_calculator
    conversion_factor = (1 * (unit.nano * unit.meter) / (unit.pico * unit.second)).value_in_unit(
        unit.angstrom / (unit.femto * unit.second)) * ase_units.Angstrom / ase_units.fs
    atoms.set_velocities(simulation.context.getState(getVelocities=True).getVelocities() * conversion_factor)

    vv = ase.md.VelocityVerlet(
        atoms=atoms,
        timestep=1 * ase_units.fs,
        trajectory="mytraj.traj")

    epots, ekins, etots, temps = [], [], [], []



    def printenergy(a=atoms):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() * eV2kJ_mol
        ekin = a.get_kinetic_energy() * eV2kJ_mol
        etot = a.get_total_energy() * eV2kJ_mol
        temp = a.get_temperature()
        epots.append(epot)
        ekins.append(ekin)
        etots.append(etot)
        temps.append(temp)
        print(f'Epot = {epot:.3f}kJ/mol  Ekin = {ekin:.3f}kJ/mol (T={temp:3.0f}K) '
              f'Etot = {etot:.3f}kJ/mol')


    # Now run the dynamics
    vv.attach(printenergy, interval=1)
    printenergy()
    vv.run(n_steps)

    steps = range(len(temps))
    make_plots(steps, epots, ekins, etots, temps, "ASE VV: ")
