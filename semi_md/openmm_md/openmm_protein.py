import ase.units as ase_units
import mdtraj as md
import openmm.app as app
import openmm.unit as unit

from semi_md.utilities.md_plotting_helpers import parse_log_file, make_plots
from semi_md.utilities.read_protein_simulation import read_protein_simulation

# https://github.com/ingcoder/OpenMM-MDSimulation/blob/main/OpenMM_Ligand_Protein_Simulation.ipynb


if __name__ == '__main__':
    log_file_path = 'openmm_protein.log'
    n_steps = 400
    time_step = 0.1 * unit.femtosecond
    ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    prod_simulation: app.Simulation
    prod_simulation, _, _ = read_protein_simulation(time_step)

    # Add a reporter to record the structure every 10 steps
    prod_simulation.reporters.append(app.StateDataReporter(log_file_path,
                                                           1,  # number of steps between each save
                                                           step=True,  # writes step number to each line
                                                           potentialEnergy=True,
                                                           # writes potential energy of the system (KJ/mole)
                                                           kineticEnergy=True,
                                                           totalEnergy=True,
                                                           temperature=True
                                                           ))
    dcd_reporter = app.DCDReporter("trajectory.dcd", 10, enforcePeriodicBox=True)
    prod_simulation.reporters.append(dcd_reporter)

    # Run the simulation
    prod_simulation.step(n_steps)

    traj = md.load_dcd('trajectory.dcd', top='../preparation/minimized_structure.pdb')
    traj.image_molecules(inplace=True)  # This re-wraps or images the molecules
    traj.save_dcd('trajectory_image.dcd')  # Save the processed trajectory
    traj.save_pdb('trajectory_image.pdb')

    # Parse the log file
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(log_file_path)
    make_plots(steps, potenergies, kinenergies, totenergies, temperatures, "OpenMM: ")
