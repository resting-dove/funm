import os
import subprocess

import ase.units as ase_units
import mdtraj as md
import numpy as np
import openmm.app as app
import openmm.unit as unit
import openmm as mm
from semi_md.utilities.md_plotting_helpers import parse_log_file, make_plots
from semi_md.utilities.read_protein_simulation import read_protein_simulation

root_path = os.getcwd()
if __name__ == '__main__':
    n_steps = 4000
    time_step = 0.1 * unit.femtosecond
    log_file_path = f'artifacts/openmm_protein_{time_step}.log'
    log_interval = 10
    prod_simulation: app.Simulation
    prod_integrator = mm.VerletIntegrator(time_step)
    prod_simulation, _, _ = read_protein_simulation(time_step, prod_integrator)

    # Add a reporter to record the structure every 10 steps
    prod_simulation.reporters.append(app.StateDataReporter(log_file_path,
                                                           log_interval,  # number of steps between each save
                                                           step=True,  # writes step number to each line
                                                           potentialEnergy=True,
                                                           # writes potential energy of the system (KJ/mole)
                                                           kineticEnergy=True,
                                                           totalEnergy=True,
                                                           temperature=True
                                                           ))
    dcd_reporter = app.DCDReporter("artifacts/trajectory.dcd", log_interval, enforcePeriodicBox=True)
    prod_simulation.reporters.append(dcd_reporter)

    # Run the simulation
    prod_simulation.step(n_steps)

    traj = md.load_dcd('artifacts/trajectory.dcd', top='../preparation/minimized_structure.pdb')
    traj.image_molecules(inplace=True)  # This re-wraps or images the molecules
    traj.save_dcd('artifacts/trajectory_image.dcd')  # Save the processed trajectory
    traj.save_pdb('artifacts/trajectory_image.pdb')

    # Parse the log file
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(log_file_path)
    steps = (np.array(steps) * time_step).value_in_unit(unit.femtosecond)
    plot_store = {
        "timestep": str(time_step),
        "filename": os.path.basename(__file__),
        "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip(),
    }
    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"_openmm_protein_{time_step}"), **plot_store)
    make_plots(steps, potenergies, kinenergies, totenergies, temperatures, f"figures/openmm_protein")
