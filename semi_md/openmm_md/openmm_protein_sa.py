import os
import subprocess

import mdtraj as md
import numpy as np
import openmm.app as app
import openmm.unit as unit

from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator, \
    LanczosDiagonalizationEvaluator
from semi_md.utilities.md_plotting_helpers import parse_log_file, make_plots
from semi_md.utilities.read_protein_simulation import read_protein_simulation
from semi_md.openmm_md.SAIntegrator import SemiAnalyticIntegrator

# https://github.com/ingcoder/OpenMM-MDSimulation/blob/main/OpenMM_Ligand_Protein_Simulation.ipynb

root_path = os.getcwd()
if __name__ == '__main__':
    n_steps = 200
    time_step = 1 * unit.femtosecond
    name = f"OSGS99_{str(time_step).replace(" ", "")}_LaDi_80"
    log_file_path = f'artifacts/openmm_protein_{name}.log'
    log_interval = 1
    prod_simulation: app.Simulation
    prod_integrator = SemiAnalyticIntegrator(time_step)
    prod_simulation, openff_forcefield, topology = read_protein_simulation(time_step, prod_integrator)
    prod_simulation.integrator.setup(prod_simulation, openff_forcefield, topology, "OneStepGS99")
    evaluator = LanczosDiagonalizationEvaluator(krylov_size=80)  # LanczosWkmEvaluator(krylov_size=80)
    prod_simulation.integrator.evaluator = evaluator

    prod_simulation.reporters.append(app.StateDataReporter(log_file_path,
                                                           log_interval,  # number of steps between each save
                                                           step=True,  # writes step number to each line
                                                           potentialEnergy=True,
                                                           # writes potential energy of the system (KJ/mole)
                                                           kineticEnergy=True,
                                                           totalEnergy=True,
                                                           temperature=True
                                                           ))
    dcd_reporter = app.DCDReporter(f"artifacts/{name}_trajectory.dcd", log_interval, enforcePeriodicBox=False)
    prod_simulation.reporters.append(dcd_reporter)

    # Run the simulation
    prod_simulation.step(n_steps)

    traj = md.load_dcd(f'artifacts/{name}_trajectory.dcd', top='../preparation/minimized_structure.pdb')
    traj.image_molecules(inplace=True)  # This re-wraps or images the molecules
    traj.save_dcd(f'artifacts/{name}_trajectory_image.dcd')  # Save the processed trajectory
    traj.save_pdb(f'artifacts/{name}_trajectory_image.pdb')

    # Parse the log file
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(log_file_path)
    steps = (np.array(steps) * time_step).value_in_unit(unit.femtosecond)
    try:
        krylov_size = evaluator.k
        max_restarts = evaluator.max_restarts
    except:
        krylov_size = np.nan
        max_restarts = np.nan
    plot_store = {
        "timestep": str(time_step),
        "integrator": str(prod_integrator),
        "evaluator": str(evaluator),
        "krylov_size": krylov_size,
        "max_restarts": max_restarts,
        "filename": os.path.basename(__file__),
        "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip(),
    }
    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"_openmm_protein_{name}"), **plot_store)
    make_plots(steps, potenergies, kinenergies, totenergies, temperatures, f"openmm_protein_{name}")
