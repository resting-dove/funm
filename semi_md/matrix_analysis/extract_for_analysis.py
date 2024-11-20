import ase.io
import ase.units as ase_units
import numpy as np
import openmm.unit as unit
import scipy.sparse
from nanover.omni.ase_omm import ASEOpenMMSimulation

from semi_md.ase_md.OmmCalculator import OmmCalculator
from semi_md.ase_md.unit_helpers import velocity_conversion_factor
from semi_md.rotations import get_Rx
from semi_md.utilities.read_protein_simulation import read_protein_simulation
from semi_md.ase_md.SemiAnalyticMd import SemiAnalyticMd


def semi_analytic_step(self: SemiAnalyticMd, r: np.ndarray, v: np.ndarray):
    xi = self.M_sqrt @ r.flatten()
    vi = self.M_sqrt @ v.flatten()
    RxLarge, Rx_invLarge = get_Rx(r, self.original_positions, self.openff_topology)
    # Omega^2 = M^(1/2) A M^(-1/2) = M^(-1/2) Q K Q^T M^(-1/2) and Omega^2 as well as K are spd
    omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

    g_x = self.md_nonlinearity(xi)
    return omega2, xi, vi, g_x, RxLarge


if __name__ == "__main__":
    n_steps = 400
    time_step = 0.1 * unit.femtosecond
    ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    trajectory = ase.io.read('../ase_md/mytraj.traj', index=':')
    simulation, openff_forcefield, openff_topology = read_protein_simulation(1 * unit.femtosecond)
    ase_omm_sim = ASEOpenMMSimulation.from_simulation(simulation)
    ase_omm_sim.time_step = ase_time_step

    openmm_calculator = OmmCalculator(simulation)
    atoms = openmm_calculator.generate_atoms()
    atoms.calc = openmm_calculator

    atoms.set_velocities(simulation.context.getState(getVelocities=True).getVelocities() * velocity_conversion_factor)

    print()
    vv = SemiAnalyticMd(
        atoms=atoms,
        timestep=ase_time_step,
        trajectory="simpleAnalytic.traj"
    )
    vv.setup_gautschi_integrator(forcefield=openff_forcefield, openff_topology=openff_topology)

    scipy.sparse.save_npz("proteinK.npz", vv.K)


    def extract_and_save(step: int):
        r, v = trajectory[step].get_positions(), trajectory[step].get_velocities()
        omega2, xi, vi, g_xi, RxLarge = semi_analytic_step(vv, r, v)
        scipy.sparse.save_npz(f"Omega2_{step * time_step}{time_step.unit._name}.npz", omega2)
        np.savez(f"vectors_{step * time_step}{time_step.unit._name}.npz", xi=xi, vi=vi, g_xi=g_xi)
        scipy.sparse.save_npz(f"RxLarge_{step * time_step}{time_step.unit._name}.npz", RxLarge)


    step = 0
    extract_and_save(step)

    step = 400
    extract_and_save(step)
