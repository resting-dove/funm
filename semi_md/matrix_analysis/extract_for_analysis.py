import mdtraj
import numpy as np
import openmm.unit as unit
import scipy.sparse
from openmm.app import PDBFile

from semi_md.openmm_md.SAIntegrator import SemiAnalyticIntegrator
from semi_md.rotations import get_Rx
from semi_md.utilities.read_protein_simulation import read_protein_simulation


def semi_analytic_step(self: SemiAnalyticIntegrator, r: np.ndarray, v: np.ndarray):
    xi = self.M_sqrt @ r.flatten()
    vi = self.M_sqrt @ v.flatten()
    RxLarge, Rx_invLarge = get_Rx(r, self.original_positions, self.openff_topology)
    # Omega^2 = M^(1/2) A M^(-1/2) = M^(-1/2) Q K Q^T M^(-1/2) and Omega^2 as well as K are spd
    omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

    g_x = self.md_nonlinearity(xi)
    return omega2, xi, vi, g_x, RxLarge


if __name__ == "__main__":
    n_steps = 4000
    time_step = 0.01 * unit.femtosecond
    omm_top2 = PDBFile("../preparation/minimized_structure.pdb").getTopology()
    trajectory: mdtraj.Trajectory = mdtraj.load(
        "../openmm_md/artifacts/trajectory.dcd", top=mdtraj.Topology.from_openmm(omm_top2)
    )
    prod_integrator = SemiAnalyticIntegrator(time_step)
    prod_simulation, openff_forcefield, openff_topology = read_protein_simulation(time_step, prod_integrator)
    prod_simulation.integrator.setup(prod_simulation, openff_forcefield, openff_topology, "TwoStepF")

    scipy.sparse.save_npz("OpenMM_K.npz", prod_simulation.integrator.K)

    state = prod_simulation.context.getState(getPositions=True, getVelocities=True)
    r = state.getPositions(asNumpy=True)
    v = state.getVelocities(asNumpy=True)


    def extract_and_save(step: int):
        if step != 0:
            raise RuntimeError("Later steps not implemented anymore with OpenMM.")
        omega2, xi, vi, g_xi, RxLarge = semi_analytic_step(prod_simulation.integrator, r, v)
        scipy.sparse.save_npz(f"OpenMM_Omega2_{(step * time_step).real}{time_step.unit._name}.npz", omega2)
        np.savez(f"OpenMM_vectors_{(step * time_step).real}{time_step.unit._name}.npz", xi=xi, vi=vi, g_xi=g_xi)
        scipy.sparse.save_npz(f"OpenMM_RxLarge_{(step * time_step).real}{time_step.unit._name}.npz", RxLarge)


    step = 0
    extract_and_save(step)
