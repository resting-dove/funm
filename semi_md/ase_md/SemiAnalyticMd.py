import ase.md.md
import numpy as np
import openff.toolkit
import scipy

from gautschiIntegrators.gautschiIntegrators.one_step import OneStepF
from semi_md.ase_md.MatfuncEvaluator import MatfuncEvaluator
from semi_md.nanover.openmm_basic_sim import copy_supply_properties, save_to_traj
from semi_md.rotations import get_Rx


class SemiAnalyticMd(ase.md.md.MolecularDynamics):
    def step(self, forces=None):
        atoms = self.atoms

        r = atoms._calc._get_positions()
        v = atoms._calc._get_velocities()

        xi = self.M_sqrt @ r.flatten()
        vi = self.M_sqrt @ v.flatten()
        RxLarge, Rx_invLarge = get_Rx(r, self.original_positions, self.openff_topology)
        # Omega^2 = M^(1/2) A M^(-1/2) = M^(-1/2) Q K Q^T M^(-1/2) and Omega^2 as well as K are spd
        omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

        forces = (self.M_sqrt_inv @
                  (-1 * omega2 @ xi.flatten() + self.md_nonlinearity(xi))).reshape((-1, 3))
        atoms_copy = copy_supply_properties(atoms, forces)
        save_to_traj(atoms_copy, "simpleAnalytic_K_forces.traj")

        xi_n, vi_n = self.integrator.step(omega2, xi, vi)
        x_n = self.M_sqrt_inv @ xi_n
        v_n = self.M_sqrt_inv @ vi_n

        self.atoms._calc._set_positions(x_n.reshape((-1, 3)))
        self.atoms._calc._set_velocities(v_n.reshape((-1, 3)))

        self.atoms.set_positions(x_n.reshape((-1, 3)))
        self.atoms.set_velocities(v_n.reshape((-1, 3)))
        return None

    def setup_gautschi_integrator(self, forcefield, openff_topology):
        self.openff_topology: openff.toolkit.Topology = openff_topology
        self.original_positions = self.atoms.get_positions()
        self.K = self.atoms._calc.calculate_k(self.original_positions, forcefield, openff_topology)
        # self.K = scipy.sparse.csr_array(
        #     compute_K_v2(self.atoms._calc.context.getState(getPositions=True).getPositions(True).in_unit_system(ase_unit_system),
        #                          forcefield.get_parameter_handler("Bonds").find_matches(openff_topology, True),
        #                          openff_topology).value_in_unit_system(ase_unit_system))
        self.M_sqrt = scipy.sparse.kron(scipy.sparse.diags_array(np.sqrt(self.atoms.get_masses().flatten())),
                                        np.eye(3), format="coo").tocsr()
        self.M_sqrt_inv = scipy.sparse.kron(scipy.sparse.diags_array(1 / np.sqrt(self.atoms.get_masses().flatten())),
                                            np.eye(3),
                                            format="coo").tocsr()
        mfE = MatfuncEvaluator()
        self.integrator = OneStepF(self.dt, cosm=mfE.sym_cosm_sqrt, sincm=mfE.sym_sincm_sqrt, msinm=mfE.sym_msinm_sqrt,
                                   g=self.md_nonlinearity)

    def md_nonlinearity(self, xi: np.array):
        """The nonlinearity g of the differential equation x'' = -A @ x + g(x).

        The semi-analytical MD is
            xi'' = -Omega^2 @ xi - lambda(xi).
        Let K be the Hessian of the Bond lenght potential, i.e. F_B(x) = -K(x-x_0)  # (sign is flipped compared to SemiAna paper),
        then
            xi''    = - M^(-1/2) Q K Q^T M^(-1/2) @ xi  + M^(-1/2) Q K x_0 + M^(-1/2) @ F_rest(M^(-1/2) @ xi)
                    = - Omega^2 @ xi + g(xi).
        In particular the nonlinearity is
            -lambda(xi) = g(xi) = M^(-1/2) @ ( F_rest(M^(-1/2) @ xi) + Q K x_0).
        """
        r = self.atoms._calc._get_positions()
        x = (self.M_sqrt_inv @ xi).reshape((-1, 3))
        RxLarge, Rx_invLarge = get_Rx(x, self.original_positions, self.openff_topology)
        self.atoms._calc._set_positions(x)
        f = self.atoms._calc._calculate_openmm(groups=0b100)[1]
        g = self.M_sqrt_inv @ (f.reshape(-1) + RxLarge @ self.K @ self.original_positions.reshape(-1))
        self.atoms.set_positions(r)
        self.atoms._calc._set_positions(r)
        return g
