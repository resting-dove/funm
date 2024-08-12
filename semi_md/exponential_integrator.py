import scipy.linalg

from semi_md.computeK import compute_K
from semi_md.matrix_functions import *
from semi_md.molecular_system import MolecularSystem
import numpy as np
from functools import partial

from semi_md.rotations import get_Rx
from src.matfuncb.matfuncb import matfuncb


class BaseSemiExponentialIntegrator():
    def __init__(self, md: MolecularSystem, original_positions: np.ndarray, time_step: float):
        self.md = md
        self.original_positions = original_positions
        self.time_step = time_step
        self.setup()

    def setup(self):
        self.M_sqrt = np.kron(np.diag(np.sqrt(self.md.get_masses().flatten())),
                              np.eye(3))
        self.M_sqrt_inv = np.kron(
            np.diag(1 / np.sqrt(self.md.get_masses().flatten())), np.eye(3))

        self.K = compute_K(self.original_positions,
                           self.md.forcefield.get_parameter_handler("Bonds").find_matches(self.md.openff_topology),
                           self.md.openff_topology,
                           self.md.units)

    def g_tilde(self, r: np.array, Q: np.array):
        """
        gtilde(R) = G(R) - Q K R_0 = -F(R) - Q K R_0
        """
        p = self.md.get_positions()
        self.md.set_positions(r.reshape((-1, 3)))
        f = self.md.harmonic_angle_force()
        ret = -f.reshape(-1) + Q @ self.K @ self.original_positions.reshape(-1)
        self.md.set_positions(p)
        return ret

    def get_lambda(self, xi, RxLarge):
        """
        Lambda(xi) = M^(-1/2) gtilde(M^(-1/2) xi)
        """
        Lambda = self.M_sqrt_inv @ self.g_tilde(self.M_sqrt_inv @ xi, RxLarge)
        return Lambda

    def advance_step(self):
        raise NotImplementedError


class ExplicitGautschi():
    def __init__(self, md: MolecularSystem, original_positions: np.ndarray, time_step: float):
        self.md = md
        self.original_positions = original_positions
        self.time_step = time_step
        self.setup()

    def setup(self):
        self.M_sqrt = np.kron(np.diag(np.sqrt(self.md.get_masses().flatten())),
                              np.eye(3))
        self.M_sqrt_inv = np.kron(
            np.diag(1 / np.sqrt(self.md.get_masses().flatten())), np.eye(3))

        self.K = compute_K(self.original_positions,
                           self.md.forcefield.get_parameter_handler("Bonds").find_matches(self.md.openff_topology),
                           self.md.openff_topology,
                           self.md.units)
        self.xsinm = partial(xsinm, t=self.time_step)
        self.sinc_sqrtm_variation = partial(sinc_sqrtm_variation, t=self.time_step)

    def advance_step(self):
        k = self.K.shape[0]
        p = self.md.get_positions()
        v = self.md.get_velocities()
        vi = self.M_sqrt @ v.reshape(-1)
        RxLarge, Rx_invLarge = get_Rx(p, self.original_positions, self.md.openff_topology)
        xi = self.M_sqrt @ p.reshape(-1)
        Omega2 = self.M_sqrt_inv @ RxLarge @ -self.K @ Rx_invLarge @ self.M_sqrt_inv
        t2Omega2 = self.time_step ** 2 * Omega2

        term1 = matfuncb(t2Omega2, xi, cos_sqrtm, k, symmetric=True)[0]
        if not np.all(v == 0 * v):
            term2 = matfuncb(Omega2, vi.reshape(-1), self.sinc_sqrtm_variation, k, symmetric=True)[0]
        else:
            term2 = 0 * term1

        g = self.M_sqrt_inv @ (self.md.harmonic_angle_force().reshape(-1) -
                               RxLarge @ self.K @ self.original_positions.reshape(-1))

        term3 = 1 / 2 * self.time_step ** 2 * matfuncb(t2Omega2 / 4, g, sinc2_sqrtm, k, symmetric=True)[0]
        xi_next = term1 + term2 + term3
        p = (self.M_sqrt_inv @ xi_next).reshape((-1, 3))

        term1 = matfuncb(Omega2, xi, self.xsinm, k, symmetric=True)[0]
        if not np.all(v == 0 * v):
            term2 = matfuncb(t2Omega2, vi.reshape(-1), cos_sqrtm, k, symmetric=True)[0]
        else:
            term2 = 0 * term1
        term3 = self.time_step * matfuncb(t2Omega2, g, sinc_sqrtm, k, symmetric=True)[0]
        vi = -term1 + term2 + term3
        v = (self.M_sqrt_inv @ vi).reshape((-1, 3))

        self.md.set_positions(p)
        self.md.set_velocities(v)
        return p


class OneStepGautschi(BaseSemiExponentialIntegrator):
    def __init__(self, md: MolecularSystem, original_positions: np.ndarray, time_step: float):
        super().__init__(md, original_positions, time_step)

    def check_energy_jump(self, p, v):
        ke, pe = self.md.kinetic_energy(), self.md.potential_energy()
        self.md.set_positions(p)
        self.md.set_velocities(v)
        nke, npe = self.md.kinetic_energy(), self.md.potential_energy()
        if np.abs((nke - ke) / ke) > 0.5:
            print(f"Kinetic Energy jumped by a relative factor of {(nke - ke) / ke * 100:.1f}%.")
        if np.abs((npe - pe) / pe) > 0.5:
            print(f"Potential Energy jumped by a relative factor of {(npe - pe) / pe * 100:.1f}%.")

    def x1_step(self, xi, vi, Omega2, RxLarge, k):
        x1 = matfuncb(self.time_step ** 2 * Omega2, xi, cos_sqrtm, k=k, symmetric=True)[0]
        if np.any(vi != 0):
            x1 += self.time_step * matfuncb(self.time_step ** 2 * Omega2, vi, sinc_sqrtm, k=k, symmetric=True)[0]
        Lambda = self.get_lambda(matfuncb(self.time_step ** 2 * Omega2, xi, sinc_sqrtm, k=k, symmetric=True)[0],
                                 RxLarge)
        x1 = x1 + 0.5 * self.time_step ** 2 * \
             matfuncb(self.time_step ** 2 * Omega2, Lambda, sinc2_sqrtm, k=k, symmetric=True)[0]
        return x1, Lambda

    def x2_step(self, xi, vi, x1, Lambda, Omega2, RxLarge, k):
        x2 = -matfuncb(Omega2, xi, partial(xsinm, t=self.time_step), k=k, symmetric=True)[0]
        if np.any(vi != 0):
            x2 += matfuncb(self.time_step ** 2 * Omega2, vi, cos_sqrtm, k=k, symmetric=True)[0]
        Lambda2 = self.get_lambda(matfuncb(self.time_step ** 2 * Omega2, x1, sinc_sqrtm, k=k, symmetric=True)[0],
                                  RxLarge)
        term = matfuncb(self.time_step ** 2 * Omega2, Lambda, cosm_sincm, k=k, symmetric=True)[0] + \
               matfuncb(self.time_step ** 2 * Omega2, Lambda2, sinc_sqrtm, k=k, symmetric=True)[0]
        x2 = x2 + 0.5 * self.time_step * term
        return x2

    def advance_step(self):
        k = self.K.shape[0]
        p = self.md.get_positions()
        v = self.md.get_velocities()
        vi = self.M_sqrt @ v.reshape(-1)
        RxLarge, Rx_invLarge = get_Rx(p, self.original_positions, self.md.openff_topology)
        xi = self.M_sqrt @ p.reshape(-1)
        Omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

        x1, Lambda = self.x1_step(xi, vi, Omega2, RxLarge, k)
        x2 = self.x2_step(xi, vi, x1, Lambda, Omega2, RxLarge, k)
        p_next = (self.M_sqrt_inv @ x1).reshape((-1, 3))
        v_next = (self.M_sqrt_inv @ x2).reshape((-1, 3))
        self.check_energy_jump(p_next, v_next)
        self.prev_v = v  # For debug purposes

        return p_next


class ScipyExponential(BaseSemiExponentialIntegrator):
    def __init__(self, md: MolecularSystem, original_positions: np.ndarray, time_step: float):
        super().__init__(md, original_positions, time_step)

    def advance_step(self):
        k = self.K.shape[0]
        p = self.md.get_positions()
        v = self.md.get_velocities()
        vi = self.M_sqrt @ v.reshape(-1)
        RxLarge, Rx_invLarge = get_Rx(p, self.original_positions, self.md.openff_topology)
        xi = self.M_sqrt @ p.reshape(-1)
        Omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

        X = np.concatenate((xi, vi))
        n = len(xi)

        def deriv(t, y):
            xi, vi = y[:n], y[n:]
            return np.concatenate((vi, -Omega2 @ xi + self.get_lambda(xi, RxLarge)))

        t_end = self.time_step
        result = scipy.integrate.solve_ivp(deriv, [0, t_end], X)
        p_next = (self.M_sqrt_inv @ result.y[:n, -1]).reshape((-1, 3))
        v_next = (self.M_sqrt_inv @ result.y[n:, -1]).reshape((-1, 3))
        self.md.set_positions(p_next)
        self.md.set_velocities(v_next)
        return p_next
