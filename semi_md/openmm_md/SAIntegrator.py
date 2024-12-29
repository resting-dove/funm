import numpy as np
import openff
import scipy
import openmm as mm
from openmm import Integrator
import openmm.unit as unit

from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator
from gautschiIntegrators.gautschiIntegrators.one_step import OneStepF, OneStepGS99, OneStep217
from gautschiIntegrators.gautschiIntegrators.two_step import TwoStepIntegratorF
from semi_md.computeK import compute_K_with_dict
from semi_md.rotations import get_Rx

integrators = {
    "TwoStepF": TwoStepIntegratorF,
    "OneStepF": OneStepF,
    "OneStepGS99": OneStepGS99,
    "OneStep217": OneStep217
}


class SemiAnalyticIntegrator(mm.CustomIntegrator):
    """
    Semi-Analytic integrator for OpenMM.

    Before usage setup() needs to be called.
    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        super(SemiAnalyticIntegrator, self).__init__(timestep)
        self.addUpdateContextState()
        # self.addConstrainPositions()
        # self.addConstrainVelocities()
        self.simulation = None
        self.K = None

    def setup(self, simulation, forcefield, openff_topology, integrator_method="TwoStepF"):
        self.simulation = simulation
        state: mm.State = self.simulation.context.getState(getPositions=True,
                                                           getVelocities=True, enforcePeriodicBox=False)
        self.original_positions = state.getPositions(True)

        self.openff_topology: openff.toolkit.Topology = openff_topology
        self.box = self.openff_topology.box_vectors
        self.K = compute_K_with_dict(self.original_positions,
                                     forcefield.get_parameter_handler("Bonds").find_matches(openff_topology, True),
                                     unit.md_unit_system)
        m = self.get_masses()
        self.M_sqrt = scipy.sparse.kron(scipy.sparse.diags_array(np.sqrt(m)),
                                        np.eye(3), format="coo").tocsr()
        self.M_sqrt_inv = scipy.sparse.kron(scipy.sparse.diags_array(1 / np.sqrt(m)),
                                            np.eye(3),
                                            format="coo").tocsr()
        self.integrator = integrators[integrator_method](self.getStepSize().value_in_unit_system(unit.md_unit_system),
                                                         g=self.md_nonlinearity, t_end=np.inf,
                                                         x0=self.original_positions.flatten(),
                                                         v0=self.original_positions.flatten() * 0,
                                                         evaluator=LanczosWkmEvaluator(krylov_size=80)
                                                         )

    def get_masses(self):
        m = np.empty(self.simulation.system.getNumParticles())
        for i in range(self.simulation.system.getNumParticles()):
            m[i] = (self.simulation.system.getParticleMass(i)
                    .value_in_unit_system(unit.md_unit_system))
        return m

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
        state: mm.State = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        r = state.getPositions(True)
        x = (self.M_sqrt_inv @ xi).reshape((-1, 3))
        RxLarge, Rx_invLarge = get_Rx(x, self.original_positions, self.openff_topology)
        self.simulation.context.setPositions(x * r.unit)
        f = self.simulation.context.getState(getForces=True, groups=0b100, enforcePeriodicBox=False).getForces(True)
        g = self.M_sqrt_inv @ (f.flatten() + RxLarge @ self.K @ self.original_positions.flatten())
        self.simulation.context.setPositions(r)
        return g

    def _step(self):
        state: mm.State = self.simulation.context.getState(getPositions=True,
                                                           getVelocities=True, enforcePeriodicBox=False)
        r = state.getPositions(True)
        v = state.getVelocities(True)

        xi = self.M_sqrt @ r.flatten()
        vi = self.M_sqrt @ v.flatten()
        RxLarge, Rx_invLarge = get_Rx(r, self.original_positions, self.openff_topology)
        # Omega^2 = M^(1/2) A M^(-1/2) = M^(-1/2) Q K Q^T M^(-1/2) and Omega^2 as well as K are spd
        omega2 = self.M_sqrt_inv @ RxLarge @ self.K @ Rx_invLarge @ self.M_sqrt_inv

        xi_n, vi_n = self.integrator.service_step(omega2, xi, vi)
        x_n = self.M_sqrt_inv @ xi_n
        v_n = self.M_sqrt_inv @ vi_n

        self.simulation.context.setPositions(x_n.reshape((-1, 3)) * r.unit)
        self.simulation.context.setVelocities(v_n.reshape((-1, 3)) * v.unit)
        return None

    def step(self, steps):
        r"""
        Advance a simulation through time by taking a series of time steps.

        The custom step is injected before calling out to the dummy step that updates state.

        Parameters
        ----------
        steps : int
            the number of time steps to take
        """
        for i in range(steps):
            res = super().step(1)
            self._step()
        return res


class WorkReporter(object):
    """
    To use it, create a WorkReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval):
        """Create a WorkReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        """
        self._reportInterval = reportInterval
        self._out = open(file, "w")

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        dict
            A dictionary describing the required information for the next report
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # return {'steps':steps, 'periodic':None, 'include':[]}
        return [steps, False, False, False, False]  # OpenMM 8.1, the line above is 8.2

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        work = simulation.integrator.integrator.work
        simulation.integrator.integrator.clear_log()
        print(f"{simulation.currentStep}: {work}", file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def __del__(self):
        try:
            self._out.close()
        except AttributeError:
            pass
