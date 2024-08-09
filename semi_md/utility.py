from openmm import System
import copy
from openmm.openmm import VerletIntegrator
from openmm.unit import pico, second, Quantity
from openmm.app.simulation import Simulation
from CustomMinimizers import NesterovMinimizer


def get_rest_positions(topology, system: System, positions, print_energies=False):
    system = copy.deepcopy(system)
    integrator = VerletIntegrator(Quantity(0.0002, pico * second))
    system.removeForce(2)
    system.removeForce(1)
    system.removeForce(0)
    simulation = Simulation(topology.to_openmm(), system, integrator)
    simulation.system.getForces()
    simulation.context.setPositions(positions)
    nm = NesterovMinimizer(simulation.system, positions, numIterations=1000)
    nm.minimize()
    nstate = nm.context.getState(getPositions=True, getEnergy=True)
    if print_energies:
        print(f"kin.E: {nstate.getKineticEnergy()}, pot.E: {nstate.getPotentialEnergy()}")
    return nstate.getPositions(), nstate.getPositions(True)
