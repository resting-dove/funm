import os
import time
from pprint import pformat
import openmm as mm
from openmm import app
from openmm.unit import *


# set up a generic Minimizer that runs a minimization and
# gives a nice report with timing information.

class BaseMinimizer(object):
    def __init__(self):
        # should set self.context (in the subclass)
        pass

    def minimize(self):
        # should do the minimization (in the subclass)
        pass

    def benchmark(self):
        initialEnergy = self.context.getState(getEnergy=True) \
            .getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        startTime = time.time()
        self.minimize()
        endTime = time.time()
        finalEnergy = self.context.getState(getEnergy=True) \
            .getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        reportLines = [
            '{name} with {platform} platform and {numParticles:d} particles',
            '  ({details})',
            '  initial energy = {initial:>{width}.4f} kJ/mol',
            '  final energy   = {final:>{width}.4f} kJ/mol',
            '  elapsed time   = {time:.4f} s',
            '',
        ]

        platform = self.context.getPlatform()
        properties = {k: platform.getPropertyValue(self.context, k) for k in platform.getPropertyNames()}
        deviceNames = [v for k, v in properties.items() if 'DeviceName' in k]

        report = os.linesep.join(reportLines).format(
            name=self.__class__.__name__, width=12,
            details=('device = ' + deviceNames[0] if len(deviceNames) > 0 else ''),
            numParticles=self.context.getSystem().getNumParticles(),
            platform=self.context.getPlatform().getName(),
            initial=initialEnergy, final=finalEnergy, time=(endTime - startTime))
        return report


class NesterovMinimizer(BaseMinimizer):
    """Local energy minimzation with Nesterov's accelerated gradient descent
    Source: https://nbviewer.org/github/rmcgibbo/openmm-cookbook/blob/master/02-nesterov-minimization.ipynb

    
    Parameters
    ----------
    system : mm.System
        The OpenMM system to minimize
    initialPositions : 2d array
        The positions to start from
    numIterations : int
        The number of iterations of minimization to run
    stepSize : int
        The step size. This isn't in time units.
    """

    def __init__(self, system, initialPositions, numIterations=1000, stepSize=1e-6):
        self.numIterations = numIterations

        integrator = mm.CustomIntegrator(stepSize)
        integrator.addGlobalVariable('a_cur', 0)
        integrator.addGlobalVariable('a_old', 0)
        integrator.addPerDofVariable('y_cur', 0)
        integrator.addPerDofVariable('y_old', 0)
        integrator.addComputeGlobal('a_cur', '0.5*(1+sqrt(1+(4*a_old*a_old)))')
        integrator.addComputeGlobal('a_old', 'a_cur')
        integrator.addComputePerDof('y_cur', 'x + dt*f')
        integrator.addComputePerDof('y_old', 'y_cur')
        integrator.addComputePerDof('x', 'y_cur + (a_old - 1) / a_cur * (y_cur - y_old)')

        self.context = mm.Context(system, integrator)
        self.context.setPositions(initialPositions)

    def minimize(self):
        self.context.getIntegrator().step(self.numIterations)
