import openff.toolkit as toolkit
import openmm as mm
import openmm.unit as unit
from openff.interchange import Interchange


def read_protein_simulation(time_step: unit.Quantity, prod_integrator: mm.Integrator):
    openff_forcefield = toolkit.ForceField("openff_unconstrained-2.2.1.offxml", "ff14sb_off_impropers_0.0.3.offxml")
    with open('../preparation/interchange.json') as input:
        interchange: Interchange = Interchange.model_validate_json(input.read())

    simulation = interchange.to_openmm_simulation(
        combine_nonbonded_forces=True,
        integrator=prod_integrator
    )
    simulation.loadCheckpoint("../preparation/minimized_checkpoint.b")
    for i in range(simulation.system.getNumForces()):
        f = simulation.system.getForce(i)
        if f.getName() == "HarmonicBondForce":
            f.setForceGroup(1)
        else:
            f.setForceGroup(2)
    simulation.context.reinitialize(True)
    return simulation, openff_forcefield, interchange.topology
