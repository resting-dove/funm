from typing import Iterable
import numpy as np
import openmm
import openmm.app as app
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import Quantity, unit
from openmm import unit as openmm_unit
from pdbfixer import PDBFixer


# https://github.com/ingcoder/OpenMM-MDSimulation/blob/main/OpenMM_Ligand_Protein_Simulation.ipynb

def check_for_large_molecules(topology, atom_count_threshold=100):
    """Verification step. Check if there are any large molecules in the topology,
    which might indicate the presence of proteins or polymers."""
    found_large_molecule = False
    for molecule in topology.molecules:
        if len(molecule.atoms) > atom_count_threshold:
            print(f"Found a large molecule with {len(molecule.atoms)} atoms, which might be a protein or polymer.")
            found_large_molecule = True
            break

    if not found_large_molecule:
        print("No large molecules found that could indicate a protein or polymer.")
    else:
        print("Possible protein or polymer present in the topology.")


def insert_molecule_and_remove_clashes(
        topology: Topology,
        insert: Molecule,
        radius: Quantity = 1.5 * unit.angstrom,
        keep: Iterable[Molecule] = []
) -> Topology:
    """
    Add a molecule to a copy of the topology, removing any clashing molecules.

    The molecule will be added to the end of the topology. A new topology is
    returned; the input topology will not be altered. All molecules that
    clash will be removed, and each removed molecule will be printed to stdout.
    Users are responsible for ensuring that no important molecules have been
    removed; the clash radius may be modified accordingly.

    Parameters
    ==========
    top
        The topology to insert a molecule into
    insert
        The molecule to insert
    radius
        Any atom within this distance of any atom in the insert is considered
        clashing.
    keep
        Keep copies of these molecules, even if they're clashing
    """
    # We'll collect the molecules for the output topology into a list
    new_top_mols = []
    # A molecule's positions in a topology are stored as its zeroth conformer
    insert_coordinates = insert.conformers[0][:, None, :]
    for molecule in topology.molecules:
        if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep):
            new_top_mols.append(molecule)
            continue
        molecule_coordinates = molecule.conformers[0][None, :, :]
        diff_matrix = molecule_coordinates - insert_coordinates

        # np.linalg.norm doesn't work on Pint quantities 😢
        working_unit = unit.nanometer
        distance_matrix = (
                np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit
        )

        if distance_matrix.min() > radius:
            # This molecule is not clashing, so add it to the topology
            new_top_mols.append(molecule)
        else:
            print(f"Removed {molecule.to_smiles()} molecule")

    # Insert the ligand at the end
    new_top_mols.append(insert)

    # This pattern of assembling a topology from a list of molecules
    # ends up being much more efficient than adding each molecule
    # to a new topology one at a time
    new_top = Topology.from_molecules(new_top_mols)

    # Don't forget the box vectors!
    new_top.box_vectors = topology.box_vectors
    return new_top


if __name__ == '__main__':
    receptor_path = "files/5tbm_prepared.pdb"
    ligand_path = "files/PT2385.sdf"

    openff_forcefield = ForceField("openff_unconstrained-2.2.1.offxml", "ff14sb_off_impropers_0.0.3.offxml")
    openff_forcefield.deregister_parameter_handler("Constraints")

    fixer = PDBFixer(receptor_path)
    fixer.addSolvent(
        padding=1.0 * (openmm_unit.nano * openmm_unit.meter), ionicStrength=0.15 * openmm_unit.molar
    )

    with open("receptor_solvated.pdb", "w") as f:
        openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

    top = Topology.from_pdb("receptor_solvated.pdb")

    # Load a molecule from a SDF file
    ligand = Molecule.from_file(ligand_path, allow_undefined_stereo=True)

    # Print out a SMILES code for the ligand
    print(ligand.to_smiles(explicit_hydrogens=False))

    print(check_for_large_molecules(top))

    top = insert_molecule_and_remove_clashes(top, ligand)
    # Verify that protein is present and was not removed due to clashes
    print(check_for_large_molecules(top))

    with open("topology.json", "w") as f:
        print(top.to_json(), file=f)

    top = Topology.from_json(open("topology.json").read())

    interchange = openff_forcefield.create_interchange(top)
    # #
    omm_system = interchange.to_openmm(
        combine_nonbonded_forces=True,
        add_constrained_forces=True,
    )
    for i in range(omm_system.getNumForces()):
        f = omm_system.getForce(i)
        if f.getName() == "HarmonicBondForce":
            f.setForceGroup(1)
        else:
            f.setForceGroup(2)
    omm_top = interchange.to_openmm_topology()

    with open('system.xml', 'w') as output:
        output.write(openmm.XmlSerializer.serialize(omm_system))
    with open('interchange.json', 'w') as output:
        output.write(interchange.model_dump_json())

    integrator = openmm.VerletIntegrator(
        1 * openmm_unit.femtosecond,
    )

    with open('system.xml') as input:
        omm_system = openmm.XmlSerializer.deserialize(input.read())
    with open('interchange.json') as input:
        interchange: Interchange = Interchange.model_validate_json(input.read())
        top: Topology = interchange.topology
        omm_top: app.Topology = interchange.to_openmm_topology()

    # Combine the topology, system, integrator and initial positions into a simulation
    simulation = interchange.to_openmm_simulation(combine_nonbonded_forces=True,
                                                  integrator=integrator, )

    before_state = simulation.context.getState(
        getEnergy=True, getPositions=True)
    print(
        "Before minimization potential Energy is",
        before_state.getPotentialEnergy())

    simulation.minimizeEnergy(
        tolerance=openmm_unit.Quantity(
            value=50.0, unit=openmm_unit.kilojoule_per_mole / (openmm_unit.nano * openmm_unit.meter)
        )
    )
    minimized_state = simulation.context.getState(
        getPositions=True, getEnergy=True, getForces=True
    )

    print(
        "Minimised to",
        minimized_state.getPotentialEnergy(),
        "with maximum force",
        max(
            np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) for v in minimized_state.getForces()
        ),
        minimized_state.getForces().unit.get_symbol(),
    )

    minimized_coords = minimized_state.getPositions()

    # Assume 'simulation' is your Simulation object
    # and 'minimized_coords' contains the positions from the minimized state

    simulation.context.setVelocitiesToTemperature(300 * openmm_unit.kelvin)
    simulation.context.computeVirtualSites()

    # Get the topology from your simulation object
    topology = simulation.topology

    # Use the PDBFile class to write the topology and minimized coordinates to a PDB file
    with open('minimized_structure.pdb', 'w') as outfile:
        app.PDBFile.writeFile(topology, minimized_coords, outfile)

    print("Minimized structure saved to minimized_structure.pdb")

    # saveState saves the state of simulation including position for later use.
    simulation.saveState('minimized_state.xml')
    simulation.saveCheckpoint("minimized_checkpoint.b")
