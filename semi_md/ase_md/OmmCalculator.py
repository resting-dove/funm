from typing import Optional

import ase
import numpy as np
import openmm as mm
import openmm.unit as unit
from ase.calculators.calculator import all_changes
from nanover.ase.converter import KJMOL_TO_EV
from nanover.ase.omm_calculator import OpenMMCalculator
from openmm.unit import kilojoules_per_mole

from semi_md.computeK import compute_K_with_dict
from unit_helpers import ase_unit_system, ase_time_base_unit


class OmmCalculator(OpenMMCalculator):
    """Adaption of Nanover's OpenMMCalculator with extended use of OpenMM units and option to select Force Groups."""

    def calculate(
            self,
            atoms: Optional[ase.Atoms] = None,
            properties=("energy", "forces"),
            system_changes=all_changes,
            groups=None
    ):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                "No ASE atoms supplied to calculator, and no ASE atoms supplied with initialisation."
            )

        self._set_positions(atoms.positions)
        energy, forces = self._calculate_openmm(groups)
        self.results["energy"] = energy
        self.results["forces"] = forces

    def _calculate_openmm(self, groups=None):
        """groups : int
            a set of bit flags for which force groups to include when computing forces and energies. Group i will be
            included if (groups&(1<<i)) != 0. The default value includes all groups."""
        if not groups is None:
            state: mm.State = self.context.getState(getEnergy=True, getForces=True, groups=groups)
        else:
            state: mm.State = self.context.getState(getEnergy=True, getForces=True)
        energy_kj_mol = state.getPotentialEnergy()
        energy = energy_kj_mol.value_in_unit(kilojoules_per_mole) * KJMOL_TO_EV
        forces_openmm = state.getForces(asNumpy=True)
        forces_angstrom = forces_openmm.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
        forces = forces_angstrom * KJMOL_TO_EV
        return energy, forces

    def calculate_k(self, original_positions, forcefield, openff_topology):
        return compute_K_with_dict(original_positions,
                                   forcefield.get_parameter_handler("Bonds").find_matches(openff_topology, True),
                                   ase_unit_system)

    def _get_positions(self):
        """Get positions in Angstrom, the ASE unit."""
        return self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit_system(ase_unit_system)

    def _set_positions(self, positions):
        """Set the position in Angstrom, the ASE unit."""
        self.context.setPositions(positions * unit.angstrom)

    def _get_velocities(self) -> np.array:
        """Get positions in the ASE units."""
        return self.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit_system(
            ase_unit_system)

    def _set_velocities(self, velocities):
        """Set the position in the ASE unit system."""
        self.context.setVelocities(velocities * (unit.angstrom / unit.Unit({ase_time_base_unit: 1.0})))
