import openmm.unit as unit
import ase.units as ase_units

ase_time_base_unit = unit.BaseUnit(unit.time_dimension, name="ase time", symbol="Å sqrt(u / eV)")

ase_time_base_unit.define_conversion_factor_to(unit.second_base_unit, 1 / ase_units.second)

ase_unit_system = unit.UnitSystem([
    unit.angstrom_base_unit,
    unit.dalton_base_unit,
    ase_time_base_unit,
    unit.ev_base_unit,
    unit.kelvin_base_unit,
    unit.radian_base_unit])

eV2kJ_mol = 96.49

# Conversion factor to express Angstrom/ fs in ASE units
velocity_conversion_factor = (1 * (unit.nano * unit.meter) / (unit.pico * unit.second)).value_in_unit(
    unit.angstrom / (unit.femto * unit.second)) * ase_units.Angstrom / ase_units.fs
