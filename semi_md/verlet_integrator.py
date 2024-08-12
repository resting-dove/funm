import numpy as np

from semi_md.molecular_system import MolecularSystem


class MyVerletIntegrator:
    def __init__(self, time_step: float, recenter=False):
        self.time_step = time_step
        self.prev_force = None
        self.recenter = recenter

    def recenter_positions(self, md: MolecularSystem):
        m = md.get_masses()[:, None]
        p = md.get_positions()
        center_positions = np.sum(p * m, axis=0) / np.sum(m)
        md.set_positions(p - center_positions)

    def remove_center_of_mass_movement(self, md: MolecularSystem):
        m = md.get_masses()[:, None]
        v = md.get_velocities()
        center_v = np.sum(v * m, axis=0) / np.sum(m)
        md.set_velocities(md.get_velocities() - center_v)

    def advance_positions(self, md: MolecularSystem):
        p = md.get_positions()
        v = md.get_velocities()
        f = md.get_forces()
        p = p + self.time_step * v + 0.5 * self.time_step ** 2 * f / md.get_masses()[:, None]
        md.set_positions(p)
        self.prev_force = f

    def advance_velocity(self, md: MolecularSystem):
        v = md.get_velocities()
        f = md.get_forces()
        v = v + 0.5 * self.time_step * (self.prev_force + f) / md.get_masses()[:, None]
        md.set_velocities(v)

    def advance_step(self, md: MolecularSystem):
        self.advance_positions(md)
        self.advance_velocity(md)
        if self.recenter:
            self.recenter_positions(md)
            self.remove_center_of_mass_movement(md)
        return md.get_positions()
