from semi_md.molecular_system import MolecularSystem


class MyVerletIntegrator:
    def __init__(self, time_step: float):
        self.time_step = time_step
        self.prev_force = None

    def advance_positions(self, molecular_system: MolecularSystem):
        p = molecular_system.get_positions()
        v = molecular_system.get_velocities()
        f = molecular_system.get_forces()
        p = p + self.time_step * v + 0.5 * self.time_step ** 2 * f / molecular_system.get_masses()[:, None]
        molecular_system.set_positions(p)
        self.prev_force = f

    def advance_velocity(self, molecular_system: MolecularSystem):
        v = molecular_system.get_velocities()
        f = molecular_system.get_forces()
        v = v + 0.5 * self.time_step * (self.prev_force + f) / molecular_system.get_masses()[:, None]
        molecular_system.set_velocities(v)

    def advance_step(self, molecular_system: MolecularSystem):
        self.advance_positions(molecular_system)
        self.advance_velocity(molecular_system)
        return molecular_system.get_positions()
