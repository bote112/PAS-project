import taichi as ti
import SPHMain

class WCSPHSolver(SPHMain.SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.gamma = self.ps.config['gamma']
        self.B = self.ps.config['B']
        self.surface_tension = ti.field(ti.f32, shape=())
        self.surface_tension[None] = self.ps.config['surfaceTension']

    @ti.func
    def calculate_particle_density_contribution(self, particle_i, particle_j, density: ti.template()):
        pos_i = self.ps.position[particle_i]
        pos_j = self.ps.position[particle_j]
        r = (pos_i - pos_j).norm()
        kernel = self.cubic_spline_kernel(r)
        if self.ps.material[particle_j] == self.ps.material_fluid:
            density += self.ps.mass[particle_j] * kernel
        elif self.ps.material[particle_j] == self.ps.material_rigid:
            density += self.ps.density0 * self.ps.volume[particle_j] * kernel

    @ti.kernel
    def update_density(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                density = self.ps.mass[i] * self.cubic_spline_kernel(0.0)
                self.ps.for_all_neighbors(i, self.calculate_particle_density_contribution, density)
                self.ps.density[i] = density

    @ti.kernel
    def update_pressure(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                self.ps.density[i] = ti.max(self.ps.density[i], self.ps.density0)
                self.ps.pressure[i] = self.B * ((self.ps.density[i] / self.ps.density0) ** self.gamma - 1)

    @ti.func
    def compute_artificial_viscosity(self, velocity_diff, position_diff, viscosity_coeff):
        dot_vr = velocity_diff.dot(position_diff)
        dot_rr = position_diff.dot(position_diff)
        denominator = dot_rr + 0.01 * self.ps.support_length ** 2
        return -viscosity_coeff * ti.min(dot_vr, 0.0) / denominator

    @ti.func
    def calculate_pressure_force_contribution(self, particle_i, particle_j, acceleration: ti.template()):
        pos_i = self.ps.position[particle_i]
        pos_j = self.ps.position[particle_j]
        displacement = pos_i - pos_j
        kernel_gradient = self.cubic_spline_kernel_derivative(displacement)
        pressure_ratio_i = self.ps.pressure[particle_i] / self.ps.density[particle_i] ** 2

        if self.ps.material[particle_j] == self.ps.material_fluid:
            mass_j = self.ps.mass[particle_j]
            pressure_ratio_j = self.ps.pressure[particle_j] / (self.ps.density[particle_j] ** 2)
            acceleration -= mass_j * (pressure_ratio_i + pressure_ratio_j) * kernel_gradient
        else:
            volume_density_factor = self.ps.density0 * self.ps.volume[particle_j]
            force_contribution = -volume_density_factor * pressure_ratio_i * kernel_gradient
            acceleration += force_contribution

            if self.ps.is_dynamic_rigid_body(particle_j):
                self.ps.acceleration[particle_j] -= force_contribution * self.ps.mass[particle_i] / self.ps.mass[particle_j]

    @ti.kernel
    def compute_pressure_force(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                self.ps.acceleration[i].fill(0.0)
            elif self.ps.material[i] == self.ps.material_fluid:
                acceleration = ti.Vector.zero(ti.f32, self.ps.dim)
                self.ps.for_all_neighbors(i, self.calculate_pressure_force_contribution, acceleration)
                self.ps.acceleration[i] += acceleration

    @ti.func
    def calculate_non_pressure_forces(self, particle_i, particle_j, acceleration: ti.template()):
        pos_i = self.ps.position[particle_i]
        pos_j = self.ps.position[particle_j]
        displacement = pos_i - pos_j
        r = displacement.norm()

        if self.ps.material[particle_j] == self.ps.material_fluid:
            acceleration -= (self.surface_tension[None] / self.ps.mass[particle_i] *
                            self.ps.mass[particle_j] * displacement * self.cubic_spline_kernel(r))
            viscosity_coeff = (2 * self.viscosity[None] * self.ps.support_length * self.c_s /
                            (self.ps.density[particle_i] + self.ps.density[particle_j]))
            velocity_diff = self.ps.velocity[particle_i] - self.ps.velocity[particle_j]
            artificial_viscosity = self.compute_artificial_viscosity(velocity_diff, displacement, viscosity_coeff)
            acceleration -= (self.ps.mass[particle_j] * artificial_viscosity *
                            self.cubic_spline_kernel_derivative(displacement))
        else:
            sigma = self.ps.rigid_bodies_sigma[self.ps.object_id[particle_j]]
            viscosity_coeff = (sigma * self.ps.support_length * self.c_s /
                            (2 * self.ps.density[particle_i]))
            velocity_diff = self.ps.velocity[particle_i] - self.ps.velocity[particle_j]
            artificial_viscosity = self.compute_artificial_viscosity(velocity_diff, displacement, viscosity_coeff)
            acceleration -= (self.ps.density0 * self.ps.volume[particle_j] * artificial_viscosity *
                            self.cubic_spline_kernel_derivative(displacement))

    @ti.kernel
    def compute_non_pressure_force(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                self.ps.acceleration[i].fill(0.0)
            else:
                # Corrected gravity vector access
                acceleration = self.g[None]  # Access the stored vector value
                self.ps.acceleration[i] = acceleration
                if self.ps.material[i] == self.ps.material_fluid:
                    self.ps.for_all_neighbors(i, self.calculate_non_pressure_forces, acceleration)
                    self.ps.acceleration[i] = acceleration

    @ti.kernel
    def advect(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_dynamic[i]:
                self.ps.velocity[i] += self.ps.acceleration[i] * self.dt[None]
                self.ps.position[i] += self.ps.velocity[i] * self.dt[None]

    def substep(self):
        self.update_density()
        self.update_pressure()
        self.compute_non_pressure_force()
        self.compute_pressure_force()
        self.advect()