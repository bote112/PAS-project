import taichi as ti
import numpy as np

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        # Convert gravity to Taichi vector field
        self.g = ti.Vector.field(self.ps.dim, dtype=ti.f32, shape=())
        self.g.from_numpy(np.array(self.ps.config['gravitation'], dtype=np.float32))
        
        # Time step management
        self.dt = ti.field(ti.f32, shape=())
        self.dt[None] = self.ps.config['dt']
        
        # Precompute frequently used constants
        self.collision_factor = self.ps.config['collisionFactor']
        self.viscosity = ti.field(ti.f32, shape=())
        self.viscosity[None] = self.ps.config['viscosity']
        self.c_s = self.ps.config['c_s']
        self.kernel_coeff = self._precompute_kernel_coefficients()

    def _precompute_kernel_coefficients(self):
        """Precompute kernel coefficients during initialization"""
        h = self.ps.support_length
        dim = self.ps.dim
        coeff_3d = 8 / np.pi
        coeff_2d = 40 / (7 * np.pi)
        return {
            'base': ti.select(dim == 3, coeff_3d, coeff_2d) / (h ** dim),
            'deriv': ti.select(dim == 3, 16/np.pi, 80/(7*np.pi)) / (h ** (dim + 1))
        }

    @ti.func
    def cubic_spline_kernel(self, r_norm):
        """Optimized cubic spline kernel implementation"""
        h = self.ps.support_length
        q = r_norm / h
        kernel_val = 0.0
        
        if q <= 1.0:
            coeff = self.kernel_coeff['base']
            if q <= 0.5:
                q2 = q * q
                kernel_val = coeff * (1.0 - 6.0 * q2 + 6.0 * q * q2)
            else:
                omq = 1.0 - q
                kernel_val = coeff * (2.0 * omq * omq * omq)
        return kernel_val

    @ti.func
    def cubic_spline_kernel_derivative(self, r):
        """Optimized derivative calculation with reduced operations"""
        h = self.ps.support_length
        r_norm = r.norm()
        q = r_norm / h
        derivative = ti.Vector.zero(ti.f32, self.ps.dim)
        
        if q <= 1.0 and r_norm > 1e-7:
            coeff = self.kernel_coeff['deriv']
            r_hat = r / r_norm  # Safe normalization
            
            if q <= 0.5:
                derivative = coeff * (9.0 * q * q - 6.0 * q) * r_hat
            else:
                omq = 1.0 - q
                derivative = coeff * (-3.0 * omq * omq) * r_hat
        return derivative

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta_bi):
        """Optimized boundary volume calculation"""
        if self.ps.material[p_j] == self.ps.material_rigid:
            delta_bi += self.cubic_spline_kernel(
                (self.ps.position[p_i] - self.ps.position[p_j]).norm())
    
    @ti.kernel
    def compute_volume_of_boundary_particle(self):
        """Optimized volume computation kernel"""
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                delta_bi = self.cubic_spline_kernel(0.0)
                self.ps.for_all_neighbors(i, self.compute_boundary_volume_task, delta_bi)
                self.ps.volume[i] = 1.0 / delta_bi

    @ti.func
    def simulate_collision(self, idx, vec):
        """Optimized collision response calculation"""
        v_dot = self.ps.velocity[idx].dot(vec)
        self.ps.velocity[idx] -= (1.0 + self.collision_factor) * v_dot * vec

    @ti.kernel
    def enforce_boundary_3D(self):
        """Optimized boundary enforcement with reduced memory accesses"""
        domain_end = ti.Vector([self.ps.domain_end[d] for d in ti.static(range(self.ps.dim))])
        padding = self.ps.padding
        
        for i in range(self.ps.total_particle_num):
            if self.ps.is_dynamic[i]:
                pos = self.ps.position[i]
                collision_vec = ti.Vector.zero(ti.f32, self.ps.dim)
                
                for d in ti.static(range(self.ps.dim)):
                    if pos[d] > domain_end[d] - padding:
                        collision_vec[d] = 1.0
                        pos[d] = domain_end[d] - padding
                    elif pos[d] < padding:
                        collision_vec[d] = -1.0
                        pos[d] = padding
                
                self.ps.position[i] = pos
                vec_norm = collision_vec.norm()
                
                if vec_norm > 1e-6:
                    self.simulate_collision(i, collision_vec / vec_norm)

    def initialize(self):
        self.ps.update_particle_system()
        self.compute_volume_of_boundary_particle()

    def substep(self):
        pass

    def step(self):
        self.ps.update_particle_system()
        self.substep()
        self.enforce_boundary_3D()