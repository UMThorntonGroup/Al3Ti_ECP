import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)


class MeanRadius:
    def __init__(self):
        # Input parameters
        self.precipitate_composition = jnp.array([1.0])  # mol/mol
        self.solution_composition = jnp.array([0.06])  # mol/mol
        self.equilibrium_solution_composition = jnp.array([0.01])  # mol/mol
        self.diffusivity = jnp.array([5e-20])  # m^2/s
        self.surface_energy = jnp.array([0.13])  # J/m^2
        self.mean_atomic_volume_solution = jnp.array([1.6e-29])  # m^3
        self.mean_atomic_volume_precipitate = jnp.array([1.6e-29])  # m^3
        self.lattice_parameter = jnp.array([4.04e-10])  # m
        self.temperature = jnp.array([433.0])  # K

        # Constants
        self.R = 8.314  # J/mol/K
        self.boltzmann_constant = 1.380e-23  # J/K
        self.avogadro_number = 6.022e23  # 1/mol

        # Calculated parameters
        self.alpha_parameter = self.compute_alpha_parameter(
            self.mean_atomic_volume_solution, self.mean_atomic_volume_precipitate
        )
        self.molar_volume_precipitate = (
            self.mean_atomic_volume_precipitate * self.avogadro_number
        )
        self.total_composition = self.solution_composition

        # This needs to be fit
        self.nucleation_site_density = jnp.array([1.0e20])  # number / m^3

        self.coarsening_radius_constant = (
            2.0
            * self.surface_energy
            * self.mean_atomic_volume_precipitate
            / (self.boltzmann_constant * self.temperature)
        )
        self.nuclei_rate = jnp.array([0.0])  # 1/s

        # Initial conditions
        self.time = jnp.array([1.0])  # s
        self.mean_radius = jnp.array([1.0e-10])  # m
        self.volume_fraction = jnp.array([0.0])  # m^3 / m^3
        self.n_precipitates = jnp.array([1.0])  # number

        # Cached variables
        self.volumetric_driving_force = self.compute_volumetric_driving_force()
        self.critical_radius = self.compute_critical_radius(
            self.volumetric_driving_force, self.surface_energy
        )
        self.condensation_rate = self.compute_condensation_rate(
            self.critical_radius,
            self.diffusivity,
            self.solution_composition,
            self.lattice_parameter,
        )
        self.zeldovich_factor = self.compute_zeldovich_factor(
            self.mean_atomic_volume_precipitate,
            self.critical_radius,
            self.surface_energy,
            self.temperature,
        )
        self.incubation_time = self.compute_incubation_time(
            self.condensation_rate, self.zeldovich_factor
        )
        self.gibbs_energy = self.compute_gibbs_energy(
            self.critical_radius,
            self.volumetric_driving_force,
            self.surface_energy,
        )
        self.nucleus_size = self.compute_nucleus_size(
            self.volumetric_driving_force,
            self.surface_energy,
        )

    def compute_volumetric_driving_force(self):
        value = (
            self.R
            * self.temperature
            * (
                self.solution_composition * jnp.log(self.solution_composition)
                + (1.0 - self.solution_composition)
                * jnp.log(1.0 - self.solution_composition)
            )
            / self.molar_volume_precipitate
        )
        return value

    @staticmethod
    def compute_supersaturation(
        precipitate_composition,
        solution_composition,
        equilibrium_solution_composition,
        alpha_parameter,
    ):
        return (solution_composition - equilibrium_solution_composition) / (
            alpha_parameter * precipitate_composition - equilibrium_solution_composition
        )

    @staticmethod
    @jit
    def compute_gibbs_energy(radius, driving_force, surface_energy):
        return (
            4.0 / 3.0 * jnp.pi * radius**3 * driving_force
            + 4.0 * jnp.pi * radius**2 * surface_energy
        )

    @staticmethod
    def compute_critical_radius(driving_force, surface_energy):
        return -2.0 * surface_energy / driving_force

    @staticmethod
    def compute_critical_driving_force(driving_force, surface_energy):
        return 16.0 * jnp.pi * surface_energy**3 / (3.0 * driving_force)

    @staticmethod
    def compute_zeldovich_factor(
        mean_atomic_volume, critical_radius, surface_energy, temperature
    ):
        boltzmann_constant = 1.380649e-23  # J/K
        return (
            2.0
            * mean_atomic_volume
            / (3.0 * critical_radius**2)
            * jnp.sqrt(surface_energy / (boltzmann_constant * temperature))
        )

    @staticmethod
    def compute_condensation_rate(
        critical_radius, diffusivity, mean_solute_atom_fraction, lattice_parameter
    ):
        # We have to be careful here with roundoff errors, so we'll
        # nondimensionalize the length scale by dividing by the critical radius
        critical_radius_nondimensional = critical_radius / critical_radius
        diffusivity_nondimensional = diffusivity / critical_radius**2
        lattice_parameter_nondimensional = lattice_parameter / critical_radius
        return (
            4.0
            * jnp.pi
            * critical_radius_nondimensional**2
            * diffusivity_nondimensional
            * mean_solute_atom_fraction
            / lattice_parameter_nondimensional**4
        )

    @staticmethod
    def compute_incubation_time(condensation_rate, zeldovich_factor):
        return 4.0 / (2.0 * jnp.pi * condensation_rate * zeldovich_factor**2)

    def compute_nucleus_size(self, driving_force, surface_energy):
        def critical_energy():
            return (
                self.compute_critical_driving_force(driving_force, surface_energy)
                - self.boltzmann_constant * self.temperature
            )

        @jit
        def f(x):
            return (
                self.compute_gibbs_energy(x, driving_force, surface_energy)
                - critical_energy()
            )

        def bisection_method(f, a, b, tol=1e-6, steps=10000):
            c = a
            step = 0
            while f(c) > tol:
                if step > steps:
                    raise ValueError("Bisection method did not converge")
                c = (a + b) / 2.0
                if f(c) * f(a) < 0.0:
                    b = c
                else:
                    a = c
                step += 1

            return c

        # Since we know that the root is somewhere about the critical radius we can
        # start a simple bisection method using that as a lower bound
        return bisection_method(f, self.critical_radius, 10.0 * self.critical_radius)

    @staticmethod
    def compute_alpha_parameter(
        mean_atomic_volume_solution, mean_atomic_volume_precipitate
    ):
        return mean_atomic_volume_solution / mean_atomic_volume_precipitate

    def compute_precipitation_rate(self):
        return (
            self.nucleation_site_density
            * self.zeldovich_factor
            * self.condensation_rate
            * jnp.exp(-self.gibbs_energy / (self.boltzmann_constant * self.temperature))
            * jnp.exp(-self.incubation_time / self.time)
        )

    def compute_growth_rate(self):
        return self.diffusivity / self.mean_radius * self.compute_supersaturation(
            self.precipitate_composition,
            self.solution_composition,
            self.equilibrium_solution_composition,
            self.alpha_parameter,
        ) + 1.0 / self.n_precipitates * self.nuclei_rate * (
            self.nucleus_size - self.mean_radius
        )

    def compute_coarsening_rate(self):
        return (
            (
                4.0
                / 27.0
                * (
                    self.equilibrium_solution_composition
                    / (
                        self.alpha_parameter * self.precipitate_composition
                        - self.equilibrium_solution_composition
                    )
                )
            )
            * (self.coarsening_radius_constant * self.diffusivity / self.mean_radius**3)
            * (
                self.coarsening_radius_constant
                * self.equilibrium_solution_composition
                / (
                    self.mean_radius
                    * (
                        self.precipitate_composition
                        - self.equilibrium_solution_composition
                    )
                )
                * (3.0 / (4.0 * jnp.pi * self.mean_radius**3) - self.n_precipitates)
                - 3.0 * self.n_precipitates
            )
        )

    def compute_coarsening_fraction(self):
        if (self.mean_radius < 1.01 * self.critical_radius) and (
            self.mean_radius > 0.99 * self.critical_radius
        ):
            return 1.0 - 1000.0 * (self.mean_radius / self.critical_radius - 1.0) ** 2
        else:
            return 1.0 - jax.scipy.special.erf(
                4.0 * (self.mean_radius / self.critical_radius - 1.0)
            )

    def compute_nuclei_rate(self):
        # Compute the various nucleation and coarsening rates
        precipitation_rate = self.compute_precipitation_rate()
        coarsening_rate = self.compute_coarsening_rate()
        coarsening_fraction = self.compute_coarsening_fraction()

        # Compute the rate of change of the number of precipitates
        if -coarsening_rate > precipitation_rate:
            return coarsening_fraction * coarsening_rate
        else:
            return precipitation_rate

    def update_solute_fraction(self):
        return (
            self.total_composition
            - self.alpha_parameter
            * 4.0
            / 3.0
            * jnp.pi
            * self.mean_radius**3
            * self.n_precipitates
            * self.precipitate_composition
        ) / (
            1.0
            - self.alpha_parameter
            * 4.0
            / 3.0
            * jnp.pi
            * self.mean_radius**3
            * self.n_precipitates
        )

    def update_mean_radius(self):
        self.nuclei_rate = self.compute_nuclei_rate()
        growth_rate = self.compute_growth_rate()
        self.solution_composition = self.update_solute_fraction()

        # Apply the runge-kutta method to update the mean radius
        timestep = 10.0

        self.n_precipitates = self.n_precipitates + self.nuclei_rate * timestep
        self.mean_radius = self.mean_radius + growth_rate * timestep

        self.time = self.time + timestep

        print(f"Mean radius: {self.mean_radius}")
        print(f"N precipitates: {self.n_precipitates}")
        print(f"Time: {self.time}")
        print(f"Solution composition: {self.solution_composition}")

    def compute_mean_radius(self, composition, temperature, pressure):
        for i in range(10):
            self.update_mean_radius()

        return self.mean_radius
