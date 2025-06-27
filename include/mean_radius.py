import jax
import jax.numpy as jnp

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
        self.nucleation_site_density = jnp.array([1.0e20])  # number / m^3

        self.time = jnp.array([1.0e-10])  # s
        self.mean_radius = jnp.array([0.0])  # m
        self.volume_fraction = jnp.array([0.0])  # m^3 / m^3
        self.n_precipitates = jnp.array([0.0])  # number

    @staticmethod
    def compute_volumetric_driving_force(
        temperature, solution_composition, molar_volume_precipitate
    ):
        R = 8.314  # J/mol/K
        return (
            -R
            * temperature
            * (
                solution_composition * jnp.log(solution_composition)
                + (1.0 - solution_composition) * jnp.log(1.0 - solution_composition)
            )
            / molar_volume_precipitate
        )

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
    def compute_gibbs_energy(radius, driving_force, surface_energy):
        return (
            4.0 / 3.0 * jnp.pi * radius**3 * driving_force
            + 4.0 * jnp.pi * radius**2 * surface_energy
        )

    @staticmethod
    def compute_critical_radius(driving_force, surface_energy):
        return 2.0 * surface_energy / driving_force

    @staticmethod
    def compute_zeldovich_factor(
        mean_atomic_volume, critical_radius, surface_energy, temperature
    ):
        # Check if the units are correct
        boltzmann_constant = 1.380649e-23  # J/K

        print(boltzmann_constant * temperature)

        print(1.0 / jnp.sqrt(jnp.pi) / 3.87305579e-12)

        return (
            mean_atomic_volume
            / (2.0 * jnp.pi * critical_radius)
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

    @staticmethod
    def compute_nucleus_size(critical_radius, zeldovich_factor):
        return critical_radius + 1.0 / (jnp.sqrt(jnp.pi) * zeldovich_factor)

    @staticmethod
    def compute_alpha_parameter(
        mean_atomic_volume_solution, mean_atomic_volume_precipitate
    ):
        return mean_atomic_volume_solution / mean_atomic_volume_precipitate

    def compute_precipitation_rate(self):
        # First calculate the rate for nucleation events
        critical_radius = self.compute_critical_radius(
            self.compute_volumetric_driving_force(
                self.temperature,
                self.solution_composition,
                self.molar_volume_precipitate,
            ),
            self.surface_energy,
        )

        condensation_rate = self.compute_condensation_rate(
            critical_radius,
            self.diffusivity,
            self.solution_composition,
            self.lattice_parameter,
        )

        zeldovich_factor = self.compute_zeldovich_factor(
            self.mean_atomic_volume_precipitate,
            critical_radius,
            self.surface_energy,
            self.temperature,
        )

        incubation_time = self.compute_incubation_time(
            condensation_rate, zeldovich_factor
        )

        gibbs_energy = self.compute_gibbs_energy(
            critical_radius,
            self.compute_volumetric_driving_force(
                self.temperature,
                self.solution_composition,
                self.molar_volume_precipitate,
            ),
            self.surface_energy,
        )

        nucleation_rate = (
            self.nucleation_site_density
            * zeldovich_factor
            * condensation_rate
            * jnp.exp(-gibbs_energy / (self.boltzmann_constant * self.temperature))
            * jnp.exp(-incubation_time / self.time)
        )

        print(f"Critical radius (m): {critical_radius}")
        print(f"Condensation rate (1/s): {condensation_rate}")
        print(f"Zeldovich factor: {zeldovich_factor}")
        print(f"Incubation time (s): {incubation_time}")
        print(f"Gibbs energy (J): {gibbs_energy}")
        print(
            f"Energy exponent: {-gibbs_energy / (self.boltzmann_constant * self.temperature)}"
        )
        print(f"Time exponent: {-incubation_time / self.time}")
        print(f"Nucleation rate (1/s): {nucleation_rate}")

    def compute_growth_rate(self):
        return (
            self.diffusivity
            / self.mean_radius
            * self.compute_supersaturation(
                self.precipitate_composition,
                self.solution_composition,
                self.equilibrium_solution_composition,
                self.alpha_parameter,
            )
            + 1.0 / self.n_precipitates
        )

    def compute_coarsening_rate(self):
        pass

    def update_mean_radius(self):
        pass

    def compute_mean_radius(self, composition, temperature, pressure):
        self.compute_precipitation_rate()
