import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

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
            self.surface_energy,
        )

        # Initial conditions
        self.time = jnp.array([1.0])  # s
        self.mean_radius = self.critical_radius  # m
        self.volume_fraction = jnp.array([0.0])  # m^3 / m^3
        self.n_precipitates = jnp.array([1.0])  # number / m^3

        # Data arrays
        self.time_evolution = []
        self.mean_radius_evolution = []
        self.volume_fraction_evolution = []
        self.precipitate_density_evolution = []
        self.solution_composition_evolution = []

    def compute_volumetric_driving_force(self):
        assert self.solution_composition > 0.0
        assert self.solution_composition < 1.0
        assert self.temperature > 0.0

        def ideal_mixing_term(composition):
            return (
                self.R
                * self.temperature
                * (
                    composition * jnp.log(composition)
                    + (1.0 - composition) * jnp.log(1.0 - composition)
                )
            )

        def ideal_mixing_term_derivative(composition):
            return (
                self.R
                * self.temperature
                * (jnp.log(composition) - jnp.log(1.0 - composition))
            )

        chemical_potential_precipitate = ideal_mixing_term_derivative(
            self.equilibrium_solution_composition
        ) * (
            self.precipitate_composition - self.equilibrium_solution_composition
        ) + ideal_mixing_term(
            self.equilibrium_solution_composition
        )

        chemical_potential_supersaturated_solution = ideal_mixing_term_derivative(
            self.solution_composition
        ) * (
            self.precipitate_composition - self.solution_composition
        ) + ideal_mixing_term(
            self.solution_composition
        )

        driving_force = (
            chemical_potential_precipitate - chemical_potential_supersaturated_solution
        ) / self.molar_volume_precipitate

        return driving_force

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
        return -2.0 * surface_energy / driving_force

    @staticmethod
    def compute_critical_driving_force(driving_force, surface_energy):
        return 16.0 * jnp.pi * surface_energy**3 / (3.0 * driving_force**2)

    @staticmethod
    def compute_zeldovich_factor(
        mean_atomic_volume, critical_radius, surface_energy, temperature
    ):
        boltzmann_constant = 1.380649e-23  # J/K
        return (
            mean_atomic_volume
            / (2.0 * jnp.pi * critical_radius**2)
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

    def compute_nucleus_size(self, surface_energy):
        return self.critical_radius + 0.5 * jnp.sqrt(
            self.boltzmann_constant * self.temperature / (jnp.pi * surface_energy)
        )

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

        print(f"Coarsening rate: {coarsening_rate}")
        print(f"Precipitation rate: {precipitation_rate}")
        print(f"Coarsening fraction: {coarsening_fraction}")

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
        timestep = 0.1

        self.n_precipitates = self.n_precipitates + self.nuclei_rate * timestep
        self.mean_radius = self.mean_radius + growth_rate * timestep

        self.time = self.time + timestep

        print(f"Mean radius: {self.mean_radius}")
        print(f"Precipitates per unit volume:: {self.n_precipitates}")
        print(f"Time: {self.time}")
        print(f"Solution composition: {self.solution_composition}")

        # Append to the evolution arrays
        self.time_evolution.append(self.time)
        self.mean_radius_evolution.append(self.mean_radius)
        self.volume_fraction_evolution.append(self.volume_fraction)
        self.precipitate_density_evolution.append(self.n_precipitates)
        self.solution_composition_evolution.append(self.solution_composition)

    def compute_mean_radius(self, composition, temperature, pressure):
        for i in range(1):
            self.update_mean_radius()

        # Plot the stuff
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        ax = axs[0, 0]
        ax.plot(self.time_evolution, self.mean_radius_evolution)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean radius (m)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax = axs[0, 1]
        ax.plot(self.time_evolution, self.precipitate_density_evolution)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Precipitate density (number/m^3)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax = axs[1, 0]
        ax.plot(self.time_evolution, self.solution_composition_evolution)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration (%)")
        ax.set_xscale("log")

        ax = axs[1, 1]
        ax.plot(self.time_evolution, self.volume_fraction_evolution)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Volume fraction")
        ax.set_xscale("log")
        plt.savefig("mean_radius_evolution.png", dpi=300)

        return self.mean_radius
