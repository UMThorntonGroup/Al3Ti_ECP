import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

DEBUG = True


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def pprint(message, type="None"):
    assert isinstance(message, str), "The message must be a string"
    assert isinstance(type, str), "The message type must be a string"
    if DEBUG:
        new_message = message
        if type == "Warning":
            new_message = bcolors.WARNING + message + bcolors.ENDC
        elif type == "Info":
            new_message = bcolors.OKBLUE + message + bcolors.ENDC
        elif type == "Good":
            new_message = bcolors.OKGREEN + message + bcolors.ENDC
        elif type == "Error":
            new_message = bcolors.FAIL + message + bcolors.ENDC

        print(new_message)


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
        self.volumetric_driving_force = self.compute_volumetric_driving_force(
            self.solution_composition,
            self.equilibrium_solution_composition,
            self.precipitate_composition,
            self.molar_volume_precipitate,
            self.temperature,
        )
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
            self.critical_radius, self.surface_energy, self.temperature
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

    @staticmethod
    def compute_volumetric_driving_force(
        solution_composition,
        equilibrium_solution_composition,
        precipitate_composition,
        molar_volume_precipitate,
        temperature,
        R=jnp.array([8.314]),
    ):
        assert isinstance(
            solution_composition, jax.Array
        ), "The solution composition vector must be a JAX array"
        assert isinstance(
            equilibrium_solution_composition, jax.Array
        ), "The equilibrium solution composition vector must be a JAX array"
        assert isinstance(
            precipitate_composition, jax.Array
        ), "The precipitate composition vector must be a JAX array"
        assert isinstance(
            molar_volume_precipitate, jax.Array
        ), "The molar volume of the precipitate vector must be a JAX array"
        assert isinstance(
            temperature, jax.Array
        ), "The temperature vector must be a JAX array"
        assert isinstance(R, jax.Array), "The gas constant must be a JAX array"
        assert jnp.all(
            solution_composition > 0.0
        ), "All solution compositions must be greater than 0.0"
        assert jnp.all(
            solution_composition < 1.0
        ), "All solution compositions must be less than 1.0"
        assert jnp.all(
            temperature > 0.0
        ), "All the temperatues must be greater than 0.0"
        assert jnp.size(solution_composition) == jnp.size(
            equilibrium_solution_composition
        ), "The vectors must be the same size"
        assert jnp.size(solution_composition) == jnp.size(
            precipitate_composition
        ), "The vectors must be the same size"
        assert jnp.size(solution_composition) == jnp.size(
            molar_volume_precipitate
        ), "The vectors must be the same size"
        assert jnp.size(solution_composition) == jnp.size(
            temperature
        ), "The vectors must be the same size"

        def ideal_mixing_term(composition):
            return (
                R
                * temperature
                * (
                    composition * jnp.log(composition)
                    + (1.0 - composition) * jnp.log(1.0 - composition)
                )
            )

        def ideal_mixing_term_derivative(composition):
            return R * temperature * (jnp.log(composition) - jnp.log(1.0 - composition))

        chemical_potential_precipitate = ideal_mixing_term_derivative(
            equilibrium_solution_composition
        ) * (
            precipitate_composition - equilibrium_solution_composition
        ) + ideal_mixing_term(
            equilibrium_solution_composition
        )

        chemical_potential_supersaturated_solution = ideal_mixing_term_derivative(
            solution_composition
        ) * (precipitate_composition - solution_composition) + ideal_mixing_term(
            solution_composition
        )

        pprint(
            f"Precipitate chemical potential (J/mol): {chemical_potential_precipitate}",
            "Info",
        )
        pprint(
            f"Solution chemical potential (J/mol): {chemical_potential_supersaturated_solution}",
            "Info",
        )
        pprint(
            f"Precipitate molar volume (m^3/mol): {molar_volume_precipitate}", "Info"
        )

        driving_force = (
            chemical_potential_precipitate - chemical_potential_supersaturated_solution
        ) / molar_volume_precipitate

        pprint(f"Volumetric driving force (J/m^3): {driving_force}", "Info")

        return driving_force

    @staticmethod
    def compute_supersaturation(
        precipitate_composition,
        solution_composition,
        equilibrium_solution_composition,
        alpha_parameter,
    ):
        assert isinstance(
            solution_composition, jax.Array
        ), "The solution composition vector must be a JAX array"
        assert isinstance(
            equilibrium_solution_composition, jax.Array
        ), "The equilibrium solution composition vector must be a JAX array"
        assert isinstance(
            precipitate_composition, jax.Array
        ), "The precipitate composition vector must be a JAX array"
        assert isinstance(
            alpha_parameter, jax.Array
        ), "The alpha parameter vector must be a JAX array"
        assert jnp.all(
            solution_composition > 0.0
        ), "All solution compositions must be greater than 0.0"
        assert jnp.all(
            solution_composition < 1.0
        ), "All solution compositions must be less than 1.0"
        assert jnp.size(solution_composition) == jnp.size(
            equilibrium_solution_composition
        ), "The vectors must be the same size"
        assert jnp.size(solution_composition) == jnp.size(
            precipitate_composition
        ), "The vectors must be the same size"
        assert jnp.size(solution_composition) == jnp.size(
            alpha_parameter
        ), "The vectors must be the same size"

        supersaturation = (solution_composition - equilibrium_solution_composition) / (
            alpha_parameter * precipitate_composition - equilibrium_solution_composition
        )

        pprint(f"Supersaturation: {supersaturation}", "Info")

        return supersaturation

    @staticmethod
    def compute_gibbs_energy(radius, driving_force, surface_energy):
        assert isinstance(radius, jax.Array), "The radius vector must be a JAX array"
        assert isinstance(
            driving_force, jax.Array
        ), "The driving force must be a JAX array"
        assert isinstance(
            surface_energy, jax.Array
        ), "The surface energy must be a JAX array"
        assert jnp.all(
            radius > 0.0
        ), "All entries in the radius vector must be greater than 0.0"
        assert jnp.all(
            surface_energy > 0.0
        ), "All entries in the surface energy vector must be greater than 0.0"
        assert jnp.size(radius) == jnp.size(
            driving_force
        ), "The vectors must be the same size"
        assert jnp.size(radius) == jnp.size(
            surface_energy
        ), "The vectors must be the same size"

        gibbs_energy = (
            4.0 / 3.0 * jnp.pi * radius**3 * driving_force
            + 4.0 * jnp.pi * radius**2 * surface_energy
        )

        pprint(f"Gibbs energy (J): {gibbs_energy}", "Info")

        return gibbs_energy

    @staticmethod
    def compute_critical_radius(driving_force, surface_energy):
        assert isinstance(
            driving_force, jax.Array
        ), "The driving force must be a JAX array"
        assert isinstance(
            surface_energy, jax.Array
        ), "The surface energy must be a JAX array"
        assert jnp.all(
            surface_energy > 0.0
        ), "All entries in the surface energy vector must be greater than 0.0"
        assert jnp.size(driving_force) == jnp.size(
            surface_energy
        ), "The vectors must be the same size"

        critical_radius = -2.0 * surface_energy / driving_force

        pprint(f"Critical radius (m): {critical_radius}", "Info")

        return critical_radius

    @staticmethod
    def compute_critical_driving_force(driving_force, surface_energy):
        assert isinstance(
            driving_force, jax.Array
        ), "The driving force must be a JAX array"
        assert isinstance(
            surface_energy, jax.Array
        ), "The surface energy must be a JAX array"
        assert jnp.all(
            surface_energy > 0.0
        ), "All entries in the surface energy vector must be greater than 0.0"
        assert jnp.size(driving_force) == jnp.size(
            surface_energy
        ), "The vectors must be the same size"

        critical_driving_force = (
            16.0 * jnp.pi * surface_energy**3 / (3.0 * driving_force**2)
        )

        pprint(f"Critical driving force (J): {critical_driving_force}", "Info")

        return critical_driving_force

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

    @staticmethod
    def compute_nucleus_size(
        critical_radius, surface_energy, temperature, boltzmann_constant=1.380e-23
    ):
        return critical_radius + 0.5 * jnp.sqrt(
            boltzmann_constant * temperature / (jnp.pi * surface_energy)
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

    def compute_growth_rate(self, nuclei_rate):
        return self.diffusivity / self.mean_radius * self.compute_supersaturation(
            self.precipitate_composition,
            self.solution_composition,
            self.equilibrium_solution_composition,
            self.alpha_parameter,
        ) + 1.0 / self.n_precipitates * nuclei_rate * (
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
        # This might be getting a lot of round-off error
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
        nuclei_rate = self.compute_nuclei_rate()
        growth_rate = self.compute_growth_rate(nuclei_rate)
        self.solution_composition = self.update_solute_fraction()

        # Apply the runge-kutta method to update the mean radius
        timestep = 0.1

        self.n_precipitates = self.n_precipitates + nuclei_rate * timestep
        self.mean_radius = self.mean_radius + growth_rate * timestep

        self.time = self.time + timestep

        print(f"Mean radius: {self.mean_radius}")
        print(f"Precipitates per unit volume:: {self.n_precipitates}")
        print(f"Time: {self.time}")
        print(f"Solution composition: {self.solution_composition}")

        assert self.mean_radius == 3.89029844e-10, f"Mean radius is {self.mean_radius}"

        # Append to the evolution arrays
        self.time_evolution.append(self.time)
        self.mean_radius_evolution.append(self.mean_radius)
        self.volume_fraction_evolution.append(self.volume_fraction)
        self.precipitate_density_evolution.append(self.n_precipitates)
        self.solution_composition_evolution.append(self.solution_composition)

    def compute_mean_radius(self, composition, temperature, pressure):
        print("\n\n\nBeginning calulation of mean radius\n")
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
