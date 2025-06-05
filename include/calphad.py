from pycalphad import Database, binplot, Workspace, equilibrium, calculate
from pycalphad.property_framework.metaproperties import IsolatedPhase
import pycalphad.variables as v
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from include.file_io import FileIO


class Calphad:
    database = None
    phases = None

    def __init__(self, database_name):
        assert isinstance(database_name, str), "The database_name must be a string"
        self.database = Database(database_name)
        self.phases = list(self.database.phases.keys())

    def get_phases(self):
        return self.phases

    def get_database(self):
        return self.database

    def compute_binary_phase_diagram(
        self,
        components,
        composition_variable,
        composition_range,
        temperature_range,
        pressure,
        output_file_path,
        filename="phase_diagram",
        sample_composition=None,
        sample_temperature=None,
    ):
        """Compute and save the binary phase diagram."""
        assert isinstance(components, list), "The components must be a list"
        assert isinstance(
            composition_variable, str
        ), "The composition_variable must be a string"
        assert isinstance(
            composition_range, tuple
        ), "The composition_range must be a tuple"
        assert isinstance(
            temperature_range, tuple
        ), "The temperature_range must be a tuple"
        assert isinstance(pressure, float), "The pressure must be a float"

        fig = plt.figure(figsize=(9, 6))
        axes = fig.gca()

        binplot(
            self.database,
            components,
            self.phases,
            {
                v.X(composition_variable): composition_range,
                v.T: temperature_range,
                v.P: pressure,
                v.N: 1,
            },
            plot_kwargs={"ax": axes},
        )
        if sample_composition is not None:
            plt.axvline(
                x=sample_composition,
                color="black",
                linestyle="--",
                label=f"{sample_composition * 100:.4f}%",
            )
            plt.legend(fontsize=14)
        if sample_temperature is not None:
            plt.axhline(
                y=sample_temperature,
                color="black",
                linestyle="--",
                label=f"{sample_temperature}",
            )
            plt.legend(fontsize=14)

        plt.xlabel(
            f"{v.X(composition_variable).display_name} [{v.X(composition_variable).display_units}]",
            fontsize=14,
        )
        plt.ylabel("Temperature [K]", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file_path + filename + ".png", dpi=300)
        plt.close(fig)

    def compute_binary_phase_equilibrium(
        self,
        components,
        phases,
        composition_variable,
        composition,
        temperature,
        pressure,
    ):
        assert isinstance(components, list), "The components must be a list"
        assert isinstance(phases, list), "The phases must be a list"
        assert isinstance(
            composition_variable, str
        ), "The composition_variable must be a string"
        assert isinstance(composition, float), "The composition must be a float"
        assert isinstance(temperature, float), "The temperature must be a float"
        assert isinstance(pressure, float), "The pressure must be a float"

        equilibria_result = equilibrium(
            self.database,
            components,
            phases,
            {
                v.X(composition_variable): composition,
                v.T: temperature,
                v.P: pressure,
                v.N: 1,
            },
        )

        return equilibria_result

    def find_equilibrium_phase_fraction(self, equilibria_result, phase):
        assert isinstance(
            equilibria_result, xr.Dataset
        ), "The equilibria_result must be a xarray.Dataset"
        assert isinstance(phase, str), "The phase must be a string"

        phase_fractions = equilibria_result.NP.where(
            equilibria_result.Phase == phase
        ).values
        # Get the first non-NaN value
        non_nan_values = phase_fractions[~np.isnan(phase_fractions)]
        return float(non_nan_values[0]) if len(non_nan_values) > 0 else 0.0

    @staticmethod
    def compute_free_energy_from_composition(
        compositions_al3ti,
        compositions_liquid,
        free_energies_al3ti,
        free_energies_liquid,
        composition,
    ):
        """
        Compute the free energy for a given composition by finding the closest points
        in the phase diagrams and taking the minimum free energy.

        Parameters
        ----------
        compositions_al3ti : np.ndarray
            Array of compositions for the Al3Ti phase
        compositions_liquid : np.ndarray
            Array of compositions for the liquid phase
        free_energies_al3ti : np.ndarray
            Array of free energies for the Al3Ti phase
        free_energies_liquid : np.ndarray
            Array of free energies for the liquid phase
        composition : np.ndarray
            Composition point to evaluate
        """
        distances_al3ti = np.linalg.norm(compositions_al3ti - composition, axis=1)
        closest_index_al3ti = np.argmin(distances_al3ti)
        free_energy_al3ti = free_energies_al3ti[closest_index_al3ti]

        distances_liquid = np.linalg.norm(compositions_liquid - composition, axis=1)
        closest_index_liquid = np.argmin(distances_liquid)
        free_energy_liquid = free_energies_liquid[closest_index_liquid]

        return min(free_energy_al3ti, free_energy_liquid)

    @staticmethod
    def compute_ECP_driving_force(base_driving_force, current_density, V_M):
        """
        Compute the driving force under electric current conditions.
        Args:
            base_driving_force: Base driving force without current (J/mol)
            current_density: Current density (A/m^2)
        Returns:
            Modified driving force accounting for electric current effects (J/mol)
        """
        # Conductivities at 850C
        sigma_al3ti = 1.0 / (200 * 10**-8)  # 1/(Ohm*m)
        sigma_liquid = 1.0 / (27 * 10**-8)  # 1/(Ohm*m)
        # Radius of the cylindrical cast
        r_0 = 20 * 10**-3  # m
        # Permeability of free space
        mu_0 = 1.25663706 * 10**-6  # N/A^2
        # Compute parameters for Dolinsky's model
        xi = (sigma_liquid - sigma_al3ti) / (2.0 * sigma_liquid - sigma_al3ti)
        p_m = np.pi**2 * r_0**2 * current_density**2 * mu_0
        dG = -4 * p_m * xi * V_M  # J/mol
        return base_driving_force + dG * np.ones(np.shape(base_driving_force))

    @staticmethod
    def molar_to_volumetric_driving_force(molar_driving_force, molar_volume):
        return molar_driving_force / molar_volume

    def compute_driving_force_for_temperature(
        self, composition, temperature, pressure=101325.0
    ):

        driving_force = []
        self.compute_driving_force(driving_force, composition, temperature, pressure)
        return driving_force[0]

    def compute_driving_force(self, driving_force, composition, temperature, pressure):

        phases = ["DO22_XAL3", "LIQUID"]
        max_ti_composition = 0.3

        # Create workspace for calculations
        al3ti_workspace = Workspace(
            self.database,
            ["AL", "TI", "VA"],
            phases,
            {
                v.X("TI"): (0, max_ti_composition, 0.005),
                v.T: temperature,
                v.P: pressure,
                v.N: 1,
            },
        )
        # Set up figure for visualization
        fig = plt.figure()
        ax = fig.add_subplot()
        x = al3ti_workspace.get(v.X("TI"))
        ax.set_xlabel(f"{v.X('TI').display_name} [{v.X('TI').display_units}]")
        # Plot phase free energies
        for phase_name in al3ti_workspace.phases:
            metastable_wks = al3ti_workspace.copy()
            metastable_wks.phases = [phase_name]
            prop = IsolatedPhase(phase_name, metastable_wks)(f"GM({phase_name})")
            prop.display_name = f"GM({phase_name})"
            ax.plot(x, al3ti_workspace.get(prop), label=prop.display_name)
        ax.axvline(
            composition, linestyle="--", color="black", label="Nominal composition"
        )

        # Calculate phase equilibria
        equilibria_result = equilibrium(
            self.database,
            ["AL", "TI", "VA"],
            phases,
            {v.X("TI"): composition, v.T: temperature, v.P: pressure, v.N: 1},
        )

        # Calculate free energy curves
        calculate_result_al3ti = calculate(
            self.database, ["AL", "TI", "VA"], "DO22_XAL3", P=pressure, T=temperature
        )
        calculate_result_liquid = calculate(
            self.database, ["AL", "TI", "VA"], "LIQUID", P=pressure, T=temperature
        )

        # Grab the compositions and free energies
        compositions_al3ti = calculate_result_al3ti.X.values[0][0][0]
        compositions_liquid = calculate_result_liquid.X.values[0][0][0]
        free_energies_al3ti = calculate_result_al3ti.GM.values[0][0][0]
        free_energies_liquid = calculate_result_liquid.GM.values[0][0][0]

        # Restrict the compositions to the range above
        mask = compositions_al3ti[:, 1] <= max_ti_composition
        compositions_al3ti = compositions_al3ti[mask]
        free_energies_al3ti = free_energies_al3ti[mask]
        mask = compositions_liquid[:, 1] <= max_ti_composition
        compositions_liquid = compositions_liquid[mask]
        free_energies_liquid = free_energies_liquid[mask]

        # Plot the equilibrium points
        for equilbrium_point in (
            equilibria_result.sel(P=pressure, T=temperature).X[0][0].values
        ):
            if np.all(np.isnan(equilbrium_point)):
                continue

            # Plot equilibrium point
            plt.scatter(
                equilbrium_point[1],
                self.compute_free_energy_from_composition(
                    compositions_al3ti,
                    compositions_liquid,
                    free_energies_al3ti,
                    free_energies_liquid,
                    equilbrium_point,
                ),
                color="black",
                label="Equilibrium Composition",
            )

        # Grab the compositions
        ti_composition = compositions_liquid[:, 1]
        nominal_ti_composition = composition
        distances_ti_composition = np.abs(ti_composition - nominal_ti_composition)
        assert np.shape(ti_composition) == np.shape(free_energies_liquid)

        # Find the closest and second closest points
        closest_index_liquid = np.argmin(distances_ti_composition)
        distances_ti_composition = np.delete(
            distances_ti_composition, closest_index_liquid
        )
        second_closest_index_liquid = np.argmin(distances_ti_composition)

        # Compute the tangent slope
        tangent_slope_liquid = (
            free_energies_liquid[second_closest_index_liquid]
            - free_energies_liquid[closest_index_liquid]
        ) / (
            ti_composition[second_closest_index_liquid]
            - ti_composition[closest_index_liquid]
        )
        assert np.all(tangent_slope_liquid < 0), "All tangent slopes should be negative"

        # Plot the tangent line at the nominal composition point
        ones_array = np.ones(np.shape(ti_composition))
        tangent_free_energy = free_energies_liquid[
            closest_index_liquid
        ] * ones_array + tangent_slope_liquid * ones_array * (
            ti_composition - ti_composition[closest_index_liquid]
        )
        plt.plot(
            ti_composition,
            tangent_free_energy,
            color="red",
            label="Tangent Line",
        )

        # Plot the point on the tangent line for the Al3Ti equilibrium point
        al3ti_equilibrium_point = (
            equilibria_result.sel(P=pressure, T=temperature).X[0][0].values[1]
        )
        distances_ti_composition = np.abs(ti_composition - al3ti_equilibrium_point[1])
        closest_index = np.argmin(distances_ti_composition)
        plt.scatter(
            al3ti_equilibrium_point[1],
            tangent_free_energy[closest_index],
            color="black",
            label="Al3Ti Equilibrium Point",
        )

        # Compute the driving force
        free_energy_supercooled_liquid = tangent_free_energy[closest_index]
        distances_ti_composition = np.abs(
            compositions_al3ti[:, 1] - al3ti_equilibrium_point[1]
        )
        closest_index = np.argmin(distances_ti_composition)
        free_energy_al3ti = free_energies_al3ti[closest_index]
        driving_force.append(free_energy_al3ti - free_energy_supercooled_liquid)

        # Create the output directory
        file_io = FileIO()
        file_io.create_directory(
            f"outputs/temp_dependent_energy_{composition}_mol_frac"
        )

        # Plot the legend and save the figure
        ax.legend()
        plt.savefig(
            f"outputs/temp_dependent_energy_{composition}_mol_frac/free_energy_{temperature}.png",
            dpi=300,
        )
        plt.close(fig)

    def plot_bulk_driving_forces(self, temperatures, driving_force, V_M):
        """
        Plot the bulk driving forces with and without electric current.

        Args:
            temperatures: Array of temperatures in Kelvin
            driving_force: Array of driving forces in J/mol
        """
        original_current_density = 13  # mA/cm^2
        current_density = original_current_density * 10  # A/m^2

        plt.plot(temperatures, driving_force, linewidth=3, label="No current")
        plt.plot(
            temperatures,
            self.compute_ECP_driving_force(driving_force, current_density, V_M),
            linewidth=3,
            label=rf"j={original_current_density} $mA/cm^2$",
        )

        original_current_density = 500  # mA/cm^2
        current_density = original_current_density * 10  # A/m^2
        plt.plot(
            temperatures,
            self.compute_ECP_driving_force(driving_force, current_density, V_M),
            linewidth=3,
            label=rf"j={original_current_density} $mA/cm^2$",
        )
        plt.axvline(1123, linestyle="--", color="black", label=r"850$\degree$C")
        plt.axhline(y=0, color="black", linestyle="--")
        plt.xlabel("Temperature [K]", fontsize=16)
        plt.ylabel("Bulk Driving Force [J/mol]", fontsize=16)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig("outputs/bulk_driving_force.png", dpi=300)
        plt.close()

    @staticmethod
    def compute_net_driving_force(bulk_driving_force, surface_energy):
        """
        Compute the net driving force for nucleation.
        Args:
            bulk_driving_force: Bulk driving force in J/m^3
            surface_energy: Surface energy in J/m^2
        Returns:
            Net driving force in J
        """
        assert np.all(
            bulk_driving_force <= 0
        ), "Bulk driving force should be all negative."
        assert np.all(
            surface_energy > 0
        ), "Surface energies should all be positive and nonzero"
        # Calculate critical radius and energy
        r_star = -2 * surface_energy / bulk_driving_force  # m
        G_star = (
            4 / 3 * np.pi * r_star**3 * bulk_driving_force
            + 4 * np.pi * r_star**2 * surface_energy
        )
        return G_star
