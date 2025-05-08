from pycalphad import Database, binplot, Workspace, equilibrium, calculate
from pycalphad.property_framework.metaproperties import IsolatedPhase
import pycalphad.variables as v
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


class Calphad:
    database = None
    phases = None

    def __init__(self, database_name):
        assert isinstance(database_name, str), "The database_name must be a string"
        self.database = Database(database_name)
        self.phases = self.database.phases.keys()

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
