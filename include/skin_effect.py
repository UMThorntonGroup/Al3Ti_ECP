from numpy import sqrt
from numpy import pi
from pint import UnitRegistry
import matplotlib.pyplot as plt

# Define the unit registry
ureg = UnitRegistry()
# Add unit alias for siemen, henry, and farad
ureg.define("siemen = kilogram**-1 * meter**-2 * second**3 * ampere**2")
ureg.define("henry = kilogram * meter**2 * second**-2 * ampere**-2")
ureg.define("farad = kilogram**-1 * meter**-2 * second**4 * ampere**2")


class SkinEffect:
    def __init__(self, shape):
        assert isinstance(shape, str), "The shape must be a string"
        self.shape = shape

        # Constants
        self.permeability_of_free_space = 4.0e-7 * pi * ureg.henry / ureg.meter
        self.permittivity_of_free_space = 8.8541878188e-12 * ureg.farad / ureg.meter

    def compute_skin_depth(
        self, resitivity, frequency, relative_permeability, relative_permittivity
    ):
        assert self.shape == "cylinder", "The shape must be a cylinder"
        resistivity = 1.0 * ureg.meter / ureg.siemen
        angular_frequency = 1.0 * ureg.second**-1
        permeability = 1.0 * ureg.henry / ureg.meter
        permittivity = 1.0 * ureg.farad / ureg.meter

        return (
            sqrt(
                2.0
                * resistivity
                / (angular_frequency * permeability)
                * (
                    sqrt(1.0 + (resistivity * angular_frequency * permittivity) ** 2)
                    + resistivity * angular_frequency * permittivity
                )
            )
            .to("m")
            .to_compact()
        )

    def plot_current_density(self):
        assert self.shape == "cylinder", "The shape must be a cylinder"
