import math
import numpy as np


class Nucleation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_critical_nucleus_radius(surface_energy, bulk_energy, shape="sphere"):
        assert isinstance(shape, str), "The nuclei shape must be a string"
        assert isinstance(surface_energy, float), "The surface_energy must be a float"
        assert isinstance(
            bulk_energy, np.ndarray
        ), "The bulk_energy must be a np.ndarray"
        assert shape == "sphere", "We only support spherical nuclei"

        return 2.0 * surface_energy / bulk_energy

    @staticmethod
    def compute_critical_nucleation_barrier(
        surface_energy, bulk_energy, shape="sphere"
    ):
        assert isinstance(shape, str), "The nuclei shape must be a string"
        assert isinstance(surface_energy, float), "The surface_energy must be a float"
        assert isinstance(
            bulk_energy, np.ndarray
        ), "The bulk_energy must be a np.ndarray"
        assert shape == "sphere", "We only support spherical nuclei"

        return 16.0 * math.pi / 3.0 * surface_energy**3 / bulk_energy**2

    @staticmethod
    def compute_relative_nucleation_rate(
        surface_energy_1,
        surface_energy_2,
        bulk_energy,
        temperature,
        use_taylor_expansion=False,
        shape="sphere",
    ):
        assert isinstance(shape, str), "The nuclei shape must be a string"
        assert isinstance(
            surface_energy_1, float
        ), "The surface_energy_1 must be a float"
        assert isinstance(
            surface_energy_2, float
        ), "The surface_energy_2 must be a float"
        assert isinstance(
            temperature, np.ndarray
        ), "The temperature must be a np.ndarray"
        assert isinstance(
            use_taylor_expansion, bool
        ), "use_taylor_expansion must be a bool"
        assert isinstance(
            bulk_energy, np.ndarray
        ), "The bulk_energy must be a np.ndarray"
        assert shape == "sphere", "We only support spherical nuclei"

        A = surface_energy_2 / surface_energy_1
        boltzmann_constant = 1.380 * 10**-23  # J/K

        if use_taylor_expansion:
            return (
                -16.0
                * math.pi
                * surface_energy_1**3
                / (boltzmann_constant * temperature * bulk_energy**2)
                * A**2
                * (A - 1.0)
            )
        else:
            return math.exp(
                16.0
                * math.pi
                * surface_energy_1**3
                / (3.0 * boltzmann_constant * temperature * bulk_energy**2)
                * (1.0 - A**3)
            )
