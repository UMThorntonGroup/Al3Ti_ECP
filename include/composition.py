import numpy as np


class CompositionOperations:
    global_tolerance = 1e-6
    global_molar_masses = None

    def __init__(self, global_molar_masses=None) -> None:
        if global_molar_masses is not None:
            CompositionOperations.global_molar_masses = global_molar_masses

    @staticmethod
    def compute_mole_fractions(fractions, molar_masses=None):
        if molar_masses is None:
            assert (
                CompositionOperations.global_molar_masses is not None
            ), "Molar masses must be provided in the constructor if none are provided in the compute function"
            molar_masses = CompositionOperations.global_molar_masses

        assert isinstance(fractions, np.ndarray), "fractions must be a NumPy array"
        assert isinstance(
            molar_masses, np.ndarray
        ), "molar_masses must be a NumPy array"
        assert len(fractions) == len(molar_masses), "Mismatched lengths"
        total = np.sum(fractions)
        assert (
            abs(total - 1.0) < CompositionOperations.global_tolerance
        ), f"Mole fractions must sum to 1. Got {total}"

        moles = fractions / molar_masses
        total_moles = np.sum(moles)
        return moles / total_moles

    @staticmethod
    def compute_mass_fractions(fractions, molar_masses=None):
        if molar_masses is None:
            assert (
                CompositionOperations.global_molar_masses is not None
            ), "Molar masses must be provided in the constructor if none are provided in the compute function"
            molar_masses = CompositionOperations.global_molar_masses

        assert isinstance(fractions, np.ndarray), "fractions must be a NumPy array"
        assert isinstance(
            molar_masses, np.ndarray
        ), "molar_masses must be a NumPy array"
        assert len(fractions) == len(molar_masses), "Mismatched lengths"
        total = np.sum(fractions)
        assert (
            abs(total - 1.0) < CompositionOperations.global_tolerance
        ), f"Mole fractions must sum to 1. Got {total}"

        masses = fractions * molar_masses
        total_mass = np.sum(masses)
        return masses / total_mass
