import numpy as np


class CompositionOperations:
    global_tolerance = 1e-6
    global_molar_masses = None

    def __init__(self, global_molar_masses=None):
        if global_molar_masses is not None:
            self.global_molar_masses = global_molar_masses

    def compute_mole_fractions(self, mass_fractions, molar_masses=None):
        if molar_masses is None:
            assert self.global_molar_masses is not None, (
                "Molar masses must be provided in the constructor if none are "
                "provided in the compute function"
            )
            molar_masses = self.global_molar_masses

        assert isinstance(
            mass_fractions, np.ndarray
        ), "mass_fractions must be a NumPy array"
        assert isinstance(
            molar_masses, np.ndarray
        ), "molar_masses must be a NumPy array"
        assert len(mass_fractions) == len(molar_masses), "Mismatched lengths"
        total = np.sum(mass_fractions)
        assert (
            abs(total - 1.0) < self.global_tolerance
        ), f"Mole fractions must sum to 1. Got {total}"

        moles = mass_fractions / molar_masses
        total_moles = np.sum(moles)
        return moles / total_moles

    def compute_mass_fractions(self, mole_fractions, molar_masses=None):
        if molar_masses is None:
            assert self.global_molar_masses is not None, (
                "Molar masses must be provided in the constructor if none are "
                "provided in the compute function"
            )
            molar_masses = self.global_molar_masses

        assert isinstance(
            mole_fractions, np.ndarray
        ), "mole_fractions must be a NumPy array"
        assert isinstance(
            molar_masses, np.ndarray
        ), "molar_masses must be a NumPy array"
        assert len(mole_fractions) == len(molar_masses), "Mismatched lengths"
        total = np.sum(mole_fractions)
        assert (
            abs(total - 1.0) < self.global_tolerance
        ), f"Mole fractions must sum to 1. Got {total}"

        masses = mole_fractions * molar_masses
        total_mass = np.sum(masses)
        return masses / total_mass
