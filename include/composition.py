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

    def compute_equilibrium_phase_fractions(
        self, mole_fractions, equilibrium_mole_fractions
    ):
        assert isinstance(
            mole_fractions, np.ndarray
        ), "equilibrium_mole_fractions must be a NumPy array"
        assert isinstance(
            equilibrium_mole_fractions, np.ndarray
        ), "equilibrium_mole_fractions must be a 2D NumPy array"
        assert isinstance(
            equilibrium_mole_fractions[0], np.ndarray
        ), "equilibrium_mole_fractions must be a 2D NumPy array"
        assert len(mole_fractions) == len(
            equilibrium_mole_fractions
        ), "Mismatched lengths"

        # Apply the lever rule
        # First construct the nominal compositions
        b = mole_fractions

        # Then construct the equilibrium phase compositions
        print(f"Mole fractions: {mole_fractions}")

        # Add constraint that phase fractions sum to 1
        A = np.vstack(
            [equilibrium_mole_fractions.T, np.ones(equilibrium_mole_fractions.shape[1])]
        )
        b = np.append(mole_fractions, 1.0)
        print(f"A: {A}")
        print(f"b: {b}")

        # Solve the system using least squares
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        return x
