#!/usr/bin/env python3

from os import wait
import numpy as np
from multiprocessing import Pool, cpu_count
from include.file_io import FileIO
from include.timer import Timer
from include.composition import CompositionOperations
from include.calphad import Calphad
from include.nucleation import Nucleation
from include.mean_radius import MeanRadius
from include.skin_effect import SkinEffect
import matplotlib.pyplot as plt

# Al-Ti system
AL_TI_SYSTEM = "TiAl.TDB"
MASS_FRACTION_TI = 0.0145
MASS_FRACTION_AL = 1.0 - MASS_FRACTION_TI
MOLAR_MASS_TI = 47.867  # g/mol
MOLAR_MASS_AL = 26.982  # g/mol

# Global constants
TEMPERATURES = np.arange(300, 1300, 5)  # K
TEMPERATURE = 850.0 + 273.0  # K
PRESSURE = 101325.0  # Pa
BASE_SURFACE_ENERGY = 0.170  # J/m^2
RHO_AL3TI = 3.42 * 10**6  # g/m^3
RHO_LIQUID = 2.30 * 10**6  # g/m^3
MOLAR_MASS_AL3TI = 128.812  # g/mol
MOLAR_MASS_LIQUID = 26.982  # g/mol
V_M_AL3TI = MOLAR_MASS_AL3TI / RHO_AL3TI  # m^3/mol
V_M_LIQUID = MOLAR_MASS_LIQUID / RHO_LIQUID  # m^3/mol

# Determine the mole fraction of Ti and Al
[COMPOSITION, _] = CompositionOperations().compute_mole_fractions(
    mass_fractions=np.array([MASS_FRACTION_TI, MASS_FRACTION_AL]),
    molar_masses=np.array([MOLAR_MASS_TI, MOLAR_MASS_AL]),
)


def compute_driving_force_for_temp(args):
    calphad, composition, pressure, temp = args
    return calphad.compute_driving_force_for_temperature(composition, temp, pressure)


def main():
    # Setup various objects
    file_io = FileIO()
    timer = Timer()

    # Create output directory
    file_io.create_output_directory()

    # Initialize calphad object
    timer.begin("Initialize calphad object")
    calphad = Calphad(AL_TI_SYSTEM)
    timer.end("Initialize calphad object")

    # Compute phase diagram
    timer.begin("Compute phase diagram")
    # calphad.compute_binary_phase_diagram(
    #     ["AL", "TI", "VA"],
    #     "TI",
    #     (0, 0.1, 0.001),
    #     (300, 2000, 10),
    #     PRESSURE,
    #     "outputs/",
    #     sample_composition=COMPOSITION,
    #     sample_temperature=TEMPERATURE,
    # )
    timer.end("Compute phase diagram")

    # Calculate the the phase fraction of Al3Ti and liquid
    timer.begin("Compute phase equilibrium")
    phase_equilibrium = calphad.compute_binary_phase_equilibrium(
        ["AL", "TI", "VA"],
        ["DO22_XAL3", "LIQUID"],
        "TI",
        COMPOSITION,
        TEMPERATURE,
        PRESSURE,
    )
    equilibirum_phase_fraction_al3ti = calphad.find_equilibrium_phase_fraction(
        phase_equilibrium, "DO22_XAL3"
    )
    equilibirum_phase_fraction_liquid = calphad.find_equilibrium_phase_fraction(
        phase_equilibrium, "LIQUID"
    )
    print(f"Equilibirum phase fraction of Al3Ti: {equilibirum_phase_fraction_al3ti}")
    print(f"Equilibirum phase fraction of liquid: {equilibirum_phase_fraction_liquid}")
    timer.end("Compute phase equilibrium")

    phase_fractions = np.array(
        [equilibirum_phase_fraction_al3ti, equilibirum_phase_fraction_liquid]
    )
    molar_volumes = np.array([V_M_AL3TI, V_M_LIQUID])
    volume_fractions = CompositionOperations().compute_volume_fractions(
        phase_fractions, molar_volumes
    )
    print(f"Volume fractions: {volume_fractions}")

    # Compute driving forces
    dump_file_name = f"outputs/bulk_driving_force_{COMPOSITION}_mol_frac.pkl"
    try:
        bulk_driving_force = file_io.load_pickle_dump(dump_file_name)
        print("Loaded bulk_driving_force from dump file.")
    except FileNotFoundError:
        file_io.create_directory(
            f"outputs/temp_dependent_energy_{COMPOSITION}_mol_frac"
        )
        num_processes = cpu_count()
        print(f"Using {num_processes} processes for parallel computation")
        with Pool(processes=num_processes) as pool:
            args = [(calphad, COMPOSITION, PRESSURE, temp) for temp in TEMPERATURES]
            bulk_driving_force = pool.map(compute_driving_force_for_temp, args)
        file_io.create_pickle_dump(bulk_driving_force, dump_file_name)
        print("Computed and saved bulk_driving_force to dump file.")
    # Plot results
    calphad.plot_bulk_driving_forces(TEMPERATURES, bulk_driving_force, V_M_AL3TI)
    bulk_driving_force = calphad.molar_to_volumetric_driving_force(
        np.array(bulk_driving_force), V_M_AL3TI
    )

    # Compute the precipitate size distribution from the mean radius approach
    timer.begin("Precipitate size distribution")
    mean_radius = MeanRadius()
    mean_radius.compute_mean_radius(COMPOSITION, TEMPERATURE, PRESSURE)
    timer.end("Precipitate size distribution")

    # Compute the skin skin effect
    skin_effect = SkinEffect("cylinder")

    # Print timing results
    timer.print_summary()


if __name__ == "__main__":
    main()
