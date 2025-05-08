import numpy as np
from multiprocessing import Pool, cpu_count
from include.file_io import FileIO
from include.timer import Timer
from include.composition import CompositionOperations
from include.calphad import Calphad

# Al-Ti system
AL_TI_SYSTEM = "TiAl.TDB"
MASS_FRACTION_TI = 0.0145
MASS_FRACTION_AL = 1.0 - MASS_FRACTION_TI
MOLAR_MASS_TI = 47.867  # g/mol
MOLAR_MASS_AL = 26.982  # g/mol


COMPOSITION_OPERATIONS = CompositionOperations()
[COMPOSITION, _] = COMPOSITION_OPERATIONS.compute_mole_fractions(
    mass_fractions=np.array([MASS_FRACTION_TI, MASS_FRACTION_AL]),
    molar_masses=np.array([MOLAR_MASS_TI, MOLAR_MASS_AL]),
)

# Global constants
TEMPERATURES = np.arange(600, 1300, 5)  # K
PRESSURE = 101325.0  # Pa
BASE_SURFACE_ENERGY = 0.701  # J/m^2
RHO_AL3TI = 3.42 * 10**6  # g/m^3
RHO_LIQUID = 2.30 * 10**6  # g/m^3
MOLAR_MASS_AL3TI = 128.812  # g/mol
MOLAR_MASS_LIQUID = 26.982  # g/mol
V_M_AL3TI = MOLAR_MASS_AL3TI / RHO_AL3TI  # m^3/mol
V_M_LIQUID = MOLAR_MASS_LIQUID / RHO_LIQUID  # m^3/mol

# Initialize calphad object
calphad = Calphad(AL_TI_SYSTEM)

# Compute phase diagram
calphad.compute_binary_phase_diagram(
    ["AL", "TI", "VA"],
    "TI",
    (0, 0.3, 0.005),
    (300, 2000, 10),
    PRESSURE,
    "outputs/",
)

# Calculate the the phase fraction of Al3Ti and liquid
TEMPERATURE = 850.0 + 273.0  # K
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

phase_fractions = np.array(
    [equilibirum_phase_fraction_al3ti, equilibirum_phase_fraction_liquid]
)
molar_volumes = np.array([V_M_AL3TI, V_M_LIQUID])
total_volume = np.sum(phase_fractions * molar_volumes)
volume_fractions = phase_fractions * molar_volumes / total_volume

print(f"Volume fractions: {volume_fractions}")


def compute_relative_nucleation_rate(
    current_density_1, current_density_2, average_radius_1, average_radius_2
):
    """
    Compute the relative nucleation between two processing conditions, given their
    current density (mA/cm^2) and average particle radius (um).
    """
    molar_frac_ti_al3ti = 0.25

    def compute_particle_count(
        average_radius, molar_volume, solute_mass, total_solute_mass
    ):
        return molar_volume * (solute_mass - total_solute_mass) / average_radius**3

    saturation_particle_count_1 = compute_particle_count(
        average_radius_1 * 10**-6, V_M, molar_frac_ti_al3ti, COMPOSITION
    )
    saturation_particle_count_2 = compute_particle_count(
        average_radius_1 * 10**-6, V_M, molar_frac_ti_al3ti, COMPOSITION
    )

    print(saturation_particle_count_1, saturation_particle_count_2)

    return 0


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
    """
    distances_al3ti = np.linalg.norm(compositions_al3ti - composition, axis=1)
    closest_index_al3ti = np.argmin(distances_al3ti)
    free_energy_al3ti = free_energies_al3ti[closest_index_al3ti]

    distances_liquid = np.linalg.norm(compositions_liquid - composition, axis=1)
    closest_index_liquid = np.argmin(distances_liquid)
    free_energy_liquid = free_energies_liquid[closest_index_liquid]

    return min(free_energy_al3ti, free_energy_liquid)


# def compute_ECP_driving_force(base_driving_force, current_density):
#     """
#     Compute the driving force under electric current conditions.

#     Args:
#         base_driving_force: Base driving force without current (J/mol)
#         current_density: Current density (A/m^2)

#     Returns:
#         Modified driving force accounting for electric current effects (J/mol)
#     """
#     # Conductivities at 850C
#     sigma_al3ti = 1.0 / (200 * 10**-8)  # 1/(Ohm*m)
#     sigma_liquid = 1.0 / (27 * 10**-8)  # 1/(Ohm*m)

#     # Radius of the cylindrical cast
#     r_0 = 20 * 10**-3  # m

#     # Permeability of free space
#     mu_0 = 1.25663706 * 10**-6  # N/A^2

#     # Compute parameters for Dolinsky's model
#     xi = (sigma_liquid - sigma_al3ti) / (2.0 * sigma_liquid - sigma_al3ti)
#     p_m = np.pi**2 * r_0**2 * current_density**2 * mu_0
#     dG = -4 * p_m * xi * V_M  # J/mol

#     print(
#         f"Bulk work in creating new nuclei with current density {current_density} A/m^2: {-dG} J/mol"
#     )

#     return base_driving_force + dG * np.ones(np.shape(base_driving_force))


# def compute_driving_force_for_temperature(temperature):
#     """
#     Compute the driving force for a single temperature.

#     Args:
#         temperature: Temperature in Kelvin

#     Returns:
#         Driving force value for the given temperature
#     """
#     driving_force = []
#     compute_driving_force(driving_force, temperature)
#     return driving_force[0]


# def compute_driving_force(driving_force, temperature=1123):
#     """
#     Compute the driving force for phase transformation at a given temperature.

#     Args:
#         driving_force: List to store the computed driving force
#         temperature: Temperature in Kelvin
#     """
#     # Create workspace for calculations
#     al3ti_workspace = Workspace(
#         "TiAl.TDB",
#         ["AL", "TI", "VA"],
#         ["DO22_XAL3", "LIQUID"],
#         {v.X("TI"): (0, 0.3, 0.005), v.T: temperature, v.P: PRESSURE, v.N: 1},
#     )

#     # Set up figure for visualization
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     x = al3ti_workspace.get(v.X("TI"))
#     ax.set_xlabel(f"{v.X('TI').display_name} [{v.X('TI').display_units}]")

#     # Plot phase free energies
#     for phase_name in al3ti_workspace.phases:
#         metastable_wks = al3ti_workspace.copy()
#         metastable_wks.phases = [phase_name]
#         prop = IsolatedPhase(phase_name, metastable_wks)(f"GM({phase_name})")
#         prop.display_name = f"GM({phase_name})"
#         ax.plot(x, al3ti_workspace.get(prop), label=prop.display_name)

#     ax.axvline(COMPOSITION, linestyle="--", color="black", label="Nominal composition")

#     # Calculate phase equilibria
#     phases = ["DO22_XAL3", "LIQUID"]
#     equilibria_result = equilibrium(
#         db_al_ti,
#         ["AL", "TI", "VA"],
#         phases,
#         {v.X("TI"): COMPOSITION, v.T: temperature, v.P: PRESSURE, v.N: 1},
#     )

#     equilibrium_free_energy = (
#         equilibria_result.sel(P=PRESSURE, T=temperature).GM[0][0].values
#     )

#     # Calculate free energy curves
#     calculate_result_al3ti = calculate(
#         db_al_ti, ["AL", "TI", "VA"], "DO22_XAL3", P=PRESSURE, T=temperature
#     )
#     calculate_result_liquid = calculate(
#         db_al_ti, ["AL", "TI", "VA"], "LIQUID", P=PRESSURE, T=temperature
#     )

#     # Process equilibrium points
#     for equilbrium_point in (
#         equilibria_result.sel(P=PRESSURE, T=temperature).X[0][0].values
#     ):
#         if np.all(np.isnan(equilbrium_point)):
#             continue

#         compositions_al3ti = calculate_result_al3ti.X.values[0][0][0]
#         compositions_liquid = calculate_result_liquid.X.values[0][0][0]
#         free_energies_al3ti = calculate_result_al3ti.GM.values[0][0][0]
#         free_energies_liquid = calculate_result_liquid.GM.values[0][0][0]

#         nominal_composition_point = np.array([1 - COMPOSITION, COMPOSITION])
#         supersaturated_free_energy = compute_free_energy_from_composition(
#             compositions_al3ti,
#             compositions_liquid,
#             free_energies_al3ti,
#             free_energies_liquid,
#             nominal_composition_point,
#         )

#         # Plot equilibrium point
#         plt.scatter(
#             equilbrium_point[1],
#             compute_free_energy_from_composition(
#                 compositions_al3ti,
#                 compositions_liquid,
#                 free_energies_al3ti,
#                 free_energies_liquid,
#                 equilbrium_point,
#             ),
#             color="black",
#             label="Equilibrium Composition",
#         )

#     print(
#         f"Supersatured G: {supersaturated_free_energy}\n"
#         f"Equilibrium G: {equilibrium_free_energy}\n"
#         f"Driving force: {supersaturated_free_energy - equilibrium_free_energy}"
#     )

#     driving_force.append(supersaturated_free_energy - equilibrium_free_energy)
#     ax.legend()
#     plt.savefig(
#         f"outputs/temp_dependent_energy_{COMPOSITION}_mol_frac/free_energy_{temperature}.png",
#         dpi=300,
#     )
#     plt.close(fig)


# def plot_bulk_driving_forces(temperatures, driving_force):
#     """
#     Plot the bulk driving forces with and without electric current.

#     Args:
#         temperatures: Array of temperatures in Kelvin
#         driving_force: Array of driving forces in J/mol
#     """
#     original_current_density = 13  # mA/cm^2
#     current_density = original_current_density * 10  # A/m^2

#     plt.plot(temperatures, driving_force, linewidth=3, label="No current")
#     plt.plot(
#         temperatures,
#         compute_ECP_driving_force(driving_force, current_density),
#         linewidth=3,
#         label=rf"j={original_current_density} $mA/cm^2$",
#     )

#     original_current_density = 500  # mA/cm^2
#     current_density = original_current_density * 10  # A/m^2
#     plt.plot(
#         temperatures,
#         compute_ECP_driving_force(driving_force, current_density),
#         linewidth=3,
#         label=rf"j={original_current_density} $mA/cm^2$",
#     )
#     plt.axvline(1123, linestyle="--", color="black", label=r"850$\degree$C")
#     plt.axhline(y=0, color="black", linestyle="--")
#     plt.xlabel("Temperature [K]", fontsize=16)
#     plt.ylabel("Bulk Driving Force [J/mol]", fontsize=16)
#     plt.legend(fontsize=14)
#     plt.tight_layout()
#     plt.savefig("outputs/bulk_driving_force.png", dpi=300)
#     plt.close()


# def compute_net_driving_force(bulk_driving_force, surface_energy):
#     """
#     Compute the net driving force for nucleation.

#     Args:
#         bulk_driving_force: Bulk driving force in J/m^3
#         surface_energy: Surface energy in J/m^2

#     Returns:
#         Net driving force in J
#     """
#     assert np.all(bulk_driving_force <= 0), "Bulk driving force should be all negative."
#     assert np.all(
#         surface_energy > 0
#     ), "Surface energies should all be positive and nonzero"

#     # Calculate critical radius and energy
#     r_star = -2 * surface_energy / bulk_driving_force  # m
#     G_star = (
#         4 / 3 * np.pi * r_star**3 * bulk_driving_force
#         + 4 * np.pi * r_star**2 * surface_energy
#     )
#     return G_star


# def molar_to_volumetric_driving_force(molar_driving_force, molar_volume):
#     """
#     Convert molar driving force to volumetric driving force.

#     Args:
#         molar_driving_force: Driving force in J/mol
#         molar_volume: Molar volume in m^3/mol

#     Returns:
#         Volumetric driving force in J/m^3
#     """
#     return molar_driving_force / molar_volume


# def compute_nucleation_rate_estimate(free_energy, temperature):
#     """
#     Estimate nucleation rate using Arrhenius equation.

#     Args:
#         free_energy: Free energy barrier in J
#         temperature: Temperature in K

#     Returns:
#         Estimated nucleation rate
#     """
#     free_energy = free_energy * 6.242 * 10**18  # Convert to eV
#     kb_boltzmann = 8.617333 * 10**-5  # eV/K
#     return np.exp(-free_energy / (kb_boltzmann * temperature))


# def main():
#     """Main function to run the nucleation rate calculations."""
#     global db_al_ti

#     # Setup various objects
#     file_io = FileIO()
#     timer = Timer()

#     # Load database and compute phase diagram
#     timer.begin("Load database")
#     db_al_ti, phases = load_database()
#     timer.end("Load database")

#     timer.begin("Compute phase diagram")
#     compute_phase_diagram(db_al_ti, phases)
#     timer.end("Compute phase diagram")

#     # Compute driving forces
#     dump_file_name = f"outputs/bulk_driving_force_{COMPOSITION}_mol_frac.pkl"
#     try:
#         bulk_driving_force = file_io.load_pickle_dump(dump_file_name)
#         print("Loaded bulk_driving_force from dump file.")
#     except FileNotFoundError:
#         file_io.create_directory(
#             f"outputs/temp_dependent_energy_{COMPOSITION}_mol_frac"
#         )

#         num_processes = cpu_count()
#         print(f"Using {num_processes} processes for parallel computation")

#         with Pool(processes=num_processes) as pool:
#             bulk_driving_force = pool.map(
#                 compute_driving_force_for_temperature, TEMPERATURES
#             )

#         file_io.create_pickle_dump(bulk_driving_force, dump_file_name)
#         print("Computed and saved bulk_driving_force to dump file.")

#     # Plot results
#     plot_bulk_driving_forces(TEMPERATURES, bulk_driving_force)

#     bulk_driving_force = molar_to_volumetric_driving_force(
#         np.array(bulk_driving_force), V_M
#     )

#     mask = (bulk_driving_force >= 0) & (TEMPERATURES > 0)
#     bulk_driving_force = bulk_driving_force[mask]
#     temperatures = TEMPERATURES[mask]

#     net_driving_force = compute_net_driving_force(
#         -bulk_driving_force, BASE_SURFACE_ENERGY
#     )
#     plt.clf()
#     plt.plot(temperatures, net_driving_force, "-o", label="No current")
#     plt.xlabel("Temperature [K]")
#     plt.ylabel("Driving Force [J]")
#     plt.savefig("outputs/net_driving_force.png", dpi=300)
#     plt.close()

#     estimate_nucleation_rate = compute_nucleation_rate_estimate(
#         net_driving_force, temperatures
#     )
#     plt.clf()
#     plt.plot(temperatures, estimate_nucleation_rate, "-o", label="No current")
#     plt.xlabel("Temperature [K]")
#     plt.ylabel("Estimated nucleation rate (Arrhenius)")
#     plt.savefig("outputs/nucleation_rate.png", dpi=300)
#     plt.close()

#     compute_relative_nucleation_rate(0, 10, 2.6, 1.8)

#     # Print timing results
#     timer.print_summary()


# if __name__ == "__main__":
#     main()
