import os
import pickle
import matplotlib.pyplot as plt
from pycalphad import Database, binplot, Workspace, equilibrium, calculate
from pycalphad.property_framework.metaproperties import IsolatedPhase
import pycalphad.variables as v
import numpy as np

# Make sure the outputs directory exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Load the database and choose what phases will be considered (in this case all)
db_al_ti = Database("TiAl.TDB")
phases = db_al_ti.phases.keys()

# Create figure
fig = plt.figure(figsize=(9, 6))
axes = fig.gca()

# Compute the phase diagram
binplot(
    db_al_ti,
    ["AL", "TI", "VA"],
    phases,
    {v.X("TI"): (0, 0.3, 0.005), v.T: (300, 2000, 10), v.P: 101325, v.N: 1},
    plot_kwargs={"ax": axes},
)

# Save plot
plt.tight_layout()
plt.savefig("outputs/phase_diagram.png", dpi=300)
plt.clf()

# Nominal composition (mol fraction)
composition = 0.00823
# Temperature range (K)
temperatures = np.arange(600, 1300, 5)
# Pressure (kPa)
pressure = 101325
# Base surface energy (J/m^2)
# Wearing, D.; Horsfield, A. P.; Xu, W.; Lee, P. D. Which Wets TiB2 Inoculant Particles:
# Al or Al3Ti? Journal of Alloys and Compounds 2016, 664, 460–468.
# https://doi.org/10.1016/j.jallcom.2015.12.203.
base_surface_energy = 0.701


# Density and molar mass from Materials Commons
rho_al3ti = 3.42 * 10**6  # g/m^3
molar_mass_al3ti = 128.812  # g/mol
V_m = molar_mass_al3ti / rho_al3ti  # m^3/mol


def plot_bulk_driving_forces(temperatures, driving_force):
    original_current_density = 13  # mA/cm^2
    current_density = original_current_density * 10  # A/m^2

    plt.plot(temperatures, driving_force, linewidth=3, label="No current")
    plt.plot(
        temperatures,
        compute_ECP_driving_force(driving_force, current_density),
        linewidth=3,
        label=rf"j={original_current_density} $mA/cm^2$",
    )

    original_current_density = 500  # mA/cm^2
    current_density = original_current_density * 10  # A/m^2
    plt.plot(
        temperatures,
        compute_ECP_driving_force(driving_force, current_density),
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


def compute_ECP_driving_force(base_driving_force, current_density):
    # Current density should be in units A/m^2

    # Conductivities at 850C
    sigma_al3ti = 1.0 / (200 * 10**-8)  # 1/(Ohm*m)
    sigma_liquid = 1.0 / (27 * 10**-8)  # 1/(Ohm*m)

    # Radius of the cylindrical cast. Dolinksy assumes a cylrindrical melt and
    # nuclei, which we obviously don't have. For now, I'll treat the experimental
    # conditions as if they were cylindrical. This difference in shape hopefully
    # shouldn't be too bad for the accuracy of the result.
    r_0 = 20 * 10**-3  # m

    # Permeability of free space
    mu_0 = 1.25663706 * 10**-6  # N/A^2

    # Caclulat the change in free energy from Dolinsky
    # Compute xi
    xi = (sigma_liquid - sigma_al3ti) / (2.0 * sigma_liquid - sigma_al3ti)
    # Compute p_m
    p_m = np.pi**2 * r_0**2 * current_density**2 * mu_0

    # Calculating the change in free energy from Dolinsky
    dG = -4 * p_m * xi * V_m  # J/mol

    print(f"Bulk work in creating new nuclei: {-dG} J/mol")

    # I assume that the change is free energy is equal for the entire temperature
    # range.
    driving_force = base_driving_force + dG * np.ones(np.shape(base_driving_force))

    return driving_force


def compute_net_driving_force(bulk_driving_force, surface_energy):
    # Make sure that bulk_driving_force has units of J/m^3 and surface_energy
    # has units of J/m^2.
    assert np.all(bulk_driving_force <= 0), "Bulk driving force should be all negative."
    assert np.all(surface_energy > 0), (
        "Surface energies should all be positive and nonzero"
    )

    # First we have to determine the critical free energy (that of the saddle point)
    # For simplicity, we'll start with the critical radius, assuming a perfectly
    # spherical particle.
    r_star = -2 * surface_energy / bulk_driving_force  # m
    G_star = (
        4 / 3 * np.pi * r_star**3 * bulk_driving_force
        + 4 * np.pi * r_star**2 * surface_energy
    )

    print(f"Critical radii (m): {r_star}")
    print(f"Critical energy (eV): {G_star * 6.242 * 10**18}")

    return G_star


def molar_to_volumetric_driving_force(molar_driving_force, molar_volume):
    # The molar_driving_force should have units of J/mol and the molar_volume should
    # have units of m^3/mol
    return molar_driving_force / molar_volume


def compute_nucleation_rate_estimate(free_energy, temperature):
    # The free_energy should have units of J and the temperature should have units
    # of K

    free_energy = free_energy * 6.242 * 10**18  # eV
    kb_boltzmann = 8.617333 * 10**-5  # eV/K

    print(f"Exponent value: {-free_energy / (kb_boltzmann * temperature)}")

    return np.exp(-free_energy / (kb_boltzmann * temperature))


def compute_driving_force(driving_force, temperature=1123):
    # Create a workspace to begin the calculation of free energy curves
    al3ti_workspace = Workspace(
        "TiAl.TDB",
        ["AL", "TI", "VA"],
        ["DO22_XAL3", "LIQUID"],
        {v.X("TI"): (0, 0.3, 0.005), v.T: temperature, v.P: pressure, v.N: 1},
    )
    fig = plt.figure()
    ax = fig.add_subplot()
    x = al3ti_workspace.get(v.X("TI"))
    ax.set_xlabel(f"{v.X('TI').display_name} [{v.X('TI').display_units}]")
    for phase_name in al3ti_workspace.phases:
        # Workaround for poor starting point selection in IsolatedPhase
        metastable_wks = al3ti_workspace.copy()
        metastable_wks.phases = [phase_name]
        prop = IsolatedPhase(phase_name, metastable_wks)(f"GM({phase_name})")
        prop.display_name = f"GM({phase_name})"
        ax.plot(x, al3ti_workspace.get(prop), label=prop.display_name)
    ax.axvline(composition, linestyle="--", color="black", label="Nominal composition")

    # Calculating the phase equilibria
    phases = ["DO22_XAL3", "LIQUID"]
    equilibria_result = equilibrium(
        db_al_ti,
        ["AL", "TI", "VA"],
        phases,
        {v.X("TI"): composition, v.T: temperature, v.P: pressure, v.N: 1},
    )

    equilibrium_free_energy = (
        equilibria_result.sel(P=pressure, T=temperature).GM[0][0].values
    )
    supersaturated_free_energy = None

    # Calculating the free energy curves
    calculate_result_al3ti = calculate(
        db_al_ti, ["AL", "TI", "VA"], "DO22_XAL3", P=pressure, T=temperature
    )
    calculate_result_liquid = calculate(
        db_al_ti, ["AL", "TI", "VA"], "LIQUID", P=pressure, T=temperature
    )

    # Plot the point for the equilbria compositions
    for equilbrium_point in (
        equilibria_result.sel(P=pressure, T=temperature).X[0][0].values
    ):
        if np.all(np.isnan(equilbrium_point)):
            continue

        # Calculate the distance from the equilbrium point to find the correspond Gibbs free energy
        compositions_al3ti = calculate_result_al3ti.X.values[0][0][0]
        compositions_liquid = calculate_result_liquid.X.values[0][0][0]

        def compute_free_energy_from_composition(composition):
            distances = np.linalg.norm(compositions_al3ti - composition, axis=1)
            closest_index = np.argmin(distances)
            free_energies_al3ti = calculate_result_al3ti.GM.values[0][0][0][
                closest_index
            ]
            distances = np.linalg.norm(compositions_liquid - composition, axis=1)
            closest_index = np.argmin(distances)
            free_energies_liquid = calculate_result_liquid.GM.values[0][0][0][
                closest_index
            ]

            # Take the minimum free energy (the one that lies on the common tangent line)
            free_energy = np.min([free_energies_al3ti, free_energies_liquid])

            return free_energy

        nominal_composition_point = np.array([1 - composition, composition])
        supersaturated_free_energy = compute_free_energy_from_composition(
            nominal_composition_point
        )

        # Plot
        plt.scatter(
            equilbrium_point[1],
            compute_free_energy_from_composition(equilbrium_point),
            color="black",
            label="Equilibrium Composition",
        )

    print(
        f"Supersatured G: {supersaturated_free_energy}\nEquilibrium G: {equilibrium_free_energy}\nDriving force: {supersaturated_free_energy - equilibrium_free_energy}"
    )
    driving_force.append(supersaturated_free_energy - equilibrium_free_energy)
    ax.legend()
    plt.savefig(
        f"outputs/temp_dependent_energy_{composition}_mol_frac/free_energy_{temperature}.png",
        dpi=300,
    )
    plt.close(fig)


dump_file_name = f"outputs/bulk_driving_force_{composition}_mol_frac.pkl"

if os.path.exists(dump_file_name):
    # Load the serialized data
    with open(dump_file_name, "rb") as f:
        bulk_driving_force = pickle.load(f)
    print("Loaded bulk_driving_force from dump file.")
else:
    bulk_driving_force = []

    # Create folder for temperature dependent free energy plots
    if not os.path.exists(f"outputs/temp_dependent_energy_{composition}_mol_frac"):
        os.makedirs(f"outputs/temp_dependent_energy_{composition}_mol_frac")

    for temperature in temperatures:
        compute_driving_force(bulk_driving_force, temperature)

    with open(dump_file_name, "wb") as f:
        pickle.dump(bulk_driving_force, f)
    print("Computed and saved bulk_driving_force to dump file.")

# Plot the bulk energy driving forces
plot_bulk_driving_forces(temperatures, bulk_driving_force)

# Now that we have the bulk drivings forces we can calculate the net driving force
# TODO the V_m should be temprature dependent and from pycalphad ideally
bulk_driving_force = molar_to_volumetric_driving_force(
    np.array(bulk_driving_force), V_m
)

mask = (bulk_driving_force >= 0) & (temperatures > 0)

bulk_driving_force = bulk_driving_force[mask]
temperatures = temperatures[mask]

print(f"Temperature (K): {temperatures}")

net_driving_force = compute_net_driving_force(-bulk_driving_force, base_surface_energy)
plt.clf()
plt.plot(temperatures, net_driving_force, "-o", label="No current")
plt.xlabel("Temperature [K]")
plt.ylabel("Driving Force [J]")
plt.savefig("outputs/net_driving_force.png", dpi=300)


estimate_nucleation_rate = compute_nucleation_rate_estimate(
    net_driving_force, temperatures
)
plt.clf()
plt.plot(temperatures, estimate_nucleation_rate, "-o", label="No current")
plt.xlabel("Temperature [K]")
plt.ylabel("Estimated nucleation rate (Arrhenius)")
plt.savefig("outputs/nucleation_rate.png", dpi=300)
