from pint import UnitRegistry

# Define the unit registry
ureg = UnitRegistry()
# Add unit alias for siemen, henry, and farad
ureg.define("siemen = kilogram**-1 * meter**-2 * second**3 * ampere**2")
ureg.define("henry = kilogram * meter**2 * second**-2 * ampere**-2")
ureg.define("farad = kilogram**-1 * meter**-2 * second**4 * ampere**2")
