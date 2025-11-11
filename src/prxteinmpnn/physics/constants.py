"""Physical constants for molecular simulations."""

from __future__ import annotations

# Coulomb constant for electrostatics
# Standard value for protein simulations in kcal/mol·Å·e⁻²
COULOMB_CONSTANT_KCAL = 332.0636  # Most common in protein force fields
COULOMB_CONSTANT_ATOMIC = 1.0  # Atomic units (Hartree)

# Default Coulomb constant (use kcal/mol units)
COULOMB_CONSTANT = COULOMB_CONSTANT_KCAL

# Numerical stability parameters
MIN_DISTANCE = 1e-7  # Minimum distance to avoid division by zero (Angstroms)
MAX_FORCE = 1e6  # Maximum force magnitude (kcal/mol/Å) for clamping

# Unit conversions
KCAL_TO_KJ = 4.184  # kcal/mol to kJ/mol
KJ_TO_KCAL = 1.0 / KCAL_TO_KJ  # kJ/mol to kcal/mol
NM_TO_ANGSTROM = 10.0  # nanometers to Angstroms
ANGSTROM_TO_NM = 0.1  # Angstroms to nanometers

# Lennard-Jones defaults (for future use)
DEFAULT_SIGMA = 3.5  # Angstroms (typical for carbon)
DEFAULT_EPSILON = 0.1  # kcal/mol (typical for nonpolar atoms)
