"""Helper utilities for populating physics parameters from force fields.

This module provides utilities to populate missing physics parameters
(charges, sigmas, epsilons) using the existing FullForceField infrastructure.
"""

import logging

import numpy as np
from biotite.structure import AtomArray

from prolix.physics import DEFAULT_EPSILON, DEFAULT_SIGMA
from prolix.physics import FullForceField, load_force_field_from_hub

logger = logging.getLogger(__name__)


def populate_physics_parameters(
  atom_array: AtomArray,
  force_field: FullForceField | None = None,
  force_field_name: str = "ff14SB",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Populate physics parameters from force field for atoms without explicit values.

  Args:
      atom_array: Biotite AtomArray with atom information
      force_field: Optional pre-loaded FullForceField. If None, loads from hub.
      force_field_name: Name of force field to load if force_field is None

  Returns:
      Tuple of (charges, sigmas, epsilons) arrays

  """
  if force_field is None:
    try:
      logger.info("Loading force field: %s", force_field_name)
      force_field = load_force_field_from_hub(force_field_name)
    except Exception as e:  # noqa: BLE001
      logger.warning("Failed to load force field %s: %s. Using defaults.", force_field_name, e)
      return _get_default_parameters(atom_array)

  n_atoms = atom_array.array_length()
  charges = np.zeros(n_atoms, dtype=np.float32)
  sigmas = np.zeros(n_atoms, dtype=np.float32)
  epsilons = np.zeros(n_atoms, dtype=np.float32)

  # Get residue and atom names
  res_names = np.array(atom_array.res_name)
  atom_names = np.array(atom_array.atom_name)

  for i in range(n_atoms):
    res_name = res_names[i]
    atom_name = atom_names[i]

    # Try to get parameters from force field
    try:
      charges[i] = force_field.get_charge(res_name, atom_name)
      sigma, epsilon = force_field.get_lj_params(res_name, atom_name)
      sigmas[i] = sigma
      epsilons[i] = epsilon
    except Exception as e:  # noqa: BLE001
      logger.debug("Failed to get params for %s:%s, using defaults: %s", res_name, atom_name, e)
      # Defaults are already set (zeros for charge, will use element-based below)
      sigmas[i] = DEFAULT_SIGMA
      epsilons[i] = DEFAULT_EPSILON

  return charges, sigmas, epsilons


def _get_default_parameters(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Get default physics parameters based on element types.

  Fallback when force field is not available.
  """
  n_atoms = atom_array.array_length()
  elements = np.array(atom_array.element)

  # Simple element-based defaults
  element_params = {
    "C": (1.908, 0.086),
    "N": (1.824, 0.170),
    "O": (1.661, 0.210),
    "S": (2.000, 0.250),
    "H": (0.600, 0.0157),
    "P": (2.100, 0.200),
  }

  charges = np.zeros(n_atoms, dtype=np.float32)
  sigmas = np.zeros(n_atoms, dtype=np.float32)
  epsilons = np.zeros(n_atoms, dtype=np.float32)

  for i, elem in enumerate(elements):
    sigmas[i], epsilons[i] = element_params.get(elem, (DEFAULT_SIGMA, DEFAULT_EPSILON))

  return charges, sigmas, epsilons
