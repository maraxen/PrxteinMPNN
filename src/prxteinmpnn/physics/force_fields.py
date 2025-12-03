"""Force field parameter storage and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from huggingface_hub import hf_hub_download, list_repo_files

from prxteinmpnn.physics.constants import DEFAULT_EPSILON, DEFAULT_SIGMA


class FullForceField(eqx.Module):
  """Force field parameters stored as a PyTree.

  This is a data container (not a computational class). JAX arrays are
  dynamic leaves that can be updated, while metadata is static.

  Attributes:
      charges_by_id: Partial charges for each atom type
      sigmas_by_id: Lennard-Jones sigma parameters (Angstroms)
      epsilons_by_id: Lennard-Jones epsilon parameters (kcal/mol)
      cmap_energy_grids: CMAP energy correction grids (N_maps, Grid, Grid)
      atom_key_to_id: Map from (residue, atom) to integer ID
      id_to_atom_key: Reverse map from ID to (residue, atom)
      atom_class_map: Map from atom to force field class
      atom_type_map: Map from atom to force field type
      bonds: Bond parameters
      angles: Angle parameters
      propers: Proper dihedral parameters
      impropers: Improper dihedral parameters
      cmap_torsions: CMAP torsion definitions
      residue_templates: Residue internal topology (bonds)
      source_files: Source XML files used to create this force field

  Example:
      >>> ff = load_force_field("ff14SB")
      >>> charge = ff.charges_by_id[ff.atom_key_to_id[("ALA", "CA")]]
      >>> print(f"CA charge in ALA: {charge}")

  """

  # Dynamic leaves (JAX arrays - trainable/updatable)
  charges_by_id: jnp.ndarray  # (n_atoms,) partial charges
  sigmas_by_id: jnp.ndarray  # (n_atoms,) LJ sigma (Angstroms)
  epsilons_by_id: jnp.ndarray  # (n_atoms,) LJ epsilon (kcal/mol)
  cmap_energy_grids: jnp.ndarray  # (n_maps, grid_size, grid_size)

  # Static metadata (immutable)
  atom_key_to_id: dict[tuple[str, str], int] = eqx.field(static=True)
  id_to_atom_key: list[tuple[str, str]] = eqx.field(static=True)
  atom_class_map: dict[str, str] = eqx.field(static=True)
  atom_type_map: dict[str, str] = eqx.field(static=True)
  bonds: list[tuple[str, str, float, float]] = eqx.field(static=True)
  angles: list[tuple[str, str, str, float, float]] = eqx.field(static=True)
  propers: list[dict[str, Any]] = eqx.field(static=True)
  impropers: list[dict[str, Any]] = eqx.field(static=True)
  cmap_torsions: list[dict[str, Any]] = eqx.field(static=True)
  residue_templates: dict[str, list[tuple[str, str]]] = eqx.field(static=True)
  source_files: list[str] = eqx.field(static=True)

  def get_charge(self, residue: str, atom: str) -> float:
    """Get charge for a specific atom.

    Args:
        residue: Residue name (e.g., "ALA")
        atom: Atom name (e.g., "CA")

    Returns:
        Partial charge in elementary charge units

    """
    atom_id = self.atom_key_to_id.get((residue, atom))
    if atom_id is None:
      return 0.0  # Unknown atoms have zero charge
    return float(self.charges_by_id[atom_id])

  def get_lj_params(self, residue: str, atom: str) -> tuple[float, float]:
    """Get Lennard-Jones parameters for a specific atom.

    Args:
        residue: Residue name
        atom: Atom name

    Returns:
        Tuple of (sigma, epsilon) in (Angstroms, kcal/mol)

    """
    atom_id = self.atom_key_to_id.get((residue, atom))
    if atom_id is None:
      return DEFAULT_SIGMA, DEFAULT_EPSILON
    return float(self.sigmas_by_id[atom_id]), float(self.epsilons_by_id[atom_id])


def _make_ff_skeleton(hyperparams: dict[str, Any]) -> FullForceField:
  """Create an empty FullForceField from hyperparameters.

  Used internally for deserialization.
  """
  num_atoms = len(hyperparams["id_to_atom_key"])
  
  # Infer CMAP shape from metadata if available, else default to empty
  num_maps = hyperparams.pop("num_cmap_maps", 0)
  grid_size = hyperparams.pop("cmap_grid_size", 24)
  
  # Backward compatibility
  if "cmap_torsions" not in hyperparams:
      hyperparams["cmap_torsions"] = []
  if "residue_templates" not in hyperparams:
      hyperparams["residue_templates"] = {}
  if "atom_type_map" not in hyperparams:
      hyperparams["atom_type_map"] = {}

  return FullForceField(

    charges_by_id=jnp.zeros(num_atoms, dtype=jnp.float32),
    sigmas_by_id=jnp.zeros(num_atoms, dtype=jnp.float32),
    epsilons_by_id=jnp.zeros(num_atoms, dtype=jnp.float32),
    cmap_energy_grids=jnp.zeros((num_maps, grid_size, grid_size), dtype=jnp.float32),
    **hyperparams,
  )


def save_force_field(
  filepath: Path | str,
  force_field: FullForceField,
) -> None:
  """Save force field to disk using Equinox serialization.

  Format: JSON hyperparameters (first line) + binary PyTree leaves.

  Args:
      filepath: Path to save file (.eqx extension recommended)
      force_field: FullForceField object to save

  """
  filepath = Path(filepath)

  hyperparams = {
    "atom_key_to_id": force_field.atom_key_to_id,
    "id_to_atom_key": force_field.id_to_atom_key,
    "atom_class_map": force_field.atom_class_map,
    "atom_type_map": force_field.atom_type_map,
    "bonds": force_field.bonds,
    "angles": force_field.angles,
    "propers": force_field.propers,
    "impropers": force_field.impropers,
    "cmap_torsions": force_field.cmap_torsions,
    "residue_templates": force_field.residue_templates,
    "source_files": force_field.source_files,
    # Metadata for reconstructing skeleton
    "num_cmap_maps": force_field.cmap_energy_grids.shape[0],
    "cmap_grid_size": force_field.cmap_energy_grids.shape[1]
  }

  sanitized_hyperparams = hyperparams.copy()
  sanitized_hyperparams["atom_key_to_id"] = {
    f"{key[0]}|{key[1]}": value
    for key, value in hyperparams["atom_key_to_id"].items()  # type: ignore[possibly-missing-attribute]
  }

  with Path(filepath).open("wb") as f:
    # Write JSON hyperparameters
    hyperparam_str = json.dumps(sanitized_hyperparams)
    f.write((hyperparam_str + "\n").encode())

    # Write PyTree leaves
    eqx.tree_serialise_leaves(f, force_field)


def load_force_field(
  filepath: Path | str,
) -> FullForceField:
  """Load force field from disk.

  Args:
      filepath: Path to .eqx file

  Returns:
      FullForceField object

  """
  filepath = Path(filepath)

  with filepath.open("rb") as f:
    hyperparams = json.loads(f.readline().decode())

    if "atom_key_to_id" in hyperparams:
      hyperparams["atom_key_to_id"] = {
        tuple(key.split("|", 1)): value for key, value in hyperparams["atom_key_to_id"].items()
      }
    if "id_to_atom_key" in hyperparams:
      hyperparams["id_to_atom_key"] = [
        tuple(item) if isinstance(item, list) else item for item in hyperparams["id_to_atom_key"]
      ]
    skeleton = _make_ff_skeleton(hyperparams)
    return eqx.tree_deserialise_leaves(f, skeleton)


def load_force_field_from_hub(
  force_field_name: str,
  repo_id: str = "maraxen/eqx-ff",
  cache_dir: Path | str | None = None,
) -> FullForceField:
  """Load force field from HuggingFace Hub.

  Args:
      force_field_name: Name of force field (e.g., "ff14SB")
      repo_id: HuggingFace repository ID
      cache_dir: Optional cache directory

  Returns:
      FullForceField object

  Example:
      >>> ff = load_force_field_from_hub("ff14SB")
      >>> print(f"Loaded {len(ff.id_to_atom_key)} atom types")

  """
  filename = f"{force_field_name}.eqx"
  local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    cache_dir=cache_dir,
  )
  return load_force_field(local_path)


def list_available_force_fields(
  repo_id: str = "maraxen/eqx-ff",
) -> list[str]:
  """List available force fields on HuggingFace Hub.

  Args:
      repo_id: HuggingFace repository ID

  Returns:
      List of force field names

  """
  files = list_repo_files(repo_id, repo_type="dataset")
  force_fields = [f.replace(".eqx", "") for f in files if f.endswith(".eqx")]
  return sorted(force_fields)
