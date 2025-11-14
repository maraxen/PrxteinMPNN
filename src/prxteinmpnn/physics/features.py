"""Compute physics-based node features for protein structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.physics.electrostatics import compute_coulomb_forces_at_backbone
from prxteinmpnn.physics.projections import project_forces_onto_backbone
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.utils.data_structures import ProteinTuple


def compute_electrostatic_node_features(
  protein: ProteinTuple,
) -> jax.Array:
  """Compute SE(3)-invariant electrostatic features for each residue.

  Computes electrostatic forces at backbone atoms (N, CA, C, O, CB/H) from all
  charged atoms, then projects these forces onto the backbone frame to create
  5D SE(3)-invariant features per residue.

  For Glycine residues (which lack CB), the 5th backbone position contains a
  hydrogen atom. The force calculation naturally handles this since the position
  and charge are already set correctly in the ProteinTuple.

  Self-interactions are automatically excluded in the force calculation - an atom
  does not exert force on itself.

  Args:
      protein: ProteinTuple containing structure and charge information.
        Must have:
        - coordinates: (n_residues, n_atom_types, 3) backbone atom positions
        - charges: (n_residues, n_atom_types) partial charges for all atoms
        - aatype: (n_residues,) amino acid type indices
      exclude_cb_self_interaction: If True, exclude forces between CB/H and itself
        within the same residue (default: True). This prevents the CB hydrogen
        on glycine from interacting with itself.

  Returns:
      Electrostatic features, shape (n_residues, 5):
      - f_forward: Force component along CA→next_N (forward chain direction)
      - f_backward: Force component along CA→prev_C (backward chain direction)
      - f_sidechain: Force component along CA→CB (sidechain direction)
      - f_oop: Out-of-plane force component (perpendicular to backbone plane)
      - f_mag: Total force magnitude

  Raises:
      ValueError: If protein.charges is None (PQR data required)

  Example:
      >>> protein = load_pqr_file("protein.pqr")
      >>> features = compute_electrostatic_node_features(protein)
      >>> print(features.shape)
      (n_residues, 5)

  """
  if protein.charges is None:
    msg = "ProteinTuple must have charges (PQR data) to compute electrostatic features"
    raise ValueError(msg)

  if protein.full_coordinates is None:
    msg = "ProteinTuple must have full_coordinates to compute electrostatic features"
    raise ValueError(msg)

  # compute_backbone_coordinates returns (n_residues, 5, 3) in this exact order
  backbone_positions = compute_backbone_coordinates(
    jnp.array(protein.coordinates),
  )
  all_positions = jnp.array(protein.full_coordinates)
  all_charges = jnp.array(protein.charges)

  n_residues = backbone_positions.shape[0]
  backbone_positions_flat = backbone_positions.reshape(-1, 3)  # (n_residues*5, 3)

  distances = jnp.linalg.norm(
    backbone_positions_flat[:, None, :] - all_positions[None, :, :],
    axis=-1,
  )

  closest_indices = jnp.argmin(distances, axis=1)
  backbone_charges_flat = all_charges[closest_indices]
  backbone_charges = backbone_charges_flat.reshape(n_residues, 5)

  forces_at_backbone = compute_coulomb_forces_at_backbone(
    backbone_positions,
    all_positions,
    backbone_charges,
    all_charges,
  )

  return project_forces_onto_backbone(
    forces_at_backbone,
    backbone_positions,
  )


def compute_electrostatic_features_batch(
  proteins: Sequence[ProteinTuple],
  max_length: int | None = None,
  *,
  pad_value: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Compute electrostatic features for a batch of proteins with padding.

  Args:
      proteins: List of ProteinTuple instances
      max_length: Maximum sequence length for padding. If None, uses the
        longest sequence in the batch.
      pad_value: Value to use for padding (default: 0.0)

  Returns:
      features: (batch_size, max_length, 5) padded feature arrays
      mask: (batch_size, max_length) binary mask (1.0 for real residues, 0.0 for padding)

  Example:
      >>> proteins = [load_pqr_file(f"protein_{i}.pqr") for i in range(4)]
      >>> features, mask = compute_electrostatic_features_batch(proteins, max_length=128)
      >>> print(features.shape, mask.shape)
      (4, 128, 5) (4, 128)

  """
  if not proteins:
    msg = "Must provide at least one protein"
    raise ValueError(msg)

  features_list = [compute_electrostatic_node_features(p) for p in proteins]

  lengths = [f.shape[0] for f in features_list]
  if max_length is None:
    max_length = max(lengths)
  elif max_length < max(lengths):
    msg = f"max_length={max_length} is less than longest sequence ({max(lengths)})"
    raise ValueError(msg)

  batch_size = len(proteins)
  n_features = 5

  features_padded = jnp.full((batch_size, max_length, n_features), pad_value)
  mask = jnp.zeros((batch_size, max_length))

  for i, (features, length) in enumerate(zip(features_list, lengths, strict=False)):
    features_padded = features_padded.at[i, :length, :].set(features)
    mask = mask.at[i, :length].set(1.0)

  return features_padded, mask
