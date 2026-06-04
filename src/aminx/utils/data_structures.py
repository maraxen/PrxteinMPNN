"""Dataclasses for the Aminx project.

aminx.utils.data_structures
"""

from __future__ import annotations

from dataclasses import dataclass as dc

import numpy as np
import equinox as eqx
from proxide.core.containers import (
  OligomerType,
  Protein,
  ProteinBatch,
  ProteinStream,
)

# Backward compatibility alias
# ProteinTuple alias removed - use Protein directly from proxide.core.containers

# Re-export these for compatibility if needed, though mostly used internally
__all__ = [
  "EstatInfo",
  "OligomerType",
  "Protein",
  "ProteinBatch",
  "ProteinStream",
  "TrajectoryStaticFeatures",
]


@dc
class TrajectoryStaticFeatures:
  """A container for pre-computed, frame-invariant protein features."""

  aatype: np.ndarray
  static_atom_mask_37: np.ndarray
  residue_indices: np.ndarray
  chain_index: np.ndarray
  valid_atom_mask: np.ndarray
  nitrogen_mask: np.ndarray
  num_residues: int


class EstatInfo(eqx.Module):
  """Electrostatics information extracted from a PQR file.

  Attributes:
    charges: Numpy array of atomic charges.
    radii: Numpy array of atomic radii.
    epsilons: Numpy array of atomic epsilons.
    estat_backbone_mask: Boolean numpy array indicating backbone atoms.
    estat_resid: Integer numpy array of residue numbers.
    estat_chain_index: Integer numpy array of chain indices (ord value).

  """

  charges: np.ndarray
  radii: np.ndarray
  epsilons: np.ndarray
  estat_backbone_mask: np.ndarray
  estat_resid: np.ndarray
  estat_chain_index: np.ndarray
