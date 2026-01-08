"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from flax.struct import dataclass

if TYPE_CHECKING:
  from jaxtyping import Int

  from prxteinmpnn.utils.types import (
    BIC,
    ComponentCounts,
    Converged,
    Covariances,
    EnsembleData,
    LogLikelihood,
    Means,
    Responsibilities,
    Weights,
  )

from dataclasses import dataclass as dc

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
  "GMM",
  "EMFitterResult",
  "EMLoopState",
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


@dataclass
class EstatInfo:
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


@dataclass
class _EStepState:
  """State for accumulating statistics during the E-step."""

  component_counts: ComponentCounts
  weighted_data: EnsembleData
  weighted_squared_data: EnsembleData
  log_likelihood_total: LogLikelihood


@dataclass
class GMM:
  """Dataclass to hold GMM parameters."""

  means: Means
  covariances: Covariances
  weights: Weights
  responsibilities: Responsibilities
  n_components: int
  n_features: int


class EMLoopState(NamedTuple):
  """State for the in-memory EM loop."""

  gmm: GMM
  n_iter: Int
  log_likelihood: LogLikelihood
  log_likelihood_diff: LogLikelihood


@dataclass
class EMFitterResult:
  """Result of the Expectation-Maximization fitting process.

  Attributes
  ----------
  gmm : GMM
      The final fitted Gaussian mixture model.
  n_iter : jax.Array
      The total number of iterations performed.
  log_likelihood : jax.Array
      The log-likelihood of the data under the final model.
  converged : jax.Array
      A boolean indicating if the algorithm converged within the max iterations.

  """

  gmm: GMM
  n_iter: Int
  log_likelihood: LogLikelihood
  log_likelihood_diff: LogLikelihood
  converged: Converged
  features: EnsembleData | None = None
  bic: BIC | None = None
