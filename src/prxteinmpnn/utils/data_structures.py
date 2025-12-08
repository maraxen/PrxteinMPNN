"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple
from dataclasses import dataclass as dc

import jax.numpy as jnp
import numpy as np  # ADDED: Required for runtime numpy operations
from flax.struct import dataclass

from prxteinmpnn.utils.residue_constants import atom_order

if TYPE_CHECKING:
  from jaxtyping import Int

  from prxteinmpnn.utils.types import (
    BIC,
    AlphaCarbonMask,
    AtomMask,
    BackboneDihedrals,
    ChainIndex,
    ComponentCounts,
    Converged,
    Covariances,
    EnsembleData,
    LogLikelihood,
    Means,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    Responsibilities,
    StructureAtomicCoordinates,
    Weights,
  )


class ProteinTuple(NamedTuple):
  """Tuple-based protein structure representation."""

  coordinates: np.ndarray
  aatype: np.ndarray
  atom_mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  full_coordinates: np.ndarray | None = None
  dihedrals: np.ndarray | None = None
  source: str | None = None
  mapping: np.ndarray | None = None
  charges: np.ndarray | None = None
  radii: np.ndarray | None = None
  sigmas: np.ndarray | None = None
  epsilons: np.ndarray | None = None
  estat_backbone_mask: np.ndarray | None = None
  estat_resid: np.ndarray | None = None
  estat_chain_index: np.ndarray | None = None
  physics_features: np.ndarray | None = None


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


def include_feature(feature_name: str, include_features: Sequence[str] | None) -> bool:
  """Determine if a feature should be included."""
  if include_features is None:
    return False
  return feature_name in include_features or "all" in include_features


@dataclass
class EstatInfo:
  """Electrostatics information extracted from a PQR file."""
  charges: np.ndarray
  radii: np.ndarray
  epsilons: np.ndarray
  estat_backbone_mask: np.ndarray
  estat_resid: np.ndarray
  estat_chain_index: np.ndarray


def none_or_jnp(array: np.ndarray | None) -> jnp.ndarray | None:
  """Convert a numpy array to jnp array, or return None if input is None."""
  if array is None:
    return None
  return jnp.asarray(array)

def none_or_numpy(array: np.ndarray | None) -> np.ndarray | None:
  """Convert array to numpy array, or return None if input is None."""
  if array is None:
    return None
  return np.asarray(array)


@dataclass(frozen=True)
class Protein:
  """Protein structure or ensemble representation."""

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  one_hot_sequence: OneHotProteinSequence
  mask: AlphaCarbonMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  dihedrals: BackboneDihedrals | None = None
  mapping: Int | None = None
  full_coordinates: StructureAtomicCoordinates | None = None
  full_atom_mask: AtomMask | None = None
  charges: jnp.ndarray | None = None
  radii: jnp.ndarray | None = None
  sigmas: jnp.ndarray | None = None
  epsilons: jnp.ndarray | None = None
  estat_backbone_mask: jnp.ndarray | None = None
  estat_resid: jnp.ndarray | None = None
  estat_chain_index: jnp.ndarray | None = None
  physics_features: jnp.ndarray | None = None

  @classmethod
  def from_tuple(
    cls,
    protein_tuple: ProteinTuple,
    *,
    include_extras: Sequence[
      Literal["dihedrals", "mapping", "full_coordinates", "full_atom_mask", "all"]
    ]
    | None = None,
  ) -> Protein:
    """Create a Protein instance from a ProteinTuple (Converts to JAX)."""
    return cls(
      coordinates=jnp.asarray(protein_tuple.coordinates, dtype=jnp.float32),
      aatype=jnp.asarray(protein_tuple.aatype, dtype=jnp.int8),
      one_hot_sequence=jnp.eye(21)[protein_tuple.aatype],
      mask=jnp.asarray(protein_tuple.atom_mask[:, atom_order["CA"]], dtype=jnp.float32),
      residue_index=jnp.asarray(protein_tuple.residue_index, dtype=jnp.int32),
      chain_index=jnp.asarray(protein_tuple.chain_index, dtype=jnp.int32),
      dihedrals=(
        None
        if protein_tuple.dihedrals is None or not include_feature("dihedrals", include_extras)
        else none_or_jnp(protein_tuple.dihedrals)
      ),
      mapping=(
        none_or_jnp(protein_tuple.mapping)
        if protein_tuple.mapping is not None
        and include_extras is not None
        and ("mapping" in include_extras or "all" in include_extras)
        else None
      ),
      full_coordinates=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_coordinates", include_extras)
        else none_or_jnp(protein_tuple.full_coordinates)
      ),
      full_atom_mask=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_atom_mask", include_extras)
        else none_or_jnp(protein_tuple.atom_mask)
      ),
      charges=none_or_jnp(protein_tuple.charges),
      radii=none_or_jnp(protein_tuple.radii),
      sigmas=none_or_jnp(protein_tuple.sigmas),
      epsilons=none_or_jnp(protein_tuple.epsilons),
      estat_backbone_mask=none_or_jnp(protein_tuple.estat_backbone_mask),
      estat_resid=none_or_jnp(protein_tuple.estat_resid),
      estat_chain_index=none_or_jnp(protein_tuple.estat_chain_index),
      physics_features=none_or_jnp(protein_tuple.physics_features),
    )

  @classmethod
  def from_tuple_numpy(
    cls,
    protein_tuple: ProteinTuple,
    *,
    include_extras: Sequence[
      Literal["dihedrals", "mapping", "full_coordinates", "full_atom_mask", "all"]
    ]
    | None = None,
  ) -> Protein:
    """Create a Protein instance from a ProteinTuple using NumPy (No JAX conversion)."""
    return cls(
      coordinates=np.asarray(protein_tuple.coordinates, dtype=np.float32),
      aatype=np.asarray(protein_tuple.aatype, dtype=np.int8),
      # Use NumPy eye
      one_hot_sequence=np.eye(21)[protein_tuple.aatype],
      mask=np.asarray(protein_tuple.atom_mask[:, atom_order["CA"]], dtype=np.float32),
      residue_index=np.asarray(protein_tuple.residue_index, dtype=np.int32),
      chain_index=np.asarray(protein_tuple.chain_index, dtype=np.int32),
      dihedrals=(
        None
        if protein_tuple.dihedrals is None or not include_feature("dihedrals", include_extras)
        else none_or_numpy(protein_tuple.dihedrals)
      ),
      mapping=(
        none_or_numpy(protein_tuple.mapping)
        if protein_tuple.mapping is not None
        and include_extras is not None
        and ("mapping" in include_extras or "all" in include_extras)
        else None
      ),
      full_coordinates=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_coordinates", include_extras)
        else none_or_numpy(protein_tuple.full_coordinates)
      ),
      full_atom_mask=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_atom_mask", include_extras)
        else none_or_numpy(protein_tuple.atom_mask)
      ),
      charges=none_or_numpy(protein_tuple.charges),
      radii=none_or_numpy(protein_tuple.radii),
      sigmas=none_or_numpy(protein_tuple.sigmas),
      epsilons=none_or_numpy(protein_tuple.epsilons),
      estat_backbone_mask=none_or_numpy(protein_tuple.estat_backbone_mask),
      estat_resid=none_or_numpy(protein_tuple.estat_resid),
      estat_chain_index=none_or_numpy(protein_tuple.estat_chain_index),
      physics_features=none_or_numpy(protein_tuple.physics_features),
    )


ProteinStream = Generator[ProteinTuple, None]
ProteinBatch = Sequence[Protein]


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
  """Result of the Expectation-Maximization fitting process."""
  gmm: GMM
  n_iter: Int
  log_likelihood: LogLikelihood
  log_likelihood_diff: LogLikelihood
  converged: Converged
  features: EnsembleData | None = None
  bic: BIC | None = None


OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]