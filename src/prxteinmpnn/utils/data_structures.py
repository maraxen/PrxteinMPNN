"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax.numpy as jnp
from flax.struct import dataclass

from prxteinmpnn.utils.residue_constants import atom_order

if TYPE_CHECKING:
  import numpy as np
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

from dataclasses import dataclass as dc


class ProteinTuple(NamedTuple):
  """Tuple-based protein structure representation.

  Attributes:
    coordinates (StructureAtomicCoordinates): Atom positions in the structure, represented as a
      3D array. Cartesian coordinates of atoms in angstroms.
      The atom types correspond to residue_constants.atom_types, i.e. the first three are N, CA, CB.
      Shape is (num_res, num_atom_type, 3), where num_res is the number of residues,
      num_atom_type is the number of atom types (e.g., N, CA, CB, C, O), and 3 is the spatial
      dimension (x, y, z).
    aatype (ProteinSequence): Amino-acid type for each residue represented as an integer between 0
    and 20,
      where 20 is 'X'. Shape is [num_res].
    atom_mask (AtomMask): Binary float mask to indicate presence of a particular atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_atom_type].
    residue_index (ResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is [num_res].
    chain_index (ChainIndex): Chain index for each residue. Shape is [num_res].
    dihedrals (BackboneDihedrals | None): Dihedral angles for backbone atoms (phi, psi, omega).
      Shape is [num_res, 3]. If not provided, defaults to None.

  """

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
  estat_backbone_mask: np.ndarray | None = None
  estat_resid: np.ndarray | None = None
  estat_chain_index: np.ndarray | None = None


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
  """Determine if a feature should be included.

  Args:
      feature_name (str): The name of the feature to check.
      include_features (Sequence[str] | None): The list of features to include.
          If None, no features are included.

  Returns:
      bool: True if the feature should be included, False otherwise.

  """
  if include_features is None:
    return False
  return feature_name in include_features or "all" in include_features


@dataclass
class EstatInfo:
  """Electrostatics information extracted from a PQR file.

  Attributes:
    charges: Numpy array of atomic charges.
    radii: Numpy array of atomic radii.
    estat_backbone_mask: Boolean numpy array indicating backbone atoms.
    estat_resid: Integer numpy array of residue numbers.
    estat_chain_index: Integer numpy array of chain indices (ord value).

  """

  charges: np.ndarray
  radii: np.ndarray
  estat_backbone_mask: np.ndarray
  estat_resid: np.ndarray
  estat_chain_index: np.ndarray


@dataclass(frozen=True)
class Protein:
  """Protein structure or ensemble representation.

  Attributes:
    coordinates (StructureAtomicCoordinates): Atom positions in the structure, represented as a
      3D array. Cartesian coordinates of atoms in angstroms. The atom types correspond to
      residue_constants.atom_types, i.e. the first three are N, CA, CB. Shape is
      (num_res, num_atom_type, 3), where num_res is the number of residues, num_atom_type is the
      number of atom types (e.g., N, CA, CB, C, O), and 3 is the spatial dimension (x, y, z).
    aatype (Sequence): Amino-acid type for each residue represented as an integer between 0 and 20,
      where 20 is 'X'. Shape is [num_res].
    mask (AlphaCarbonMask): Binary float mask to indicate presence of alpha carbon atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_atom_type].
    residue_index (AtomResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is [num_res].
    chain_index (ChainIndex): Chain index for each residue. Shape is [num_res].
    dihedrals (BackboneDihedrals | None): Dihedral angles for backbone atoms (phi, psi, omega).
      Shape is [num_res, 3]. If not provided, defaults to None.
    mapping (jnp.Array | None): Optional array mapping residues in the ensemble to original
      structure indices. Shape is [num_res, num_frames]. If not provided, defaults to None.
    full_coordinates (StructureAtomicCoordinates | None): Full atomic coordinates
      including all heavy atoms. Shape is (num_res, num_full_atom_type, 3), where num_full_atom_type
      is the number of all heavy atom types (e.g., N, CA, CB, C, O, CG, etc.), and 3 is the spatial
      dimension (x, y, z). If not provided, defaults to None.
    full_atom_mask (AtomMask | None): Binary float mask to indicate presence of a particular
      heavy atom. 1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_full_atom_type]. If not provided, defaults to None.

  """

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
  charges: np.ndarray | None = None
  radii: np.ndarray | None = None
  estat_backbone_mask: np.ndarray | None = None
  estat_resid: np.ndarray | None = None
  estat_chain_index: np.ndarray | None = None

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
    """Create a Protein instance from a ProteinTuple.

    Args:
        protein_tuple (ProteinTuple): The input protein tuple.
        include_extras:
            Optional list of extra fields to include from the tuple.
            If 'all' is included, all optional fields will be included.
            If None, no optional fields will be included.

    Returns:
        Protein: The output protein dataclass.

    """
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
        else jnp.asarray(protein_tuple.dihedrals, dtype=jnp.float64)
      ),
      mapping=jnp.asarray(protein_tuple.mapping, dtype=jnp.int32)
      if protein_tuple.mapping is not None
      and include_extras is not None
      and ("mapping" in include_extras or "all" in include_extras)
      else None,
      full_coordinates=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_coordinates", include_extras)
        else jnp.asarray(protein_tuple.full_coordinates, dtype=jnp.float32)
      ),
      full_atom_mask=(
        None
        if protein_tuple.full_coordinates is None
        or not include_feature("full_atom_mask", include_extras)
        else jnp.asarray(protein_tuple.atom_mask, dtype=jnp.float32)
      ),
      charges=protein_tuple.charges,
      radii=protein_tuple.radii,
      estat_backbone_mask=protein_tuple.estat_backbone_mask,
      estat_resid=protein_tuple.estat_resid,
      estat_chain_index=protein_tuple.estat_chain_index,
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


OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]
