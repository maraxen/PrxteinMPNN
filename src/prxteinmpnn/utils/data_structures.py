"""Dataclasses and enums for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.utils.types import (
    AtomChainIndex,
    AtomMask,
    AtomResidueIndex,
    BFactors,
    ChainIndex,
    DecodingOrder,
    InputBias,
    InputLengths,
    ModelParameters,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


@dataclass(frozen=True)
class ProteinStructure:
  """Protein structure representation.

  Attributes:
    coordinates (StructureAtomicCoordinates): Atom positions in the structure, represented as a
      3D array. Cartesian coordinates of atoms in angstroms. The atom types correspond to
      residue_constants.atom_types, i.e. the first three are N, CA, CB. Shape is
      (num_res, num_atom_type, 3), where num_res is the number of residues, num_atom_type is the
      number of atom types (e.g., N, CA, CB, C, O), and 3 is the spatial dimension (x, y, z).
    aatype (Sequence): Amino-acid type for each residue represented as an integer between 0 and 20,
      where 20 is 'X'. Shape is [num_res].
    atom_mask (AtomMask): Binary float mask to indicate presence of a particular atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_atom_type].
    residue_index (AtomResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is [num_res].
    b_factors (BFactors): B-factors, or temperature factors, of each residue
      (in sq. angstroms units), representing the displacement of the residue from its ground truth
      mean value. Shape is [num_res, num_atom_type].

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  b_factors: BFactors


@dataclass(frozen=True)
class ModelInputs:
  """Dataclass for general model inputs.

  Note that any of these can be stacked together to form a batch of inputs.

  Attributes:
    structure_coordinates (StructureAtomicCoordinates): Atomic coordinates of the structure.
      Shape is (num_residues, num_atoms, 3), where num_residues is the number of residues
      and num_atoms is the number of atoms per residue (e.g., 37 for standard residues).
    sequence (Sequence): Sequence of amino acids as an array of integers.
      Shape is (num_residues,), where num_residues is the number of residues.
    mask (AtomMask): Mask for the model input, indicating valid atoms in the structure.
      Shape is (num_residues, num_atoms), where num_atoms is the number of atoms per residue.
    residue_index (AtomResidueIndex): Index of residues in the structure, used for mapping atoms
      in structures to their residues. Shape is (num_residues,).
    chain_index (AtomChainIndex): Index of chains in the structure, used for mapping
      atoms in structures to their chains. Shape is (num_residues,).
    lengths (InputLengths): Lengths of the sequences in the batch, used for padding and
      batching. Shape is (num_sequences,), where num_sequences is the number of sequences in the
      batch.
    bias (InputBias): Bias for the model input, used for classification tasks.
      Defaults to zero bias of shape (sum(lengths), 20), where 20 is the number of amino acid types.
      Shape is (sum(lengths), num_classes), where num_classes is the number of classes
      (e.g., 20 for amino acids).

  """

  structure_coordinates: StructureAtomicCoordinates = field(default_factory=lambda: jnp.array([]))
  """Structure atomic coordinates for the model input."""
  sequence: ProteinSequence = field(default_factory=lambda: jnp.array([]))
  """A sequence of amino acids for the model input. As MPNN-alphabet based array of integers."""
  mask: AtomMask = field(default_factory=lambda: jnp.array([]))
  """Mask for the model input, indicating valid atoms structure."""
  residue_index: AtomResidueIndex = field(default_factory=lambda: jnp.array([]))
  """Index of residues in the structure, used for mapping atoms in structures to their residues."""
  chain_index: AtomChainIndex = field(default_factory=lambda: jnp.array([]))
  """Index of chains in the structure, used for mapping atoms in structures to their chains."""
  lengths: InputLengths = field(default_factory=lambda: jnp.array([]))
  """Lengths of the sequences in the batch, used for padding and batching."""
  bias: InputBias = field(default_factory=lambda: jnp.array([]))
  """Bias for the model input, used for classification tasks.
  Defaults to zero bias of shape (sum(lengths), 20)."""


@dataclass(frozen=True)
class ScoringInputs:
  """Dataclass for inputs used in sequence scoring.

  Attributes:
    key (PRNGKeyArray): Random key for JAX operations.
    sequence (Sequence): Sequence of amino acids as an array of integers.
    decoding_order (DecodingOrder): Order in which residues are processed during decoding.
    model_parameters (ModelParameters): Model parameters for the scoring function.
    structure_coordinates (StructureAtomicCoordinates): Atomic coordinates of the structure.
    atom_mask (AtomMask): Mask indicating valid atoms in the structure.
    residue_index (AtomResidueIndex): Index of residues in the structure.
    chain_index (AtomChainIndex): Index of chains in the structure.
    k_neighbors (int): Number of neighbors to consider for each atom.
    augment_eps (float): Epsilon value for adding noise to the backbone coordinates.

  """

  key: PRNGKeyArray = field(default_factory=lambda: jax.random.PRNGKey(0))
  sequence: ProteinSequence = field(default_factory=lambda: jnp.array([]))
  """Sequence of amino acids as an array of integers."""
  decoding_order: DecodingOrder = field(default_factory=lambda: jnp.array([]))
  model_parameters: ModelParameters = field(default_factory=lambda: jnp.array([]))
  structure_coordinates: StructureAtomicCoordinates = field(default_factory=lambda: jnp.array([]))
  atom_mask: AtomMask = field(default_factory=lambda: jnp.array([]))
  residue_index: AtomResidueIndex = field(default_factory=lambda: jnp.array([]))
  chain_index: AtomChainIndex = field(default_factory=lambda: jnp.array([]))
  k_neighbors: int = 48
  """Number of neighbors to consider for each atom."""
  augment_eps: float = 0.0
  """Epsilon value for adding noise to the backbone coordinates."""


class OligomerType(enum.Enum):
  """Enum for different types of oligomers."""

  MONOMER = "monomer"
  HETEROMER = "heteromer"
  HOMOOLIGOMER = "homooligomer"
  TIED_HOMOOLIGOMER = "tied_homooligomer"


class SamplingEnum(enum.Enum):
  """Enum for different sampling strategies."""

  GREEDY = "greedy"
  TOP_K = "top_k"
  TOP_P = "top_p"
  TEMPERATURE = "temperature"
  BEAM_SEARCH = "beam_search"
  STRAIGHT_THROUGH = "straight_through"

  def __str__(self) -> str:
    return self.value
