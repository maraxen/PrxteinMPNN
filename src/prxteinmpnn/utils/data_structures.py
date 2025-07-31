"""Dataclasses and enums for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax.struct import dataclass, field

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AtomChainIndex,
    AtomMask,
    AtomResidueIndex,
    BFactors,
    InputBias,
    InputLengths,
    ResidueIndex,
    Sequence,
    StructureAtomicCoordinates,
  )


@dataclass(frozen=True)
class ProteinStructure:
  """Protein structure representation.

  Attributes:
    atom_positions (StructureAtomicCoordinates): Atom positions in the structure, represented as a
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
  aatype: Sequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  b_factors: BFactors


@dataclass(frozen=True)
class ModelInputs:
  """Dataclass for general model inputs.

  Note that any of these can be stacked together to form a batch of inputs.
  """

  structure_coordinates: StructureAtomicCoordinates = field(default_factory=lambda: jnp.array([]))
  """Structure atomic coordinates for the model input."""
  sequence: Sequence = field(default_factory=lambda: jnp.array([]))
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
