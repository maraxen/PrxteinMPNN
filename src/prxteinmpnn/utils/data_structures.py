"""Dataclasses and enums for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AtomMask,
    ChainIndex,
    InputBias,
    InputLengths,
    ProteinSequence,
    ResidueIndex,
    SamplingHyperparameters,
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

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex


@dataclass(frozen=True)
class DihedralStructure:
  """Protein structure representation with dihedral angles.

  Attributes:
    dihedrals (jnp.ndarray): Dihedral angles in radians. Shape is (num_res, num_dihedrals),
      where num_res is the number of residues and num_dihedrals is the number of dihedral angles
      per residue (e.g., 7 for standard residues).
    aatype (ProteinSequence): Amino-acid type for each residue represented as an integer between
      0 and 20, where 20 is 'X'. Shape is (num_res,).
    atom_mask (AtomMask): Binary float mask to indicate presence of a particular atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is (num_res, num_atom_type).
    residue_index (ResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is (num_res,).

  """

  phi_angles: jnp.ndarray
  psi_angles: jnp.ndarray
  omega_angles: jnp.ndarray
  aatype: ProteinSequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex


ProteinEnsemble = Iterator["ProteinStructure"]
ProteinDihedralEnsemble = Iterator["DihedralStructure"]


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
    residue_index (ResidueIndex): Index of residues in the structure, used for mapping atoms
      in structures to their residues. Shape is (num_residues,).
    chain_index (ChainIndex): Index of chains in the structure, used for mapping
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
  mask: AlphaCarbonMask = field(default_factory=lambda: jnp.array([]))
  """Mask for the model input, indicating valid atoms structure."""
  residue_index: ResidueIndex = field(default_factory=lambda: jnp.array([]))
  """Index of residues in the structure, used for mapping atoms in structures to their residues."""
  chain_index: ChainIndex = field(default_factory=lambda: jnp.array([]))
  """Index of chains in the structure, used for mapping atoms in structures to their chains."""
  lengths: InputLengths = field(default_factory=lambda: jnp.array([]))
  """Lengths of the sequences in the batch, used for padding and batching."""
  bias: InputBias | None = field(default_factory=lambda: None)
  """Bias for the model input, used for classification tasks.
  Defaults to zero bias of shape (sum(lengths), 20)."""
  k_neighbors: int = 48
  """Number of neighbors to consider for each atom in the structure."""
  augment_eps: float = 0.0
  """Epsilon value for adding noise to the backbone coordinates, used for data augmentation."""


@dataclass(frozen=True)
class SamplingInputs(ModelInputs):
  """Dataclass for inputs used in sequence sampling.

  Attributes:
    prng_key (PRNGKeyArray): Random key for JAX operations.
    initial_sequence (ProteinSequence): Initial sequence of amino acids as an array of integers.
    structure_coordinates (StructureAtomicCoordinates): Atomic coordinates of the structure.
    mask (AtomMask): Mask indicating valid atoms in the structure.
    residue_index (AtomResidueIndex): Index of residues in the structure.
    chain_index (AtomChainIndex): Index of chains in the structure.
    bias (InputBias | None): Bias for the model input, used for classification tasks.
    k_neighbors (int): Number of neighbors to consider for each atom.
    augment_eps (float): Epsilon value for adding noise to the backbone coordinates.
    hyperparameters (SamplingHyperparameters): Hyperparameters for sampling, e.g., temperature,
      top-k, etc.
    iterations (int): Number of iterations for sampling.

  """

  prng_key: PRNGKeyArray = field(default_factory=lambda: jax.random.PRNGKey(0))
  """Random key for JAX operations."""
  hyperparameters: SamplingHyperparameters = (0.0,)
  """Hyperparameters for sampling, e.g., temperature, top-k, etc."""
  iterations: int = 1
  """Number of iterations for sampling."""


class OligomerType(enum.Enum):
  """Enum for different types of oligomers."""

  MONOMER = "monomer"
  HETEROMER = "heteromer"
  HOMOOLIGOMER = "homooligomer"
  TIED_HOMOOLIGOMER = "tied_homooligomer"
