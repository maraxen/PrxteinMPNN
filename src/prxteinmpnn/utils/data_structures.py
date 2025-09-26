"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax.numpy as jnp
from flax.struct import dataclass

if TYPE_CHECKING:
  import numpy as np
  from jaxtyping import Int

  from prxteinmpnn.utils.types import (
    AtomMask,
    BackboneDihedrals,
    ChainIndex,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
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
  nitrogen_mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  full_coordinates: np.ndarray | None = None
  dihedrals: np.ndarray | None = None
  source: str | None = None
  mapping: np.ndarray | None = None


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
    atom_mask (AtomMask): Binary float mask to indicate presence of a particular atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_atom_type].
    residue_index (AtomResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is [num_res].

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  one_hot_sequence: OneHotProteinSequence
  atom_mask: AtomMask
  nitrogen_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  dihedrals: BackboneDihedrals | None = None
  mapping: Int | None = None
  full_coordinates: StructureAtomicCoordinates | None = None

  @classmethod
  def from_tuple(cls, protein_tuple: ProteinTuple) -> Protein:
    """Create a Protein instance from a ProteinTuple.

    Args:
        protein_tuple (ProteinTuple): The input protein tuple.

    Returns:
        Protein: The output protein dataclass.

    """
    return cls(
      coordinates=jnp.asarray(protein_tuple.coordinates, dtype=jnp.float32),
      aatype=jnp.asarray(protein_tuple.aatype, dtype=jnp.int8),
      one_hot_sequence=jnp.eye(21)[protein_tuple.aatype],
      atom_mask=jnp.asarray(protein_tuple.atom_mask, dtype=jnp.float32),
      residue_index=jnp.asarray(protein_tuple.residue_index, dtype=jnp.int32),
      chain_index=jnp.asarray(protein_tuple.chain_index, dtype=jnp.int32),
      nitrogen_mask=jnp.asarray(protein_tuple.nitrogen_mask, dtype=jnp.float32),
      dihedrals=(
        None
        if protein_tuple.dihedrals is None
        else jnp.asarray(protein_tuple.dihedrals, dtype=jnp.float64)
      ),
      mapping=jnp.asarray(protein_tuple.mapping, dtype=jnp.int32)
      if protein_tuple.mapping is not None
      else None,
      full_coordinates=(
        None
        if protein_tuple.full_coordinates is None
        else jnp.asarray(protein_tuple.full_coordinates, dtype=jnp.float32)
      ),
    )


ProteinStream = Generator[ProteinTuple, None]
ProteinBatch = Sequence[Protein]


@dataclass(frozen=True)
class ProteinEnsemble:
  """Protein structure or ensemble representation.

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
    chain_index (ChainIndex): Chain index for each residue. Shape is [num_res].
    dihedrals (BackboneDihedrals | None): Dihedral angles for backbone atoms (phi, psi, omega).
      Shape is [num_res, 3]. If not provided, defaults to None.
    mapping (jnp.Array | None): Optional array mapping residues in the ensemble to original
      structure indices. Shape is [num_res, num_frames]. If not provided, defaults to None.

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  one_hot_sequence: OneHotProteinSequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  dihedrals: BackboneDihedrals | None = None
  mapping: Int | None = None


OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]
