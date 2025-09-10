"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Literal

from flax.struct import dataclass

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AtomMask,
    BackboneDihedrals,
    ChainIndex,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


@dataclass(frozen=True)
class Protein:
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
  one_hot_sequence: OneHotProteinSequence
  atom_mask: AtomMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  dihedrals: BackboneDihedrals | None = None


ProteinEnsemble = AsyncGenerator[tuple[Protein, str], None]


OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]
