"""Atom ordering constants for PrxteinMPNN.

This module defines constants for handling the different atom ordering schemes
used in protein structure files and the AlphaFold atom37 standard.

PDB File Order vs Atom37 Standard
----------------------------------
PDB files typically have atoms in the order: N, CA, C, O, CB, ...
The atom37 standard expects: N, CA, C, CB, O, ...

The difference is that O and CB are swapped. This matters because:
1. The parser outputs coordinates in the order they appear in the PDB file
2. The atom_order dict (from residue_constants.py) uses atom37 indices
3. We need to extract atoms using the correct indices based on context

Usage
-----
When working with coordinates from the parser, use PDB_ORDER_INDICES.
When working with atom37 standard ordering, use atom_order from residue_constants.

Example:
    >>> from prxteinmpnn.utils.atom_ordering import PDB_ORDER_INDICES
    >>> # Extract backbone atoms from parser output
    >>> nitrogen = coords[:, PDB_ORDER_INDICES['N'], :]
    >>> oxygen = coords[:, PDB_ORDER_INDICES['O'], :]  # Index 3, not 4!
"""

from typing import Final

# PDB file order indices for backbone atoms
# This is the order atoms typically appear in PDB files
PDB_ORDER_INDICES: Final[dict[str, int]] = {
    "N": 0,   # Nitrogen (backbone)
    "CA": 1,  # Alpha carbon
    "C": 2,   # Carbonyl carbon
    "O": 3,   # Carbonyl oxygen (index 3 in PDB, but 4 in atom37!)
    "CB": 4,  # Beta carbon (index 4 in PDB, but 3 in atom37!)
}

# Atom37 standard indices (for reference)
# This is defined in residue_constants.py as atom_order
# but we document it here for clarity
ATOM37_ORDER_INDICES: Final[dict[str, int]] = {
    "N": 0,   # Nitrogen
    "CA": 1,  # Alpha carbon
    "C": 2,   # Carbonyl carbon
    "CB": 3,  # Beta carbon (index 3 in atom37, but 4 in PDB!)
    "O": 4,   # Carbonyl oxygen (index 4 in atom37, but 3 in PDB!)
}

# Backbone atom names in PDB order
PDB_ORDER_BACKBONE: Final[tuple[str, ...]] = ("N", "CA", "C", "O", "CB")

# Backbone atom names in atom37 order
ATOM37_ORDER_BACKBONE: Final[tuple[str, ...]] = ("N", "CA", "C", "CB", "O")


def validate_ordering() -> None:
    """Validate that our ordering constants are correct.

    This function checks that the only difference between PDB and atom37
    ordering is the swap of O and CB at indices 3 and 4.

    Raises:
        AssertionError: If the ordering is incorrect.
    """
    # Check N, CA, C are the same
    assert PDB_ORDER_INDICES["N"] == ATOM37_ORDER_INDICES["N"] == 0
    assert PDB_ORDER_INDICES["CA"] == ATOM37_ORDER_INDICES["CA"] == 1
    assert PDB_ORDER_INDICES["C"] == ATOM37_ORDER_INDICES["C"] == 2

    # Check O and CB are swapped
    assert PDB_ORDER_INDICES["O"] == 3
    assert PDB_ORDER_INDICES["CB"] == 4
    assert ATOM37_ORDER_INDICES["CB"] == 3
    assert ATOM37_ORDER_INDICES["O"] == 4

    # Check tuple lengths
    assert len(PDB_ORDER_BACKBONE) == 5
    assert len(ATOM37_ORDER_BACKBONE) == 5


# Run validation on import
validate_ordering()
