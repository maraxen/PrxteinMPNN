"""Conformance: aminx's alphabet constants match the ecosystem's declarations.

Phase 0 (decision D4): **dev-dependency-only**. Nothing under `src/` imports the library.

If a test here fails, do NOT edit it to match. One of the two sides is wrong, and which one is
the finding.

NOTE ON DUPLICATION. `aminx/utils/aa_convert.py:16-17` declares the same two alphabets as
`proxide/chem/conversion.py:16-17`, in a repo that already depends on proxide and already
imports `proxide.chem.residues` in that same file. So aminx's eventual fix is
`delete these constants, re-export from proxide` -- it needs no new package at all. Pinning
them here first means that change can be made with a test proving nothing moved.
"""

from __future__ import annotations

import pytest

alphex = pytest.importorskip(
  "alphex",
  reason="alphabet conformance library not installed; add it to the dev group",
)
known = alphex.known
SpecialKind = alphex.SpecialKind


def test_aa_convert_declares_both_base_orderings() -> None:
  from aminx.utils.aa_convert import AF_ALPHABET, MPNN_ALPHABET

  assert MPNN_ALPHABET[:20] == known.MPNN_X_21.symbols
  assert MPNN_ALPHABET[20] == "X"
  assert AF_ALPHABET[:20] == known.AF_X_21.symbols
  assert AF_ALPHABET[20] == "X"
  assert known.MPNN_X_21.specials[SpecialKind.UNKNOWN] == 20
  assert known.AF_X_21.specials[SpecialKind.UNKNOWN] == 20


def test_the_two_orderings_are_not_interchangeable() -> None:
  """17 of 20 positions differ. This is the premise of the whole census."""
  from aminx.utils.aa_convert import AF_ALPHABET, MPNN_ALPHABET

  differ = [
    i for i in range(20) if MPNN_ALPHABET[i] != AF_ALPHABET[i]
  ]
  assert len(differ) == 17


def test_potts_alphabet_matches_the_mpnn_declaration() -> None:
  from aminx.potts.model import POTTS_ALPHABET

  assert POTTS_ALPHABET[:20] == known.MPNN_X_21.symbols
  assert POTTS_ALPHABET[20] == "X"


def test_aa_convert_duplicates_proxide_exactly() -> None:
  """The duplication that makes aminx's fix a deletion rather than a migration.

  If these ever diverge, one of the two copies has been edited in isolation -- which is the
  failure mode a single declaration exists to prevent.
  """
  from aminx.utils.aa_convert import AF_ALPHABET as AMINX_AF
  from aminx.utils.aa_convert import MPNN_ALPHABET as AMINX_MPNN

  proxide_conversion = pytest.importorskip(
    "proxide.chem.conversion",
    reason="proxide not importable in this environment",
  )
  assert AMINX_MPNN == proxide_conversion.MPNN_ALPHABET
  assert AMINX_AF == proxide_conversion.AF_ALPHABET
