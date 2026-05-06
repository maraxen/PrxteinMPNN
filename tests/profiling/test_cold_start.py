"""Regression / smoke tests for JIT compile-cache visibility (roadmap critic #6)."""

from __future__ import annotations


def test_import_model_public_surface_smoke() -> None:
  """Import stack after Phase **5e** split must remain cycle-free and public symbols stable."""
  import prxteinmpnn.model.ligand_mpnn as lm  # noqa: PLC0415
  import prxteinmpnn.model.mpnn as mp  # noqa: PLC0415

  assert mp.PrxteinMPNN is not None
  assert lm.PrxteinLigandMPNN is not None
  assert mp.PrxteinLigandMPNN is lm.PrxteinLigandMPNN
