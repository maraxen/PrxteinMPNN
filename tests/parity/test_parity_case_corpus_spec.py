"""Schema checks for the parity evidence case corpus."""

from __future__ import annotations

import json
from pathlib import Path


def test_parity_case_corpus_schema() -> None:
  """Case corpus includes required real and synthetic case definitions."""
  corpus_path = Path(__file__).with_name("parity_case_corpus.json")
  payload = json.loads(corpus_path.read_text(encoding="utf-8"))

  assert payload["version"] == 1
  real_backbones = payload["real_backbones"]
  synthetic_backbones = payload["synthetic_backbones"]
  assert isinstance(real_backbones, list)
  assert isinstance(synthetic_backbones, list)
  assert len(real_backbones) >= 5
  assert len(synthetic_backbones) >= 6

  for item in real_backbones:
    assert "id" in item
    assert "path" in item
    assert "max_length" in item

  for item in synthetic_backbones:
    assert "id" in item
    assert "length" in item
    assert "seed" in item

  ligand_context = payload["ligand_context"]
  assert ligand_context["atom_context_num"] > 0
  assert 0.0 < ligand_context["mask_keep_probability"] <= 1.0

  intrinsic_noise = payload["intrinsic_noise"]
  assert intrinsic_noise["repeats"] >= 2
  assert intrinsic_noise["seed_step"] >= 1

  macro_signals = payload["macro_signals"]
  assert macro_signals["samples_per_case"] >= 2
  assert isinstance(macro_signals["include_no_ligand"], bool)
  assert isinstance(macro_signals["include_ligand"], bool)
  assert isinstance(macro_signals["include_sidechain_conditioned"], bool)
