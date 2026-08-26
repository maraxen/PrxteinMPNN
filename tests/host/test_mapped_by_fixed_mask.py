"""G1 (audit 260826_chain-selection-vendor-superset-audit): json-mapped-by-X batch input
mapping, `by="path"` only.

Two concerns:

1. TestResolveMappedBy -- unit-level: `MappedBy`/`resolve_mapped_by` in isolation (no model,
   no batch). Fast, always runs.

2. TestSampleWithMappedByFixedMask -- real, real-checkpoint two-structure differential
   confirming `sample()` actually applies DISTINCT per-structure fixed_mask/fixed_tokens when
   given a `MappedBy(by="path", ...)`, keyed by each structure's canonical id (the
   `Path(input).stem` convention `_canonical_structure_id` already uses -- see
   `MappedBy`'s own docstring for why this deviates from vendor's literal-path JSON keys).
   Marked `slow`: loads a real proteinmpnn_v_48_020 checkpoint, same pattern as
   `test_decode_path_differential.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.run.batch_mapping import MappedBy, resolve_mapped_by

_MAX_LENGTH = 128
_STRUCTURE_A = "tests/data/1ubq.pdb"  # canonical id "1ubq", 76 real residues
_STRUCTURE_B = "tests/data/1mbn.pdb"  # canonical id "1mbn", 153 real residues
_CHECKPOINT = "proteinmpnn_v_48_020"


class TestResolveMappedBy:
  def test_by_path_is_supported(self) -> None:
    MappedBy(by="path", mapping={"1ubq": 1})

  def test_unsupported_by_raises_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError, match="chain_id"):
      MappedBy(by="chain_id", mapping={"1ubq": 1})

  def test_non_mapped_by_value_broadcasts_to_every_structure(self) -> None:
    resolved = resolve_mapped_by(5, structure_ids=["1ubq", "1mbn"], field_name="fixed_mask")
    assert resolved == [5, 5]

  def test_none_broadcasts_to_every_structure(self) -> None:
    resolved = resolve_mapped_by(None, structure_ids=["1ubq", "1mbn"], field_name="fixed_mask")
    assert resolved == [None, None]

  def test_mapped_by_resolves_in_structure_id_order(self) -> None:
    mapped = MappedBy(by="path", mapping={"1ubq": "a", "1mbn": "b"})
    resolved = resolve_mapped_by(
      mapped,
      structure_ids=["1mbn", "1ubq"],
      field_name="fixed_mask",
    )
    assert resolved == ["b", "a"]

  def test_missing_structure_id_raises_value_error(self) -> None:
    mapped = MappedBy(by="path", mapping={"1ubq": "a"})
    with pytest.raises(ValueError, match="1mbn"):
      resolve_mapped_by(mapped, structure_ids=["1ubq", "1mbn"], field_name="fixed_mask")


def _decoded_sequence_from_result(result: dict, row: int) -> str:
  from aminx.utils.aa_convert import MPNN_ALPHABET

  seqs = np.asarray(result["sequences"])
  tokens = seqs.reshape(seqs.shape[0], -1)[row]
  return "".join(MPNN_ALPHABET[int(t)] for t in tokens)


def _fixed_tokens_row(token_id: int) -> np.ndarray:
  row = np.zeros((_MAX_LENGTH,), dtype=np.int32)
  row[0] = token_id
  return row


def _fixed_mask_row() -> np.ndarray:
  row = np.zeros((_MAX_LENGTH,), dtype=np.float32)
  row[0] = 1.0
  return row


@pytest.mark.slow
class TestSampleWithMappedByFixedMask:
  """Real two-structure batch: distinct forced identity per structure via MappedBy(by='path')."""

  def test_mapped_by_forces_distinct_identity_per_structure(self) -> None:
    from aminx.host.runner import sample
    from aminx.run.specs import SamplingSpecification
    from aminx.utils.aa_convert import MPNN_ALPHABET

    token_a = MPNN_ALPHABET.index("I")  # forced on structure A (1ubq)
    token_b = MPNN_ALPHABET.index("W")  # forced on structure B (1mbn), deliberately different

    spec = SamplingSpecification(
      inputs=[_STRUCTURE_A, _STRUCTURE_B],
      checkpoint_id=_CHECKPOINT,
      max_length=_MAX_LENGTH,
      batch_size=2,
      num_samples=1,
      random_seed=0,
      fixed_mask=MappedBy(
        by="path",
        mapping={"1ubq": _fixed_mask_row(), "1mbn": _fixed_mask_row()},
      ),
      fixed_tokens=MappedBy(
        by="path",
        mapping={"1ubq": _fixed_tokens_row(token_a), "1mbn": _fixed_tokens_row(token_b)},
      ),
    )
    result = sample(spec)

    seq_a = _decoded_sequence_from_result(result, row=0)
    seq_b = _decoded_sequence_from_result(result, row=1)

    assert seq_a[0] == "I"
    assert seq_b[0] == "W"

  def test_missing_batch_entry_raises_value_error(self) -> None:
    from aminx.host.runner import sample
    from aminx.run.specs import SamplingSpecification

    spec = SamplingSpecification(
      inputs=[_STRUCTURE_A, _STRUCTURE_B],
      checkpoint_id=_CHECKPOINT,
      max_length=_MAX_LENGTH,
      batch_size=2,
      num_samples=1,
      random_seed=0,
      fixed_mask=MappedBy(by="path", mapping={"1ubq": _fixed_mask_row()}),
      fixed_tokens=MappedBy(
        by="path",
        mapping={"1ubq": _fixed_tokens_row(0)},
      ),
    )
    with pytest.raises(ValueError, match="1mbn"):
      sample(spec)
