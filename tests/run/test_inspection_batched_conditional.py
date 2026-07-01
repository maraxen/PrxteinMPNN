"""InspectionSpecification validation for the 'batched_conditional_logits' feature.

Part 2 (Option B) of the aminx batched conditional-logits plan (tev_design necklace
campaign, task 260629_necklace-library-campaign). Numeric correctness of the batched
encode/decode path itself (shape (R, C, L, 21), spot-parity against the Python-loop
helper) is covered at the kernel level in
tests/sampling/test_batched_conditional_logits.py — the host/runner.py branch is a
thin wire-up (string->index candidate conversion + calling that already-tested split
fn) with no independent numerics to re-verify. No existing test in this suite invokes
aminx.host.runner.inspect() end-to-end (it requires a real structure-file fixture the
suite does not yet have); this file follows the established InspectionSpecification
validation-only test convention (tests/run/test_inspection_fixed_mask.py).
"""

from __future__ import annotations

import pytest

from aminx.run.specs import InspectionSpecification


def test_batched_conditional_logits_requires_candidate_sequences() -> None:
  """candidate_sequences must be non-empty when the feature is requested."""
  with pytest.raises(ValueError, match="candidate_sequences"):
    InspectionSpecification(
      inputs=["test.pdb"],
      inspection_features=("batched_conditional_logits",),
    )


def test_batched_conditional_logits_rejects_n_replicates_below_one() -> None:
  """n_replicates must be >= 1."""
  with pytest.raises(ValueError, match="n_replicates"):
    InspectionSpecification(
      inputs=["test.pdb"],
      inspection_features=("batched_conditional_logits",),
      candidate_sequences=("ACDE",),
      n_replicates=0,
    )


def test_batched_conditional_logits_rejects_multi_replicate_at_bb_zero() -> None:
  """n_replicates > 1 requires backbone_noise > 0 (replicates are key-invariant at bb=0)."""
  with pytest.raises(ValueError, match="backbone_noise"):
    InspectionSpecification(
      inputs=["test.pdb"],
      inspection_features=("batched_conditional_logits",),
      candidate_sequences=("ACDE", "ACDF"),
      n_replicates=4,
      backbone_noise=0.0,
    )


def test_batched_conditional_logits_accepts_valid_config() -> None:
  """Valid config round-trips: candidates set, n_replicates > 1, backbone_noise > 0."""
  spec = InspectionSpecification(
    inputs=["test.pdb"],
    inspection_features=("batched_conditional_logits",),
    candidate_sequences=("ACDE", "ACDF", "ACDG"),
    n_replicates=8,
    backbone_noise=0.1,
    replicate_batch_size=4,
    candidate_batch_size=2,
  )
  assert spec.candidate_sequences == ("ACDE", "ACDF", "ACDG")
  assert spec.n_replicates == 8
  assert spec.replicate_batch_size == 4
  assert spec.candidate_batch_size == 2


def test_batched_conditional_logits_default_single_replicate_allows_bb_zero() -> None:
  """n_replicates=1 (default) does not require backbone_noise > 0."""
  spec = InspectionSpecification(
    inputs=["test.pdb"],
    inspection_features=("batched_conditional_logits",),
    candidate_sequences=("ACDE",),
  )
  assert spec.n_replicates == 1
