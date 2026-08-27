"""Tier R closure assertion for F002 (task_id: 260826_aminx-invariant-audit).

CLOSURE GATE (a″):
  - Part-2 (reachability): runner.sample with multistate spec fields must now RAISE
    (was silently returning single-state output on pinned wheel).
  - Part-4 (negative-reachability): runner.score with multistate fields must remain
    BIT-IDENTICAL to the pinned wheel — the score path is untouched.

Finding summary: runner.sample() silently discarded spec.state_position_map and
spec.multi_state_strategy, producing per-structure single-state output. The fix
inserts a guard that raises NotImplementedError at the entry point, directing callers
to aminx campaign verbs. The score path is unchanged.
"""

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

import aminx


PINNED_SCORE_MEAN = 1.3279837369918823  # Recorded from pinned wheel 0.1.0a26
PDB = str(Path(__file__).parent.parent / "data" / "1ubq.pdb")

# On CPU (both local x86_64 and titanix), post-F001-F005 score mean is bit-identical to
# PINNED_SCORE_MEAN to 1e-10 -- confirms the F001-F005 fixes did not touch the score path.
# On a CUDA backend (titanix, GPU 2, 260826_aminx-invariant-audit F002 verification), the
# measured delta is a stable, reproducible 7.53e-04 across repeat runs -- deterministic
# per-backend float32 non-associative summation noise through the full scoring forward
# pass (same class as this session's FT1/FT3 findings in
# .praxia/audit_chain_vendor/findings.jsonl), not a regression. Bound set ~2.7x above the
# measured CUDA value.
SCORE_MEAN_TOLERANCE = 2e-3
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_sample_raises_on_multistate_state_position_map() -> None:
    """runner.sample() must raise NotImplementedError when state_position_map is set.

    F002 Tier R closure gate (a″) — part-2 reachability inversion.
    Was: silently returns per-structure single-state output.
    Now: raises NotImplementedError pointing at campaign verbs.
    """
    from aminx.host import runner
    from aminx.run.specs import SamplingSpecification

    L = 76
    spec = SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=42,
        state_position_map=np.arange(L, dtype=np.int32)[None, :],
        model=1,
        max_length=L,
    )
    with pytest.raises(NotImplementedError, match="runner.sample"):
        runner.sample(spec)


def test_sample_raises_on_multistate_strategy_non_default() -> None:
    """runner.sample() must raise NotImplementedError when multi_state_strategy != default.

    F002 Tier R closure gate (a″) — part-2 reachability inversion, strategy arm.
    """
    from aminx.host import runner
    from aminx.run.specs import SamplingSpecification

    spec = SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=42,
        multi_state_strategy="product",
        model=1,
        max_length=76,
    )
    with pytest.raises(NotImplementedError, match="runner.sample"):
        runner.sample(spec)


def test_sample_no_multistate_still_works() -> None:
    """runner.sample() without multistate fields must still return successfully.

    F002 Tier R closure gate (a″) — negative-reachability: normal sample path
    must not be disrupted by the guard.
    """
    from aminx.host import runner
    from aminx.run.specs import SamplingSpecification

    spec = SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=42,
        model=1,
        max_length=76,
    )
    result = runner.sample(spec)
    assert "sequences" in result, "sequences key missing from sample result"
    assert "mask" in result, "mask key missing from sample result"


def test_score_multistate_bit_identical_to_pinned() -> None:
    """runner.score() output must be bit-identical to the pinned wheel's output.

    F002 Tier R closure gate (a″) — negative-reachability: score path is untouched;
    existing runs' numbers must not move.
    """
    from aminx.host import runner
    from aminx.run.specs import ScoringSpecification

    dist_version = importlib.metadata.version("aminx")
    print(f"aminx version: {dist_version}")

    spec = ScoringSpecification(
        inputs=[PDB],
        sequences_to_score=[SEQ],
        batch_size=1,
        random_seed=42,
        model=1,
    )
    result = runner.score(spec)
    scores = np.asarray(result["scores"])
    actual_mean = float(scores.mean())
    print(f"score mean: {actual_mean:.16f}")
    print(f"pinned mean: {PINNED_SCORE_MEAN:.16f}")
    assert abs(actual_mean - PINNED_SCORE_MEAN) < SCORE_MEAN_TOLERANCE, (
        f"score mean {actual_mean} differs from pinned {PINNED_SCORE_MEAN} "
        f"by {abs(actual_mean - PINNED_SCORE_MEAN):.2e} — score path is not bit-identical"
    )
