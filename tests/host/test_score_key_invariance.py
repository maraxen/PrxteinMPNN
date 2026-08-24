"""End-to-end: ``runner.score`` must be deterministic in the PRNG key at zero backbone noise.

This is the property ``run/specs.py`` already assumes for the conditional-logits path
(``n_replicates > 1`` is refused at ``backbone_noise=0`` on the grounds that replicate draws
would be identical). The score path did NOT have it: ``scoring/score.py`` drew a fresh
decoding order per key and turned it into a fresh causal mask, so the score moved with the
key even at ``backbone_noise=0``.

That was not a rounding-scale wobble. Measured on 1LVB chain A at L=214, 8 seeds: sd 0.0177
nats, growing to 0.0299 at ``max_length=512`` -- against a reference effect size of 0.016 in
the consuming project. Worse, ``host/runner.py`` derives one key per CANDIDATE, so two
sequences scored in a single call were scored under DIFFERENT decoding orders, making every
WT/MUT contrast needlessly unpaired.

With the default mask now order-free (``full_context_ar_mask``), replicates at zero backbone
noise are bit-identical, which is both the correct estimand and ~7x less noisy for paired
contrasts.

Marked slow: it loads real weights and runs the full pipeline several times.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
SEEDS = (1, 7, 42, 1234)

# 1UBQ chain A, 76 residues. The identity is irrelevant to the property under test -- what
# matters is that the SAME sequence is scored under different keys -- but a real sequence
# keeps the logits in a realistic range rather than an all-one-residue degenerate corner.
UBIQUITIN_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.mark.slow
@pytest.mark.requires_weights
def test_score_is_bit_identical_across_prng_keys_at_zero_backbone_noise() -> None:
  """Different keys, zero backbone noise, same score -- exactly."""
  pytest.importorskip("aminx")
  # Local: must follow importorskip so collection stays cheap and does not hard-fail when
  # the package or its weights are unavailable.
  from aminx.host import runner  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  if not STRUCTURE.is_file():
    pytest.skip(f"structure fixture missing: {STRUCTURE}")

  scores = []
  for seed in SEEDS:
    result = runner.score(
      ScoringSpecification(
        inputs=str(STRUCTURE),
        chain_id="A",
        checkpoint_id="proteinmpnn_v_48_020",
        sequences_to_score=[UBIQUITIN_SEQUENCE],
        backbone_noise=0.0,
        random_seed=seed,
        return_logits=False,
      ),
    )
    scores.append(float(np.asarray(result["scores"], dtype=np.float64).ravel()[0]))

  spread = max(scores) - min(scores)
  assert spread == 0.0, (
    f"score varies across PRNG keys at backbone_noise=0 (spread {spread:.3e} nats over "
    f"seeds {SEEDS}): {scores}. The default AR mask has become key-dependent again — check "
    "that scoring/score.py still defaults to full_context_ar_mask rather than deriving the "
    "mask from a freshly drawn decoding order."
  )
