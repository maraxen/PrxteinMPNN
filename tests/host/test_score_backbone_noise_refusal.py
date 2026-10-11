"""Debt #2509: plain ``runner.score`` must not silently ignore ``backbone_noise``.

The plain per-structure path never forwards a noise level to the scorer and returns one score set
with no noise axis, so a nonzero ``backbone_noise`` there is refused with a ValueError (the
brief's "raise" option). The averaged-feature path is the one that applies noise and is exempt.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification

_STRUCTURE = "tests/data/1ubq.pdb"
_PROTEIN_CHECKPOINT = "proteinmpnn_v_48_020"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _spec(**kwargs: object) -> ScoringSpecification:
  return ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=_PROTEIN_CHECKPOINT,
    sequences_to_score=[_UBIQUITIN],
    max_length=128,
    return_logits=True,
    **kwargs,  # type: ignore[arg-type]
  )


def test_nonzero_backbone_noise_refused_on_plain_path() -> None:
  """Counterfactual: nonzero noise on the plain path raises before any model is loaded."""
  spec = _spec(backbone_noise=0.25)
  with pytest.raises(ValueError, match="average_node_features"):
    score(spec)


def test_nonzero_backbone_noise_level_in_sequence_refused() -> None:
  """A multi-level sweep with any nonzero level is refused too, not only a scalar."""
  spec = _spec(backbone_noise=(0.0, 0.1))
  with pytest.raises(ValueError, match="nonzero"):
    score(spec)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_zero_backbone_noise_plain_path_unchanged() -> None:
  """Positive regression pin: backbone_noise=0 gives the same scores as the default spec."""
  explicit = score(_spec(backbone_noise=0.0))
  default = score(_spec())
  np.testing.assert_array_equal(np.asarray(explicit["scores"]), np.asarray(default["scores"]))
  np.testing.assert_array_equal(np.asarray(explicit["logits"]), np.asarray(default["logits"]))
