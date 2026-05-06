"""Host-side scoring orchestration (Phase **5f**), mirroring :class:`~prxteinmpnn.run.sampling_driver.SamplingDriver`.

Centralizes ``make_score_fn`` wiring so ``run/scoring.py`` stays aligned with registry-ready patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from prxteinmpnn.scoring.score import make_score_fn

if TYPE_CHECKING:
  from prxteinmpnn.run.specs import ScoringSpecification


class ScoringDriver:
  """Thin factory around :func:`~prxteinmpnn.scoring.score.make_score_fn` bound to a spec."""

  def __init__(self, spec: ScoringSpecification) -> None:
    self.spec = spec

  def build_score_single_pair(self, model: object) -> Callable[..., Any]:
    """Return ``partial(make_score_fn(model), multi_state_*=...)`` for batched scoring vmaps."""
    return partial(
      make_score_fn(model=model),
      multi_state_strategy=self.spec.multi_state_strategy,
      multi_state_temperature=self.spec.multi_state_temperature,
    )
