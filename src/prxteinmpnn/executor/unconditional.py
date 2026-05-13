"""UnconditionalExecutor: executor for unconditional scoring with StageSet support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prxteinmpnn.executor.base import Executor
from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
from prxteinmpnn.pipeline_registry import StageSet

if TYPE_CHECKING:
  pass


class UnconditionalExecutor(Executor):
  """Executor wrapping UnconditionalPipeline with StageSet support.

  Coordinates unconditional sequence scoring over a multistate stack,
  resolving all pipeline stages (featurize, encode, decode, logit_transform)
  from the stored StageSet.

  Inputs:  MultistateStackPayload (stacked backbone geometry)
  Outputs: (logits: (L, V), state_logits: (S, L, V))
           where logits = logit_transform_fn(state_logits, state_index, state_weights)
           and state_logits is the raw per-state encoder/decoder output.
  """

  def __init__(
    self,
    stage_set: StageSet | None = None,
    multi_state_strategy_idx: int = 0,
    inference: bool = True,
  ) -> None:
    """Initialize unconditional executor.

    Args:
      stage_set: StageSet containing UID strings for all pipeline stages.
                 Defaults to StageSet.default() if None.
      multi_state_strategy_idx: Strategy index for multistate routing.
      inference: Whether to run in inference mode (no gradients).
    """
    if stage_set is None:
      stage_set = StageSet.default()
    super().__init__(stage_set)
    self.multi_state_strategy_idx = multi_state_strategy_idx
    self.inference = inference

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: Any,  # MultistateStackPayload
    stage_set: StageSet | None = None,
    **kwargs: Any,
  ) -> tuple[Any, Any]:
    """Run unconditional scoring and return (combined_logits, state_logits).

    Args:
      module: Model instance with score_unconditional_from_payload method.
      key: JAX PRNG key.
      inputs: MultistateStackPayload with stacked backbone geometry.
      stage_set: Optional StageSet to override the one from __init__.
                 If None, uses the StageSet from initialization.
      **kwargs: Additional keyword arguments passed to the pipeline.

    Returns:
      (logits, state_logits) where logits is (L, V) and state_logits is (S, L, V).
    """
    # Use provided stage_set or fall back to instance's stage_set
    if stage_set is None:
      stage_set = self._stage_set

    # Create pipeline and delegate
    pipeline = UnconditionalPipeline(
      multi_state_strategy_idx=self.multi_state_strategy_idx,
      inference=self.inference,
    )

    return pipeline(module, key, inputs, stage_set=stage_set, **kwargs)
