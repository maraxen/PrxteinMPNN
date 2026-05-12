"""ConditionalExecutor: executor for conditional/teacher-forced scoring with StageSet support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prxteinmpnn.executor.base import Executor
from prxteinmpnn.pipeline.conditional import ConditionalPipeline
from prxteinmpnn.pipeline_registry import StageSet

if TYPE_CHECKING:
  pass


class ConditionalExecutor(Executor):
  """Executor wrapping ConditionalPipeline with StageSet support.

  Coordinates teacher-forced conditional sequence scoring over a multistate stack,
  resolving all pipeline stages (featurize, encode, decode, logit_transform)
  from the stored StageSet.

  Inputs:  ConditionalInputs with stack (MultistateStackPayload),
           seq_oh_stack (S, L, 21), ar_mask_stack (S, L, L)
  Outputs: (logits: (L, V), state_logits: (S, L, V))
  """

  def __init__(
    self,
    stage_set: StageSet | None = None,
    multi_state_strategy_idx: int = 0,
    inference: bool = True,
  ) -> None:
    """Initialize conditional executor.

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
    inputs: Any,  # ConditionalInputs
    stage_set: StageSet | None = None,
    **kwargs: Any,
  ) -> tuple[Any, Any]:
    """Run conditional scoring and return (combined_logits, state_logits).

    Args:
      module: Model instance with score_conditional_from_payload method.
      key: JAX PRNG key.
      inputs: ConditionalInputs with stack and sequence information.
      stage_set: Optional StageSet to override the one from __init__.
                 If None, uses the StageSet from initialization.
      **kwargs: Additional keyword arguments passed to the pipeline.

    Returns:
      (logits, state_logits) where logits is (L, V) and state_logits is (S, L, V).
    """
    # Use provided stage_set or fall back to instance's stage_set
    if stage_set is None:
      stage_set = self._stage_set

    # Resolve all stages from StageSet UIDs via registry
    stages = stage_set.resolve_all()

    # Create pipeline and delegate
    pipeline = ConditionalPipeline(
      multi_state_strategy_idx=self.multi_state_strategy_idx,
      inference=self.inference,
    )

    # Create a fake PipelineFns-like object that resolve_* methods can use
    # This bridges the old PipelineFns API with StageSet
    class _FnsAdapter:
      def __init__(self, stages_dict: dict[str, Any]):
        self._stages = stages_dict

      def resolve_logit_transform(self) -> Any:
        return self._stages["logit_transform_fn"]

      def resolve_encoder_state_fn(self) -> Any:
        return self._stages["encoder_state_fn"]

    fns_adapter = _FnsAdapter(stages)
    return pipeline(module, key, inputs, fns=fns_adapter, **kwargs)
