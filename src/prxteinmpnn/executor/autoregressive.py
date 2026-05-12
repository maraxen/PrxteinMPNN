"""AutoregressiveExecutor: executor for autoregressive sequence sampling with StageSet support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prxteinmpnn.executor.base import Executor
from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline
from prxteinmpnn.pipeline_registry import StageSet

if TYPE_CHECKING:
  pass


class AutoregressiveExecutor(Executor):
  """Executor wrapping AutoregressivePipeline with StageSet support.

  Coordinates autoregressive temperature-sampled sequence design over a multistate
  stack, resolving all pipeline stages (featurize, encode, ar_logit_transform)
  from the stored StageSet.

  Inputs:  AutoregressiveInputs with stack (MultistateStackPayload),
           wave (WaveParallelPayload), autoregressive_mask_stack (S, L, L),
           bias_stack (S, L, 21)
  Outputs: (sequences: OneHotProteinSequence, logits: Logits)
  """

  def __init__(
    self,
    stage_set: StageSet | None = None,
    temperature: float = 1.0,
    multi_state_strategy_idx: int = 0,
  ) -> None:
    """Initialize autoregressive executor.

    Args:
      stage_set: StageSet containing UID strings for all pipeline stages.
                 Defaults to StageSet.default() if None.
      temperature: Temperature for Boltzmann sampling (< 1.0 = more deterministic).
      multi_state_strategy_idx: Strategy index for multistate routing.
    """
    if stage_set is None:
      stage_set = StageSet.default()
    super().__init__(stage_set)
    self.temperature = temperature
    self.multi_state_strategy_idx = multi_state_strategy_idx

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: Any,  # AutoregressiveInputs
    stage_set: StageSet | None = None,
    **kwargs: Any,
  ) -> tuple[Any, Any]:
    """Sample sequences autoregressively and return (sequences, logits).

    Args:
      module: Model instance with sample_autoregressive_from_payload method.
      key: JAX PRNG key.
      inputs: AutoregressiveInputs with stack, wave, mask, and bias information.
      stage_set: Optional StageSet to override the one from __init__.
                 If None, uses the StageSet from initialization.
      **kwargs: Additional keyword arguments passed to the pipeline.

    Returns:
      (sequences, logits) where sequences is the sampled OneHotProteinSequence
      and logits are the per-position logits during sampling.
    """
    # Use provided stage_set or fall back to instance's stage_set
    if stage_set is None:
      stage_set = self._stage_set

    # Resolve all stages from StageSet UIDs via registry
    stages = stage_set.resolve_all()

    # Create pipeline and delegate
    pipeline = AutoregressivePipeline(
      temperature=self.temperature,
      multi_state_strategy_idx=self.multi_state_strategy_idx,
    )

    # Create a fake PipelineFns-like object that resolve_* methods can use
    # This bridges the old PipelineFns API with StageSet
    class _FnsAdapter:
      def __init__(self, stages_dict: dict[str, Any]):
        self._stages = stages_dict

      def resolve_logit_transform(self) -> Any:
        return self._stages["logit_transform_fn"]

      def resolve_ar_logit_transform(self) -> Any:
        return self._stages["ar_logit_transform_fn"]

    fns_adapter = _FnsAdapter(stages)
    return pipeline(module, key, inputs, fns=fns_adapter, **kwargs)
