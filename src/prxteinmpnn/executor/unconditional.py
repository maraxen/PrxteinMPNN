"""UnconditionalExecutor: executor for unconditional scoring with StageSet support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prxteinmpnn.executor.base import Executor
import jax.numpy as jnp

from prxteinmpnn.executor.base import Executor
from prxteinmpnn.model_inputs import UnconditionalInputs, StackInputs
from prxteinmpnn.pipeline_registry import StageSet
from prxteinmpnn.model._inference.scoring import score_unconditional
if TYPE_CHECKING:
  pass


class UnconditionalExecutor(Executor):
  """Executor wrapping UnconditionalPipeline with StageSet support.

  Coordinates unconditional sequence scoring over a multistate stack,
  resolving all pipeline stages (featurize, encode, decode, logit_transform)
  from the stored StageSet.

  Inputs:  ProteinBundle (stacked backbone geometry)
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
    inputs: Any,  # ProteinBundle
    stage_set: StageSet | None = None,
    **kwargs: Any,
  ) -> tuple[Any, Any]:
    """Run unconditional scoring and return (combined_logits, state_logits).

    Args:
      module: Model instance with score_unconditional_from_payload method.
      key: JAX PRNG key.
      inputs: ProteinBundle with stacked backbone geometry.
      stage_set: Optional StageSet to override the one from __init__.
                 If None, uses the StageSet from initialization.
      **kwargs: Additional keyword arguments passed to the pipeline.

    Returns:
      (logits, state_logits) where logits is (L, V) and state_logits is (S, L, V).
    """
    # Use provided stage_set or fall back to instance's stage_set
    if stage_set is None:
      stage_set = self._stage_set

    resolved = stage_set.resolve_all()
    logit_transform_fn = resolved["logit_transform_fn"]
    encoder_state_fn = resolved["encoder_state_fn"]

    captured_state_logits: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured_state_logits.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    state_weights = jnp.ones(inputs.n_states, dtype=jnp.float32) / inputs.n_states
    
    # Construct the PyTree payload
    stack_inputs = StackInputs(
        coords=inputs.coords,
        mask=inputs.mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        n_states=inputs.n_states,
    )
    uncond_inputs = UnconditionalInputs(state_stack=stack_inputs)

    logits = score_unconditional(
      module,
      key,
      uncond_inputs,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
      encoder_state_fn=encoder_state_fn,
    )
    
    state_logits = captured_state_logits[0] if captured_state_logits else None
    return logits, state_logits
