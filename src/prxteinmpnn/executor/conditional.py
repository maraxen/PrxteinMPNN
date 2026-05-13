"""ConditionalExecutor: executor for conditional/teacher-forced scoring with StageSet support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from prxteinmpnn.executor.base import Executor
from prxteinmpnn.model_inputs import ConditionalInputs, StackInputs
from prxteinmpnn.pipeline_registry import StageSet
from prxteinmpnn.model._inference.scoring import score_conditional

if TYPE_CHECKING:
  pass


class ConditionalExecutor(Executor):
  """Executor wrapping ConditionalPipeline with StageSet support.

  Coordinates teacher-forced conditional sequence scoring over a multistate stack,
  resolving all pipeline stages (featurize, encode, decode, logit_transform)
  from the stored StageSet.

  Inputs:  ConditionalInputs with stack (ProteinBundle),
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

    resolved = stage_set.resolve_all()
    logit_transform_fn = resolved["logit_transform_fn"]
    encoder_state_fn = resolved["encoder_state_fn"]

    captured: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    # In legacy payload, inputs is a dict-like or legacy structure. We need to build ConditionalInputs.
    # Assuming inputs provides stack attributes: coords, mask, residue_index, chain_index, n_states
    # as well as seq_oh_stack and ar_mask_stack.
    S = inputs.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S
    
    stack_inputs = StackInputs(
        coords=inputs.coords,
        mask=inputs.mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        n_states=inputs.n_states,
    )
    
    # Check for bias_stack; default to None if missing.
    bias_stack = getattr(inputs, "bias_stack", None)
    
    cond_inputs = ConditionalInputs(
        state_stack=stack_inputs,
        seq_oh_stack=inputs.seq_oh_stack,
        ar_mask_stack=inputs.ar_mask_stack,
        bias_stack=bias_stack,
    )

    logits = score_conditional(
      module,
      key,
      cond_inputs,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
      encoder_state_fn=encoder_state_fn,
    )
    
    state_logits = captured[0] if captured else None
    return logits, state_logits
