"""ConditionalPipeline: teacher-forced conditional scoring over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ConditionalInputs(eqx.Module):
  """Inputs for ConditionalPipeline.

  stack: MultistateStackPayload with backbone geometry.
  seq_oh_stack: one-hot sequences per state, shape (S, L, 21).
  ar_mask_stack: autoregressive mask per state, shape (S, L, L).
  """

  stack: Any
  seq_oh_stack: Float[Array, ...]
  ar_mask_stack: Float[Array, ...]


@dataclasses.dataclass(frozen=True)
class ConditionalPipeline:
  """Wraps score_conditional_from_payload with PipelineFns hooks.

  Inputs:  ConditionalInputs
  Outputs: (logits: (L, V), state_logits: (S, L, V))
  """

  multi_state_strategy_idx: int = 0
  inference: bool = True

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: ConditionalInputs,
    *,
    fns: Any,
  ) -> tuple[Any, Any]:
    """Run conditional scoring and return (combined_logits, state_logits)."""
    logit_transform_fn = fns.resolve_logit_transform()
    encoder_state_fn = fns.resolve_encoder_state_fn()
    captured: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S

    logits = module.score_conditional_from_payload(
      key,
      inputs.stack,
      seq_oh_stack=inputs.seq_oh_stack,
      ar_mask_stack=inputs.ar_mask_stack,
      tie_group_map=None,
      multi_state_strategy_idx=self.multi_state_strategy_idx,

      state_weights=state_weights,
      state_mapping=None,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
      encoder_state_fn=encoder_state_fn,
    )
    state_logits = captured[0] if captured else None
    return logits, state_logits
