"""UnconditionalPipeline: unconditional sequence scoring over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class UnconditionalPipeline:
  """Wraps score_unconditional_state_vmap_exact_from_payload with PipelineFns hooks.

  Inputs:  MultistateStackPayload (stacked backbone geometry)
  Outputs: (logits: (L, V), state_logits: (S, L, V))
           where logits = logit_transform_fn(state_logits, state_index, state_weights)
           and state_logits is the raw per-state encoder/decoder output.
  """

  multi_state_strategy_idx: int = 0
  inference: bool = True

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: Any,  # MultistateStackPayload
    *,
    fns: Any,  # PipelineFns
  ) -> tuple[Any, Any]:
    """Run unconditional scoring and return (combined_logits, state_logits).

    Returns:
      (logits, state_logits) where logits is (L, V) and state_logits is (S, L, V).
    """
    logit_transform_fn = fns.resolve_logit_transform()

    captured_state_logits: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured_state_logits.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    state_weights = jnp.ones(inputs.n_states, dtype=jnp.float32) / inputs.n_states
    logits = module.score_unconditional_state_vmap_exact_from_payload(
      key,
      inputs,
      tie_group_map=None,
      multi_state_strategy_idx=self.multi_state_strategy_idx,
      multi_state_temperature=1.0,
      state_weights=state_weights,
      state_mapping=None,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
    )
    state_logits = captured_state_logits[0] if captured_state_logits else None
    return logits, state_logits
