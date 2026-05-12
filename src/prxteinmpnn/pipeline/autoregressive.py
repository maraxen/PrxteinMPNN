"""AutoregressivePipeline: temperature-sampled AR sequence design over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class AutoregressiveInputs(eqx.Module):
  """Inputs for AutoregressivePipeline.

  stack: MultistateStackPayload with backbone geometry.
  wave: WaveParallelPayload with wave-parallel decode schedule.
  autoregressive_mask_stack: (S, L, L) AR mask per state.
  bias_stack: (S, L, 21) logit bias per state.
  """

  stack: Any
  wave: Any  # WaveParallelPayload
  autoregressive_mask_stack: Float[Array, ...]
  bias_stack: Float[Array, ...]


@dataclasses.dataclass(frozen=True)
class AutoregressivePipeline:
  """Wraps sample_autoregressive_from_payload with PipelineFns hooks.

  Inputs:  AutoregressiveInputs
  Outputs: (sequences: OneHotProteinSequence, logits: Logits)
  """

  temperature: float = 1.0
  multi_state_strategy_idx: int = 0

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: AutoregressiveInputs,
    *,
    fns: Any,
  ) -> tuple[Any, Any]:
    """Sample sequences autoregressively and return (sequences, logits)."""
    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S
    batch_fn = fns.resolve_logit_transform()
    ar_logit_transform_fn = fns.resolve_ar_logit_transform()

    sequences, logits = module.sample_autoregressive_from_payload(
      key,
      inputs.stack,
      inputs.autoregressive_mask_stack,
      inputs.bias_stack,
      self.temperature,
      self.multi_state_strategy_idx,

      state_weights,
      inputs.wave.wave_group_ids,
      inputs.wave.wave_group_positions,
      inputs.wave.wave_group_valid,
      inputs.wave.wave_position_valid,
      ar_logit_transform_fn=ar_logit_transform_fn,
    )
    return sequences, logits
