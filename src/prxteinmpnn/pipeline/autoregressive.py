"""AutoregressivePipeline: temperature-sampled AR sequence design over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from prxteinmpnn.model_inputs import AutoregressiveInputs, StageSet


@dataclasses.dataclass(frozen=True)
class AutoregressivePipeline:
  """Wraps sample_autoregressive_from_payload with StageSet hooks.

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
    stage_set: StageSet,
  ) -> tuple[Any, Any]:
    """Sample sequences autoregressively and return (sequences, logits)."""
    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S
    resolved = stage_set.resolve_all()
    ar_logit_transform_fn = resolved["ar_logit_transform_fn"]

    sequences, logits = module.sample_autoregressive_from_payload(
      key,
      inputs.stack,
      inputs.autoregressive_mask,
      inputs.bias,
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
