"""UnconditionalDecode mode class.

Unconditional scoring without sequence conditioning.
Vmaps over state axis using injected MapIterator strategy.

Risk D-1 mitigation: shared kernel logic in _kernel.py, mode class owns
only iterator orchestration.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.inference.decode._kernel import _project_logits
from prxteinmpnn.tiling.iterator import MapIterator
from prxteinmpnn.types.bundles import EncoderOutput, InferenceBundle
from prxteinmpnn.types.stages import StageSet


class UnconditionalDecode(eqx.Module):
  """Unconditional decode mode (no sequence conditioning).

  Decodes without teacher-forcing by vmapping decode_one over the state axis.
  The per-state decode function can be either stage_set.decode_step (if set)
  or model.decoder (fallback).

  Attributes
  ----------
  model : Any
      Model instance (static field).
  state_iterator : MapIterator
      Iterator for state-axis (Vmap or SafeMap).

  Notes
  -----
  Unlike ConditionalDecode, UnconditionalDecode does NOT consume sequence_oh.
  The per-state function receives only (node_features, edge_features,
  neighbor_indices, mask).

  This class is NOT a subclass of _ConditionalDecodeBase because unconditional
  decoding does not have the teacher-forcing contract.
  """

  model: Any = eqx.field(static=True)
  state_iterator: MapIterator

  def __call__(
    self,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    config: Any,
    stage_set: StageSet,
  ) -> jnp.ndarray:
    """Decode unconditionally (no sequence context).

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    bundle : InferenceBundle
        Inference bundle with conditioning data. For unconditional, only bias is used;
        sequence_oh and ar_mask are ignored.
    config : InferenceConfig
        Inference configuration.
    stage_set : StageSet
        Pipeline stages (decode_step, logit_transform).

    Returns
    -------
    jnp.ndarray
        Fused logits. Shape: (L, 21).
    """
    cond = bundle.conditioning

    # Define the per-state decode function (no sequence_oh)
    def decode_one(inputs):
      """Decode a single state without sequence conditioning.

      inputs: (node_features, edge_features, neighbor_indices, mask)
      """
      node_features, edge_features, neighbor_indices, mask = inputs
      if stage_set.decode_step is not None:
        return stage_set.decode_step(
          node_features,
          edge_features,
          neighbor_indices,
          mask,
          key=key,
          inference=config.inference,
        )
      # Fallback: use model.decoder (unconditional path)
      # Note: model.decoder() does not accept inference parameter
      return self.model.decoder(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        key=key,
      )

    # Bundle inputs as a single pytree for the iterator
    inputs = (enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask)

    # Vmap over state axis: (S, L, H) -> (S, L, H)
    decoded = self.state_iterator(decode_one, inputs, in_axes=0)

    # Project to logits: (S, L, H) -> (S, L, V)
    logits_stack = _project_logits(self.model, decoded)

    # Fuse across states: (S, L, V) -> (L, V)
    return self._apply_logit_transform(logits_stack, stage_set, bias=cond.bias)

  @staticmethod
  def _apply_logit_transform(
    logits: jnp.ndarray,
    stage_set: StageSet,
    bias: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    """Apply logit fusion via stage_set.logit_transform.

    Parameters
    ----------
    logits : ndarray
        Logits to fuse. Shape (S, L, V).
    stage_set : StageSet
        Contains logit_transform.
    bias : ndarray | None, default None
        Optional position-specific bias. Shape (L, V).

    Returns
    -------
    ndarray
        Fused logits. Shape (L, V).
    """
    if stage_set.logit_transform is None:
      # Identity: single-state passthrough
      return logits

    # logit_transform signature: (S, L, V) + (L, V) -> (L, V)
    return stage_set.logit_transform(logits, bias=bias)
