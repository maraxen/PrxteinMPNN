"""ConditionalDecode mode class (Task 7).

Conditional scoring kernel that performs teacher-forced decoding with
sequence context, iterating over states via a MapIterator, projecting
to logits, and fusing via stage_set.logit_transform.

Pattern 5 (injection): The state_iterator field is a MapIterator instance
(VmapIterator, SafeMapIterator, or other) injected at factory time. The
__call__ method contains zero branching on iterator type; it simply applies
the iterator to the per-state decoding function.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.inference.decode._base import _ConditionalDecodeBase
from aminx.inference.decode._kernel import (
  _decode_one_step,
  _project_logits,
)
from aminx.tiling.iterator import MapIterator
from aminx.types.arrays import Logits
from aminx.types.bundles import EncoderOutput, InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.stages import StageSet


class ConditionalDecode(_ConditionalDecodeBase):
  """Conditional decode mode: teacher-forced scoring with state iteration.

  Applies a MapIterator (Vmap, SafeMap) over the state (S) axis to perform
  conditional decoding. Each per-state call decodes with sequence context,
  projects to logits, and fuses across states.

  Parameters
  ----------
  model : Any
      Model instance with decoder and w_out attributes.
  state_iterator : MapIterator
      Iterator for the S axis (VmapIterator, SafeMapIterator, etc.).
      Injected at factory time; determines parallelism strategy.

  Attributes
  ----------
  model : Any
      The MPNN model. A dynamic field: its weight arrays are traced JAX
      leaves so filter_jit partitions them as runtime inputs (not hashed
      static constants).
  state_iterator : MapIterator
      State axis iterator. Allows strategy switching without changing __call__.

  Notes
  -----
  This class is the primary implementation of Pattern 5 (injection):
  the state_iterator field is a dependency injection point. Different
  MapIterator instances (Vmap, SafeMap) can be swapped at factory time
  without changing __call__ logic.

  The class is structured to match driver.py:_decode_conditional behavior
  exactly: decode per-state, project to logits, fuse via logit_transform.

  """

  model: Any
  state_iterator: MapIterator

  def __call__(
    self,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
  ) -> Logits:
    """Conditional decode: vmap over states, fuse logits.

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key for dropout/stochasticity.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    bundle : InferenceBundle
        Inference bundle with conditioning (sequence_oh, ar_mask, bias).
    config : InferenceConfig
        Inference configuration.
    stage_set : StageSet
        Pipeline stages with decode_step and logit_transform.

    Returns
    -------
    Logits
        Fused logits. Shape (L, 21).

    Notes
    -----
    Workflow:
    1. Broadcast sequence_oh to all S states.
    2. Build per_state_fn that decodes one state (calls _decode_one_step).
    3. Apply state_iterator over all S states.
    4. Project decoded features to logits via _project_logits.
    5. Apply logit_transform fusion to get final output.

    """
    S = enc.node_features.shape[0]
    cond = bundle.conditioning

    # Broadcast sequence one-hot to all states: (1, L, 21) -> (S, L, 21)
    seq_oh_stack = jnp.broadcast_to(
      cond.sequence_oh[None, ...],
      (S, *cond.sequence_oh.shape),
    )

    # Bundle per-state inputs as a pytree: each field is (S, ...)
    per_state_inputs = (
      enc.node_features,
      enc.edge_features,
      enc.neighbor_indices,
      enc.mask,
      cond.ar_mask,
      seq_oh_stack,
    )

    # Per-state decode closure: unpacks the pytree tuple
    def per_state_fn(inputs):
      node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh = inputs
      return _decode_one_step(
        model=self.model,
        node_features=node_features,
        edge_features=edge_features,
        neighbor_indices=neighbor_indices,
        mask=mask,
        ar_mask=ar_mask,
        sequence_oh=seq_oh,
        key=key,
        inference=config.inference,
        decode_step=stage_set.decode_step,
      )

    # Apply state_iterator to iterate over S axis
    # For vmap, this calls jax.vmap(per_state_fn, in_axes=0)(per_state_inputs)
    # which vmaps over axis 0 of each element of the tuple
    decoded = self.state_iterator(per_state_fn, per_state_inputs, in_axes=0)

    # Project to logits: (S, L, H) -> (S, L, V)
    logits_stack = _project_logits(self.model, decoded)

    # Fuse across states via logit_transform
    return self._apply_logit_transform(logits_stack, stage_set, bias=cond.bias)
