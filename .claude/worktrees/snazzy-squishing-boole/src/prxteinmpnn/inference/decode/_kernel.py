"""Pure kernel helpers for decode paths.

Extracted from driver.py and optimize_ste.py for reusability and testing.
These functions contain zero side-effects and are suitable for JAX transforms
(jit, vmap, scan).
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


def _decode_one_step(
  model: Any,
  node_features: jnp.ndarray,
  edge_features: jnp.ndarray,
  neighbor_indices: jnp.ndarray,
  mask: jnp.ndarray,
  ar_mask: jnp.ndarray,
  sequence_oh: jnp.ndarray,
  key: PRNGKeyArray,
  inference: bool = True,
  decode_step: Callable | None = None,
) -> jnp.ndarray:
  """Decode inner body: conditional decoding of a single state (or batch).

  Pure helper extracted from driver.py:_decode_conditional inner body
  (lines 170-176). Can be vmapped over states.

  Parameters
  ----------
  model : Any
      Model instance with decoder and w_s_embed attributes.
  node_features : ndarray
      Shape (L, H) for single state, or (S, L, H) if vmapped.
  edge_features : ndarray
      Shape (L, K, H) for single state, or (S, L, K, H) if vmapped.
  neighbor_indices : ndarray
      Shape (L, K) for single state, or (S, L, K) if vmapped.
  mask : ndarray
      Shape (L,) for single state, or (S, L) if vmapped.
  ar_mask : ndarray
      Shape (L,) for single state, or (S, L) if vmapped.
  sequence_oh : ndarray
      Shape (L, 21) one-hot encoded sequence.
  key : PRNGKeyArray
      PRNG key for dropout (if used).
  inference : bool, default True
      Inference mode (disables dropout).
  decode_step : Callable | None, default None
      Optional override decode function. If None, uses model.decoder.call_conditional.

  Returns
  -------
  ndarray
      Decoded hidden features. Shape (L, H) per state.

  """
  if decode_step is not None:
    return decode_step(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      sequence_oh,
      key=key,
      inference=inference,
    )
  return model.decoder.call_conditional(
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    ar_mask,
    sequence_oh,
    model.w_s_embed.weight,
    inference=inference,
    key=key,
  )


def _project_logits(
  model: Any,
  decoded: jnp.ndarray,
) -> jnp.ndarray:
  """Project decoder hidden features to logits.

  Pure helper for logit projection: applies model.w_out projection.
  Extracted from driver.py:_decode_conditional line 187.

  Parameters
  ----------
  model : Any
      Model instance with w_out projection matrix/function.
  decoded : ndarray
      Decoded hidden features. Shape (S, L, H) or (L, H) for single state.

  Returns
  -------
  ndarray
      Logits. Shape (S, L, V) or (L, V) matching input batch structure.
      V = 21 (vocabulary size).

  """
  # Double vmap over (S, L) to apply w_out (which expects H -> V)
  return jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)


def _tied_group_einsum_average(
  logits: jnp.ndarray,
  tie_group_map: jnp.ndarray,
  num_groups: int,
) -> jnp.ndarray:
  """Average logits over tied positions (Risk D-2).

  Reproduces the einsum block from optimize_ste.py:197-207 character-for-character.
  Groups positions by tie_group_map and averages logits within each group.

  Parameters
  ----------
  logits : ndarray
      Shape (S, L, V) logits where L is position axis.
  tie_group_map : ndarray
      Shape (L,) group index for each position. Values in [0, num_groups).
  num_groups : int
      Total number of groups.

  Returns
  -------
  ndarray
      Averaged logits. Shape (S, L, V). Positions in the same group
      will have identical values.

  Notes
  -----
  The einsum block is:
  1. One-hot encode each position's group membership: (L, num_groups)
  2. Sum logits per group: einsum("ng,na->ga") -> (num_groups, V)
  3. Count positions per group and average
  4. Broadcast group averages back to positions: einsum("ng,ga->na")

  This operation is applied per-state (along S axis), preserving shape.

  """
  S = logits.shape[0]
  L = logits.shape[1]
  V = logits.shape[2]

  def average_one_state(state_logits: jnp.ndarray) -> jnp.ndarray:
    """Average a single state's logits (S, L, V) -> (L, V)."""
    # Character-for-character from optimize_ste.py:197-207
    group_one_hot = jax.nn.one_hot(
      tie_group_map,
      num_groups,
      dtype=jnp.float32,
    )

    group_logit_sums = jnp.einsum("ng,na->ga", group_one_hot, state_logits)
    group_counts = group_one_hot.sum(axis=0)
    group_avg_logits = group_logit_sums / (group_counts[:, None] + 1e-8)
    averaged = jnp.einsum("ng,ga->na", group_one_hot, group_avg_logits)

    return averaged

  # Vmap over S axis
  return jax.vmap(average_one_state, in_axes=0)(logits)
