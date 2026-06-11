"""Base abstract class for conditional decode paths.

Provides common interface and shared helpers for decode modes that consume
a ConditioningBundle (i.e., teacher-forced conditional decoding).

Risk D-7 mitigation: _ConditionalDecodeBase mediates coupling between
ConditionalDecode and STEDecode implementations.
"""

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.types.bundles import EncoderOutput, InferenceBundle
from aminx.types.stages import StageSet


class _ConditionalDecodeBase(eqx.Module, abc.ABC):
  """Abstract base class for conditional decode implementations.

  Defines the interface for decode modes that perform teacher-forced decoding
  (sequence given as input, not sampled). Subclasses must implement __call__
  to wire state iteration and kernel logic.

  Parameters
  ----------
  (Subclass-specific; see ConditionalDecode, STEDecode, etc.)

  Notes
  -----
  Equinox modules and Python ABCs may have metaclass conflicts in some
  Python versions. This implementation uses ABC as a marker for documentation
  and IDE support, but does not enforce abstractness via metaclass magic.

  Subclasses should still implement the abstract method signature:
      __call__(self, key, enc, bundle, config, stage_set) -> Any

  See Also
  --------
  ConditionalDecode : Non-iterative conditional decode
  STEDecode : Straight-Through Estimator optimization over logits

  """

  @abc.abstractmethod
  def __call__(
    self,
    key: Any,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    config: Any,
    stage_set: StageSet,
  ) -> Any:
    """Decode with teacher forcing.

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key for dropout/stochasticity.
    enc : EncoderOutput
        Encoder output (node/edge features, indices, mask).
    bundle : InferenceBundle
        Full inference bundle; each mode extracts `bundle.conditioning`
        (and `bundle.wave` for AR) internally.
    config : InferenceConfig
        Inference configuration.
    stage_set : StageSet
        Pipeline stages (decode_step, logit_transform, etc.).

    Returns
    -------
    Any
        Decode result (typically shape (L, 21) logits or SampleResult).

    """
    ...

  @staticmethod
  def _apply_logit_transform(
    logits: jnp.ndarray,
    stage_set: StageSet,
    bias: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    """Apply logit fusion via stage_set.logit_transform.

    Shared helper for fusing per-state logits to unified output.

    Parameters
    ----------
    logits : ndarray
        Logits to fuse. Shape (S, L, V) for multi-state, (L, V) for single.
    stage_set : StageSet
        Contains logit_transform (may be None for identity).
    bias : ndarray | None, default None
        Optional position-specific bias. Shape (L, V).

    Returns
    -------
    ndarray
        Fused logits. Shape (L, V).

    """
    if stage_set.logit_transform is None:
      # Identity: single-state passthrough (already shape (L, V))
      return logits

    # logit_transform signature: (S, L, V) + (L, V) -> (L, V)
    return stage_set.logit_transform(logits, bias=bias)

  @staticmethod
  def _apply_tie_group_fuse(
    logits: jnp.ndarray,
    stage_set: StageSet,
    tie_group_map: jnp.ndarray,
  ) -> jnp.ndarray:
    """Fuse logits across tied positions so a tie group shares one distribution.

    Positions that share a tie-group id are fused via ``stage_set.tie_group_fuse``
    (product-of-experts: sum of log-softmax), matching the AR/STE tie semantics
    (autoregressive.py, ste.py). Singleton positions — a group of size one, which
    is the default when no tie groups are supplied (tie_group_map = arange) — pass
    through unchanged so the untied path is value-preserving.

    Parameters
    ----------
    logits : ndarray
        Per-position logits after state fusion. Shape (L, V).
    stage_set : StageSet
        Contains tie_group_fuse (no-op when None).
    tie_group_map : ndarray
        Group id per position. Shape (L,).

    Returns
    -------
    ndarray
        Logits with tied groups fused. Shape (L, V).

    """
    if stage_set.tie_group_fuse is None:
      return logits

    # (L, L): same_group[i, j] is True when positions i and j share a tie group.
    same_group = tie_group_map[:, None] == tie_group_map[None, :]
    group_size = jnp.sum(same_group, axis=1)  # (L,)

    # For each position i, fuse over its group members via the configured strategy.
    fused = jax.vmap(lambda member_mask: stage_set.tie_group_fuse(logits, member_mask))(
      same_group,
    )  # (L, V)

    # Only non-singleton groups are fused; singletons keep their raw logits.
    is_tied = group_size > 1
    return jnp.where(is_tied[:, None], fused, logits)
