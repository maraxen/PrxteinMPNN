"""Base abstract class for conditional decode paths.

Provides common interface and shared helpers for decode modes that consume
a ConditioningBundle (i.e., teacher-forced conditional decoding).

Risk D-7 mitigation: _ConditionalDecodeBase mediates coupling between
ConditionalDecode and STEDecode implementations.
"""

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from prxteinmpnn.types.bundles import EncoderOutput, InferenceBundle
from prxteinmpnn.types.stages import StageSet


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
