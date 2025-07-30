"""Layer normalization utilities.

prxteinmpnn.utils.normalize
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.utils.types import ModelParameters

STANDARD_EPSILON = 1e-5
ScaleConstant = Float[Array, "C"]  # Scale parameter for normalization
OffsetConstant = Float[Array, "C"]  # Offset parameter for normalization


@partial(jax.jit, static_argnames=("axis", "eps"))
def layer_normalization(
  x: Array,
  layer_parameters: ModelParameters,
  axis: int | Sequence[int] | None = -1,
  eps: float = STANDARD_EPSILON,
) -> Array:
  """Apply layer normalization to an input tensor in a functional manner.

  Args:
      x: The input tensor.
      layer_parameters: The layer parameters containing 'scale' and 'offset'.
      axis: The axis or axes to normalize over. Defaults to the last axis.
      eps: A small epsilon value to prevent division by zero.

  Returns:
      The normalized tensor.

  """
  scale = layer_parameters["norm"]["scale"]
  offset = layer_parameters["norm"]["offset"]
  return normalize(
    x,
    scale,
    offset,
    axis=axis,
    eps=eps,
  )


@partial(jax.jit, static_argnames=("axis", "eps"))
def normalize(
  x: Array,
  scale: ScaleConstant,
  offset: OffsetConstant,
  axis: int | Sequence[int] | None = -1,
  eps: float = STANDARD_EPSILON,
) -> Array:
  """Apply layer normalization to an input tensor in a functional manner.

  Args:
      x: The input tensor.
      scale: The learnable 'gamma' scaling factor.
      offset: The learnable 'beta' offset factor.
      axis: The axis or axes to normalize over. Defaults to the last axis.
      eps: A small epsilon value to prevent division by zero.

  Returns:
      The normalized tensor.

  """
  mean = jnp.mean(x, axis=axis, keepdims=True)
  variance = jnp.var(x, axis=axis, keepdims=True)

  x_normalized = (x - mean) / jnp.sqrt(variance + eps)

  return x_normalized * scale + offset
