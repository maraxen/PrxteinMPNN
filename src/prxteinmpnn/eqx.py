"""Equinox-based neural network modules for PrxteinMPNN.

This module contains Equinox implementations of the core neural network
components used in ProteinMPNN, enabling a more modular and composable
architecture.

prxteinmpnn.eqx
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

if TYPE_CHECKING:
  from jaxtyping import Array

LayerNorm = eqx.nn.LayerNorm


STANDARD_EPSILON = 1e-5


class DenseLayer(eqx.Module):
  """Two-layer feedforward network with GeLU activation.

  This implements a standard feedforward block used in transformers:
    output = Linear_out(GeLU(Linear_in(x)))

  Attributes:
    linear_in: First linear transformation.
    linear_out: Second linear transformation.

  """

  linear_in: eqx.nn.Linear
  linear_out: eqx.nn.Linear

  def __init__(
    self,
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize DenseLayer module.

    Args:
      in_features: Input feature dimension.
      hidden_features: Hidden layer dimension.
      out_features: Output feature dimension.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> dense = DenseLayer(128, 512, 128, key=key)
      >>> x = jax.random.normal(jax.random.PRNGKey(1), (10, 128))
      >>> y = dense(x)

    """
    key_in, key_out = jax.random.split(key)
    self.linear_in = eqx.nn.Linear(in_features, hidden_features, key=key_in)
    self.linear_out = eqx.nn.Linear(hidden_features, out_features, key=key_out)

  def __call__(self, x: Array) -> Array:
    """Apply dense layer to input.

    Args:
      x: Input tensor of shape (..., in_features).

    Returns:
      Output tensor of shape (..., out_features).

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> dense = DenseLayer(128, 512, 128, key=key)
      >>> x = jax.random.normal(jax.random.PRNGKey(1), (10, 128))
      >>> y = dense(x)
      >>> y.shape
      (10, 128)

    """
    x = self.linear_in(x)
    x = jax.nn.gelu(x, approximate=False)  # Match functional API
    return self.linear_out(x)
