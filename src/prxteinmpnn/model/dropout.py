"""Custom Dropout module with dynamic probability for leaf alignment."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

class Dropout(eqx.Module):
  """Dropout layer with dynamic probability `p`.
  
  Unlike `equinox.nn.Dropout`, this module treats the dropout probability `p`
  as a dynamic leaf. This is required for alignment with legacy checkpoints
  where `p` was serialized as a tensor.
  """
  p: Array
  inference: Array

  def __init__(self, p: float, inference: bool = True) -> None:
    self.p = jnp.array(p)
    self.inference = jnp.array(inference)

  def __call__(
    self, 
    x: Array, 
    *, 
    key: PRNGKeyArray | None = None, 
    inference: bool | None = None
  ) -> Array:
    """Apply dropout to the input.
    
    Args:
      x: Input array.
      key: PRNG key for random mask generation. Required if `inference` is False.
      inference: Whether to run in inference mode (no dropout). 
        Defaults to `self.inference`.
        
    Returns:
      The array with dropout applied.
    """
    if inference is None:
      inference = self.inference

    # Handle both scalar and array inference flags
    inference_bool = jnp.asarray(inference, dtype=jnp.bool_)
      
    # Bernoulli mask
    if key is None:
        # Fallback for inference=True or missing key
        return x
        
    mask = jax.random.bernoulli(key, 1.0 - self.p, x.shape)
    dropped_x = jnp.where(mask, x / (1.0 - self.p), 0.0)
    
    return jnp.where(inference_bool, x, dropped_x)
