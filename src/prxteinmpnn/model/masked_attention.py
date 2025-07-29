"""Masked attention functions."""

import enum

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.types import AttentionMask, Message


class MaskedAttentionEnum(enum.Enum):
  """Enum for different types of masked attention."""

  NONE = "none"
  CROSS = "cross"
  CONDITIONAL = "conditional"


@jax.jit
def mask_attention(message: Message, attention_mask: AttentionMask) -> Message:
  """Apply attention mask to the message.

  Args:
    message: The message to be masked.
    attention_mask: The attention mask to apply.

  Returns:
    The masked message.

  """
  return jnp.expand_dims(attention_mask, -1) * message
