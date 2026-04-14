"""Efficient storage of generated designs using ArrayRecord and Msgpack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import jax
import msgpack
import msgpack_numpy as m
from array_record.python import array_record_module

if TYPE_CHECKING:
  import jax.numpy as jnp

# Patch msgpack for numpy support
m.patch()

class DesignMetadata(TypedDict):
  """Metadata for a single design."""
  pool_type: Literal["BackboneOnly", "BackboneLigand", "BackboneSidechain", "FullContext"]
  state_mapping: list[int]
  weight_strategy: str
  combination_algorithm: str
  structure_ids: list[str]
  parent_structure_idx: int

class DesignPayload(TypedDict):
  """Serialized design payload."""
  sequence: Any  # jnp.ndarray (int8)
  logits: Any    # jnp.ndarray (float16)
  scores: Any    # jnp.ndarray (float32)
  state_weights: Any  # jnp.ndarray (float32)
  metadata: DesignMetadata

class DesignArrayRecordWriter:
  """Writer for storing designs in compressed ArrayRecord format."""

  def __init__(self, path: str, options: str = "zstd:9,group_size:1"):
    """Initialize the writer.
    
    Args:
      path: Path to the output ArrayRecord file.
      options: ArrayRecord writer options (e.g. compression).
    """
    self.path = path
    self.writer = array_record_module.ArrayRecordWriter(path, options)

  def write(self, payload: DesignPayload):
    """Serialize and write a design payload."""
    # Convert JAX arrays to numpy for msgpack-numpy compatibility
    processed_payload = {}
    for k, v in payload.items():
      if hasattr(v, "__jax_array__") or str(type(v)).find("jax") != -1:
        processed_payload[k] = jax.device_get(v)
      else:
        processed_payload[k] = v

    serialized = msgpack.packb(processed_payload, default=m.encode)
    self.writer.write(serialized)

  def close(self):
    """Close the writer."""
    self.writer.close()

def stream_design_to_host(
  writer: DesignArrayRecordWriter,
  sequence: jnp.ndarray,
  logits: jnp.ndarray,
  scores: jnp.ndarray,
  state_weights: jnp.ndarray,
  metadata: DesignMetadata,
):
  """Host-side callback for streaming designs from device.
  
  This function should be called via jax.experimental.io_callback.
  """
  payload: DesignPayload = {
    "sequence": sequence,
    "logits": logits,
    "scores": scores,
    "state_weights": state_weights,
    "metadata": metadata,
  }
  writer.write(payload)

def get_io_callback_fn(writer: DesignArrayRecordWriter):
  """Returns a JAX-compatible callback function for io_callback."""

  def callback(sequence, logits, scores, state_weights, **kwargs):
    # Note: metadata might need careful handling as it's not a JAX array
    # We can pass metadata fields through kwargs if they are constant
    metadata = kwargs.get("metadata", {})
    stream_design_to_host(writer, sequence, logits, scores, state_weights, metadata)

  return callback
