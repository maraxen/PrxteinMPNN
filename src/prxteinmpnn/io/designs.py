"""Efficient storage of generated designs using ArrayRecord with zero-copy binary serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import jax
import numpy as np
from array_record.python import array_record_module

if TYPE_CHECKING:
  import jax.numpy as jnp

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
  sequence: Any  # jnp.ndarray (uint8), shape (n_canonical,)
  logits: Any    # jnp.ndarray (float32), shape (n_canonical, 21)
  scores: Any    # jnp.ndarray (float32)
  state_weights: Any  # jnp.ndarray (float32)
  metadata: DesignMetadata

class DesignArrayRecordWriter:
  """Writer for storing designs in compressed ArrayRecord format using zero-copy binary serialization."""

  def __init__(self, path: str, options: str = "zstd:9,group_size:1",
               n_canonical: int = 214, n_states: int = 9):
    """Initialize the writer.

    Args:
      path: Path to the output ArrayRecord file.
      options: ArrayRecord writer options (e.g. compression).
      n_canonical: Number of canonical residues (for shape validation).
      n_states: Number of states (for shape validation).
    """
    self.path = path
    self.writer = array_record_module.ArrayRecordWriter(path, options)
    self.n_canonical = n_canonical
    self.n_states = n_states

    # Binary schema: field_name -> (shape, dtype)
    # Flattens individual designs to fixed-size records
    self.schema = {
        "sequence": (n_canonical,),      # uint8
        "logits": (n_canonical, 21),     # float32
        "scores": (1,),                  # float32
        "state_weights": (n_states,)     # float32
        # metadata (task_id, model, etc) stored as suffix of logits
    }

  def write(self, payload: DesignPayload):
    """Serialize and write a design payload using zero-copy binary format."""
    record_bytes = bytearray()

    # 1. Write sequence (uint8)
    seq = np.asarray(jax.device_get(payload["sequence"]), dtype=np.uint8)
    assert seq.shape == (self.n_canonical,), f"sequence shape {seq.shape} != {(self.n_canonical,)}"
    record_bytes.extend(seq.tobytes())

    # 2. Write logits (float32)
    logits = np.asarray(jax.device_get(payload["logits"]), dtype=np.float32)
    assert logits.shape == (self.n_canonical, 21), f"logits shape {logits.shape} != {(self.n_canonical, 21)}"
    record_bytes.extend(logits.tobytes())

    # 3. Write scores (float32)
    scores = np.asarray(jax.device_get(payload["scores"]), dtype=np.float32).flatten()
    assert scores.shape == (1,), f"scores shape {scores.shape} != (1,)"
    record_bytes.extend(scores.tobytes())

    # 4. Write state_weights (float32)
    weights = np.asarray(jax.device_get(payload["state_weights"]), dtype=np.float32)
    assert weights.shape == (self.n_states,), f"weights shape {weights.shape} != {(self.n_states,)}"
    record_bytes.extend(weights.tobytes())

    # 5. Write metadata as JSON suffix (model, task_id, etc)
    import json
    metadata_json = json.dumps(payload["metadata"]).encode("utf-8")
    metadata_len = np.uint32(len(metadata_json)).tobytes()
    record_bytes.extend(metadata_len)
    record_bytes.extend(metadata_json)

    self.writer.write(bytes(record_bytes))

  def close(self):
    """Close the writer and flush all data to disk."""
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
