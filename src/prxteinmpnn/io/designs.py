"""Efficient storage of generated designs using ArrayRecord with zero-copy binary serialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Self, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from array_record.python import array_record_module


def _to_numpy_uint8(x: jnp.ndarray | np.ndarray) -> np.ndarray:
  """Host-side snapshot to uint8; skips ``device_get`` when ``x`` is already NumPy."""
  if isinstance(x, np.ndarray):
    return np.asarray(x, dtype=np.uint8)
  return np.asarray(jax.device_get(x), dtype=np.uint8)


def _to_numpy_float32(x: jnp.ndarray | np.ndarray) -> np.ndarray:
  """Host-side snapshot to float32; skips ``device_get`` when ``x`` is already NumPy."""
  if isinstance(x, np.ndarray):
    return np.asarray(x, dtype=np.float32)
  return np.asarray(jax.device_get(x), dtype=np.float32)


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
  logits: Any    # jnp.ndarray (float32 input, cast to float16 for storage), shape (n_canonical, 21)
  scores: Any    # jnp.ndarray (float32)
  state_weights: Any  # jnp.ndarray (float32)
  metadata: DesignMetadata

class DesignArrayRecordWriter:
  """Writer for storing designs in compressed ArrayRecord format using zero-copy binary serialization.

  Note on logit precision:
    Logits are stored as float16 to halve storage and bandwidth. float16 preserves
    amino acid rank-ordering and softmax probabilities to <1% relative error for
    logit values in [-20, 20]. When reading, upcast to float32 before computing
    softmax or logsumexp to avoid float16 accumulation error:
        logits = record['logits'].astype(np.float32)
        probs = scipy.special.softmax(logits, axis=-1)
  """

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
        "logits": (n_canonical, 21),     # float16 — upcast to float32 before softmax/logsumexp (see class docstring)
        "scores": (1,),                  # float32
        "state_weights": (n_states,),     # float32
        # metadata (task_id, model, etc) stored as suffix of logits
    }

    # Async I/O: single-threaded executor to avoid blocking device on disk writes
    self._executor = ThreadPoolExecutor(max_workers=1)
    self._futures: list = []

  @classmethod
  def from_multistate_shapes(
    cls,
    path: str,
    *,
    n_canonical: int,
    n_states: int,
    options: str = "zstd:9,group_size:1",
  ) -> Self:
    """Writer sized like :class:`prxteinmpnn.payloads.MultistateStackPayload` static axes.

    ``n_canonical`` and ``n_states`` match the stack payload's ``n_canonical`` /
    ``n_states`` (roadmap §3.2) so ArrayRecord rows align with multistate campaigns.
    """
    return cls(path, options=options, n_canonical=n_canonical, n_states=n_states)

  def write(self, payload: DesignPayload) -> None:
    """Enqueue a design payload for async serialization and writing.

    Snapshot arrays from device when needed, then enqueue the serialization and disk I/O
    to avoid blocking the device. Payload fields that are already NumPy arrays skip an extra
    ``jax.device_get`` (matches ``run/sampling.py`` after ``np.asarray`` staging).

    TODO(io_callback integration): Prefer producers that hand off host arrays strictly via
    ``jax.experimental.io_callback`` (and ``jax.effects_barrier`` at batch boundaries); see
    ``prxteinmpnn/TODO_io_callback.txt``.
    """
    # 1. Snapshot sequence (uint8) — device_get is fast
    seq = _to_numpy_uint8(payload["sequence"])
    assert seq.shape == (self.n_canonical,), f"sequence shape {seq.shape} != {(self.n_canonical,)}"

    # 2. Snapshot logits (float32), validate range, then cast to float16
    # Store logits as float16. Bounded to [-20, 20] in practice; fits float16 range (±65504)
    # with 5 orders of magnitude headroom. Readers MUST upcast to float32 before softmax.
    logits_f32 = _to_numpy_float32(payload["logits"])
    assert logits_f32.shape == (self.n_canonical, 21), f"logits shape {logits_f32.shape} != {(self.n_canonical, 21)}"
    assert np.isfinite(logits_f32).all() and np.abs(logits_f32).max() < 1e4, \
        f"Logit values out of float16-safe range: max={np.abs(logits_f32).max():.1f}"
    logits = logits_f32.astype(np.float16)

    # 3. Snapshot scores (float32)
    scores = _to_numpy_float32(payload["scores"]).flatten()
    assert scores.shape == (1,), f"scores shape {scores.shape} != (1,)"

    # 4. Snapshot state_weights (float32)
    weights = _to_numpy_float32(payload["state_weights"])
    assert weights.shape == (self.n_states,), f"weights shape {weights.shape} != {(self.n_states,)}"

    # Enqueue serialization and disk write to thread pool
    future = self._executor.submit(self._write_sync, seq, logits, scores, weights, payload["metadata"])
    self._futures.append(future)

  def _write_sync(self, seq: np.ndarray, logits: np.ndarray, scores: np.ndarray,
                  weights: np.ndarray, metadata: DesignMetadata) -> None:
    """Synchronous serialization and disk write (runs in thread pool)."""
    record_bytes = bytearray()

    # Write sequence (uint8)
    record_bytes.extend(seq.tobytes())

    # Write logits (float16)
    record_bytes.extend(logits.tobytes())

    # Write scores (float32)
    record_bytes.extend(scores.tobytes())

    # Write state_weights (float32)
    record_bytes.extend(weights.tobytes())

    # Write metadata as JSON suffix
    import json
    metadata_json = json.dumps(metadata).encode("utf-8")
    metadata_len = np.uint32(len(metadata_json)).tobytes()
    record_bytes.extend(metadata_len)
    record_bytes.extend(metadata_json)

    self.writer.write(bytes(record_bytes))

  def close(self) -> None:
    """Wait for all pending async writes, then close the writer."""
    for future in self._futures:
      future.result()  # Propagate exceptions from worker threads
    self._executor.shutdown(wait=True)
    self.writer.close()

  def __enter__(self):
    """Context manager entry."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit: always close and wait for pending writes."""
    self.close()
    return False

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
