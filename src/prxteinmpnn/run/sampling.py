"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.run.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)
from prxteinmpnn.utils.safe_map import safe_map as _safe_map

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from grain.python import IterDataset
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.data_structures import Protein
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneCoordinates,
    ChainIndex,
    DecodingOrder,
    Logits,
    ProteinSequence,
    ResidueIndex,
  )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

RANK_WITH_TEMPERATURE = 4
AMINO_ACID_VOCAB_SIZE = 21
LIGAND_PLACEHOLDER_ATOMS = 1
GRID_SCHEMA_VERSION = "grid_v1"
SAMPLING_SCHEMA_VERSION = "sampling_v1"
LIGAND_CONTEXT_KEYS = ("Y", "Y_t", "Y_m")


def _canonical_structure_id(input_item: Any, index: int) -> str:  # noqa: ANN401
  if isinstance(input_item, Path):
    source = str(input_item)
  elif isinstance(input_item, str):
    source = input_item
  else:
    source_name = getattr(input_item, "name", None)
    source = str(source_name) if source_name is not None else ""
  stem = Path(source).stem if source else ""
  if stem:
    return stem
  return f"structure_{index}"


def _canonical_structure_ids_for_spec(spec: SamplingSpecification) -> list[str]:
  raw_inputs = spec.inputs
  if isinstance(raw_inputs, (str, Path)) or hasattr(raw_inputs, "read"):
    input_items = [raw_inputs]
  else:
    try:
      input_items = list(raw_inputs)
    except TypeError:
      input_items = [raw_inputs]
  return [_canonical_structure_id(item, idx) for idx, item in enumerate(input_items)]


def _structure_ids_for_batch(
  canonical_ids: list[str],
  *,
  structure_offset: int,
  batch_size: int,
) -> list[str]:
  structure_ids: list[str] = []
  for local_idx in range(batch_size):
    global_idx = structure_offset + local_idx
    if global_idx < len(canonical_ids):
      structure_ids.append(canonical_ids[global_idx])
    else:
      structure_ids.append(f"structure_{global_idx}")
  return structure_ids


def _resolve_grid_lineage(spec: SamplingSpecification) -> dict[str, int | str] | None:
  if not spec.grid_mode:
    return None
  sample_count = int(spec.sample_count if spec.sample_count is not None else spec.num_samples)
  if sample_count <= 0:
    msg = "sample_count must be positive when grid_mode=True."
    raise ValueError(msg)
  sample_start = int(spec.sample_start if spec.sample_start is not None else 0)
  if sample_start < 0:
    msg = "sample_start must be non-negative when grid_mode=True."
    raise ValueError(msg)
  chunk_id = int(spec.chunk_id if spec.chunk_id is not None else 0)
  if chunk_id < 0:
    msg = "chunk_id must be non-negative when grid_mode=True."
    raise ValueError(msg)
  job_id = spec.job_id or f"grid_{spec.random_seed}"
  return {
    "job_id": job_id,
    "chunk_id": chunk_id,
    "sample_start": sample_start,
    "sample_count": sample_count,
  }


def _grid_sample_indices(lineage: dict[str, int | str]) -> np.ndarray:
  sample_start = int(lineage["sample_start"])
  sample_count = int(lineage["sample_count"])
  return np.arange(sample_start, sample_start + sample_count, dtype=np.int64)


def _grid_iteration_arrays(
  lineage: dict[str, int | str],
  *,
  chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  sample_start = int(lineage["sample_start"])
  sample_count = int(lineage["sample_count"])
  if chunk_size <= 0:
    msg = "samples_chunk_size must be positive when provided."
    raise ValueError(msg)
  iteration_ids: list[int] = []
  iteration_starts: list[int] = []
  iteration_counts: list[int] = []
  local_offset = 0
  while local_offset < sample_count:
    count = min(chunk_size, sample_count - local_offset)
    iteration_ids.append(len(iteration_ids))
    iteration_starts.append(sample_start + local_offset)
    iteration_counts.append(count)
    local_offset += count
  return (
    np.asarray(iteration_ids, dtype=np.int64),
    np.asarray(iteration_starts, dtype=np.int64),
    np.asarray(iteration_counts, dtype=np.int64),
  )


def _canonical_float_strings(values: Any) -> list[str]:  # noqa: ANN401
  return [format(float(value), ".17g") for value in values]


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def _grid_manifest_row_hash(
  spec: SamplingSpecification,
  lineage: dict[str, int | str],
) -> str:
  payload = {
    "schema_version": GRID_SCHEMA_VERSION,
    "job_id": str(lineage["job_id"]),
    "chunk_id": int(lineage["chunk_id"]),
    "sample_start": int(lineage["sample_start"]),
    "sample_count": int(lineage["sample_count"]),
    "model_family": spec.model_family,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "multi_state_strategy": spec.multi_state_strategy,
    "temperature": _canonical_float_strings(spec.temperature),
    "backbone_noise": _canonical_float_strings(spec.backbone_noise),
  }
  return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _grid_job_seed_hash(
  spec: SamplingSpecification,
  lineage: dict[str, int | str],
) -> str:
  payload = {
    "schema_version": GRID_SCHEMA_VERSION,
    "job_id": str(lineage["job_id"]),
    "model_family": spec.model_family,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "multi_state_strategy": spec.multi_state_strategy,
    "temperature": _canonical_float_strings(spec.temperature),
    "backbone_noise": _canonical_float_strings(spec.backbone_noise),
  }
  return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _seed_words_from_manifest_hash(manifest_row_hash: str) -> tuple[int, int, int, int]:
  digest = bytes.fromhex(manifest_row_hash)
  words = [
    int.from_bytes(digest[offset : offset + 4], byteorder="big", signed=False)
    for offset in range(0, 16, 4)
  ]
  return (words[0], words[1], words[2], words[3])


def _base_sampling_key(
  spec: SamplingSpecification,
  *,
  grid_lineage: dict[str, int | str] | None,
) -> jax.Array:
  key = jax.random.key(spec.random_seed)
  if grid_lineage is None:
    return key
  seed_hash = _grid_job_seed_hash(spec, grid_lineage)
  for seed_word in _seed_words_from_manifest_hash(seed_hash):
    key = jax.random.fold_in(key, seed_word)
  return key


def _split_ligand_payload_key(payload_key: str) -> tuple[str, str] | None:
  for tensor_name in LIGAND_CONTEXT_KEYS:
    for separator in ("::", "/"):
      suffix = f"{separator}{tensor_name}"
      if payload_key.endswith(suffix):
        structure_id = payload_key[: -len(suffix)]
        if structure_id:
          return structure_id, tensor_name
  return None


def _normalize_keyed_ligand_array(
  value: np.ndarray,
  *,
  expected_ndim: int,
  key: str,
) -> np.ndarray:
  arr = np.asarray(value)
  if arr.ndim == expected_ndim + 1 and arr.shape[0] == 1:
    arr = arr[0]
  if arr.ndim != expected_ndim:
    msg = f"Ligand payload entry '{key}' must have rank {expected_ndim}."
    raise ValueError(msg)
  return arr


def _broadcast_per_structure(
  value: Any,  # noqa: ANN401
  *,
  batch_size: int,
  expected_len: int,
  dtype: jnp.dtype,
  name: str,
) -> jax.Array:
  arr = jnp.asarray(value, dtype=dtype)
  if arr.ndim == 1:
    if arr.shape[0] != expected_len:
      msg = f"{name} length mismatch: expected {expected_len}, got {arr.shape[0]}"
      raise ValueError(msg)
    return jnp.broadcast_to(arr[None, :], (batch_size, expected_len))
  if arr.ndim == 2:
    if arr.shape[1] != expected_len:
      msg = f"{name} length mismatch: expected {expected_len}, got {arr.shape[1]}"
      raise ValueError(msg)
    if arr.shape[0] == 1 and batch_size > 1:
      return jnp.broadcast_to(arr, (batch_size, expected_len))
    if arr.shape[0] != batch_size:
      msg = f"{name} batch mismatch: expected {batch_size}, got {arr.shape[0]}"
      raise ValueError(msg)
    return arr
  msg = f"{name} must have rank 1 or 2, got rank {arr.ndim}"
  raise ValueError(msg)


def _normalize_ligand_tensor(
  value: Any,  # noqa: ANN401
  *,
  batch_size: int,
  seq_len: int,
  target_rank: int,
  dtype: jnp.dtype,
  name: str,
) -> jax.Array:
  arr = jnp.asarray(value, dtype=dtype)
  if target_rank == 4:
    if arr.ndim == 3:
      arr = jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))
    elif arr.ndim == 4 and arr.shape[0] == 1 and batch_size > 1:
      arr = jnp.broadcast_to(arr, (batch_size, *arr.shape[1:]))
  elif target_rank == 3:
    if arr.ndim == 2:
      arr = jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))
    elif arr.ndim == 3 and arr.shape[0] == 1 and batch_size > 1:
      arr = jnp.broadcast_to(arr, (batch_size, *arr.shape[1:]))
  if arr.ndim != target_rank:
    msg = f"{name} must have rank {target_rank}, got rank {arr.ndim}"
    raise ValueError(msg)
  if arr.shape[0] != batch_size:
    msg = f"{name} batch mismatch: expected {batch_size}, got {arr.shape[0]}"
    raise ValueError(msg)
  if arr.shape[1] != seq_len:
    msg = f"{name} sequence length mismatch: expected {seq_len}, got {arr.shape[1]}"
    raise ValueError(msg)
  return arr


def _load_ligand_context_file(
  ligand_context_path: Path,
  *,
  canonical_structure_ids: Sequence[str] | None,
  batch_structure_ids: Sequence[str] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  with np.load(ligand_context_path, allow_pickle=False) as npz_data:
    canonical_ids = (
      list(canonical_structure_ids)
      if canonical_structure_ids is not None
      else (list(batch_structure_ids) if batch_structure_ids is not None else None)
    )
    selected_ids = (
      list(batch_structure_ids)
      if batch_structure_ids is not None
      else (list(canonical_ids) if canonical_ids is not None else None)
    )
    keyed_payload: dict[str, dict[str, np.ndarray]] = {}
    parsed_keys: set[str] = set()
    for file_key in npz_data.files:
      split = _split_ligand_payload_key(file_key)
      if split is None:
        continue
      structure_id, tensor_name = split
      keyed_payload.setdefault(structure_id, {})[tensor_name] = np.asarray(npz_data[file_key])
      parsed_keys.add(file_key)

    if keyed_payload:
      canonical_ids = canonical_ids if canonical_ids is not None else sorted(keyed_payload)
      selected_ids = selected_ids if selected_ids is not None else list(canonical_ids)
      payload_ids = set(keyed_payload)
      canonical_id_set = set(canonical_ids)
      missing_ids = sorted(canonical_id_set - payload_ids)
      extra_ids = sorted(payload_ids - canonical_id_set)
      if missing_ids or extra_ids:
        msg = (
          "Ligand context keys must exactly match canonical structure IDs. "
          f"missing keys: {missing_ids}; extra keys: {extra_ids}."
        )
        raise ValueError(msg)
      unexpected_keys = sorted(set(npz_data.files) - parsed_keys)
      if unexpected_keys:
        msg = (
          f"Ligand context file '{ligand_context_path}' has unsupported keys: "
          f"{unexpected_keys}. Expected only <structure_id>::Y, <structure_id>::Y_t, "
          "<structure_id>::Y_m entries."
        )
        raise ValueError(msg)
      missing_tensor_keys = [
        f"{structure_id}:{tensor_name}"
        for structure_id in canonical_ids
        for tensor_name in LIGAND_CONTEXT_KEYS
        if tensor_name not in keyed_payload[structure_id]
      ]
      if missing_tensor_keys:
        msg = (
          f"Ligand context file '{ligand_context_path}' is missing keyed tensors: "
          f"{missing_tensor_keys}"
        )
        raise ValueError(msg)
      y_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y"],
          expected_ndim=3,
          key=f"{structure_id}::Y",
        )
        for structure_id in selected_ids
      ]
      y_t_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y_t"],
          expected_ndim=2,
          key=f"{structure_id}::Y_t",
        )
        for structure_id in selected_ids
      ]
      y_m_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y_m"],
          expected_ndim=2,
          key=f"{structure_id}::Y_m",
        )
        for structure_id in selected_ids
      ]
      return (
        jnp.asarray(np.stack(y_items, axis=0)),
        jnp.asarray(np.stack(y_t_items, axis=0)),
        jnp.asarray(np.stack(y_m_items, axis=0)),
      )

    missing = [key for key in (*LIGAND_CONTEXT_KEYS, "structure_ids") if key not in npz_data]
    if missing:
      msg = (
        f"Ligand context file '{ligand_context_path}' is missing required keys: "
        f"{', '.join(missing)}"
      )
      raise ValueError(msg)

    payload_ids = [str(item) for item in np.asarray(npz_data["structure_ids"]).tolist()]
    canonical_ids = canonical_ids if canonical_ids is not None else payload_ids
    selected_ids = selected_ids if selected_ids is not None else canonical_ids
    payload_id_set = set(payload_ids)
    canonical_id_set = set(canonical_ids)
    missing_ids = sorted(canonical_id_set - payload_id_set)
    extra_ids = sorted(payload_id_set - canonical_id_set)
    if missing_ids or extra_ids:
      msg = (
        "Ligand context keys must exactly match canonical structure IDs. "
        f"missing keys: {missing_ids}; extra keys: {extra_ids}."
      )
      raise ValueError(msg)

    id_to_index = {structure_id: idx for idx, structure_id in enumerate(payload_ids)}
    gather_indices = np.asarray([id_to_index[structure_id] for structure_id in selected_ids], dtype=np.int32)
    return (
      jnp.asarray(np.asarray(npz_data["Y"])[gather_indices]),
      jnp.asarray(np.asarray(npz_data["Y_t"])[gather_indices]),
      jnp.asarray(np.asarray(npz_data["Y_m"])[gather_indices]),
    )


def _prepare_ligand_context(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  batch_size: int,
  seq_len: int,
  canonical_structure_ids: Sequence[str] | None = None,
  batch_structure_ids: Sequence[str] | None = None,
) -> dict[str, jax.Array | None]:
  if spec.model_family != "ligandmpnn":
    return {
      "Y": None,
      "Y_t": None,
      "Y_m": None,
      "xyz_37": None,
      "xyz_37_m": None,
      "chain_mask": None,
    }

  Y = getattr(batched_ensemble, "Y", None)
  Y_t = getattr(batched_ensemble, "Y_t", None)
  Y_m = getattr(batched_ensemble, "Y_m", None)

  if spec.ligand_context_path is not None:
    file_Y, file_Y_t, file_Y_m = _load_ligand_context_file(
      spec.ligand_context_path,
      canonical_structure_ids=canonical_structure_ids,
      batch_structure_ids=batch_structure_ids,
    )
    Y = file_Y
    Y_t = file_Y_t
    Y_m = file_Y_m

  if Y is None or Y_t is None or Y_m is None:
    if spec.ligand_conditioning:
      msg = (
        "ligand_conditioning=True requires ligand context tensors (Y, Y_t, Y_m) "
        "on the batch or via ligand_context_path."
      )
      raise ValueError(msg)
    Y = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS, 3), dtype=jnp.float32)
    Y_t = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS), dtype=jnp.int32)
    Y_m = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS), dtype=jnp.float32)

  Y = _normalize_ligand_tensor(
    Y,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=4,
    dtype=jnp.float32,
    name="Y",
  )
  Y_t = _normalize_ligand_tensor(
    Y_t,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=3,
    dtype=jnp.int32,
    name="Y_t",
  )
  Y_m = _normalize_ligand_tensor(
    Y_m,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=3,
    dtype=jnp.float32,
    name="Y_m",
  )

  if not spec.sidechain_conditioning:
    return {
      "Y": Y,
      "Y_t": Y_t,
      "Y_m": Y_m,
      "xyz_37": None,
      "xyz_37_m": None,
      "chain_mask": None,
    }

  xyz_37 = jnp.asarray(batched_ensemble.coordinates, dtype=jnp.float32)
  if xyz_37.ndim != 4 or xyz_37.shape[0] != batch_size or xyz_37.shape[1] != seq_len:
    msg = "coordinates must have shape (batch, residues, atoms, xyz) for sidechain conditioning."
    raise ValueError(msg)

  xyz_37_m_source = (
    batched_ensemble.atom_mask
    if batched_ensemble.atom_mask is not None
    else batched_ensemble.full_atom_mask
  )
  if xyz_37_m_source is None:
    msg = "sidechain_conditioning=True requires atom_mask or full_atom_mask."
    raise ValueError(msg)
  xyz_37_m = jnp.asarray(xyz_37_m_source, dtype=jnp.float32)
  if xyz_37_m.ndim != 3 or xyz_37_m.shape[0] != batch_size or xyz_37_m.shape[1] != seq_len:
    msg = "atom mask must have shape (batch, residues, atoms) for sidechain conditioning."
    raise ValueError(msg)

  chain_mask = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)

  return {
    "Y": Y,
    "Y_t": Y_t,
    "Y_m": Y_m,
    "xyz_37": xyz_37,
    "xyz_37_m": xyz_37_m,
    "chain_mask": chain_mask,
  }


def _prepare_fixed_controls(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  tie_map_for_vmap: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None]:
  batch_size = batched_ensemble.coordinates.shape[0]
  seq_len = batched_ensemble.coordinates.shape[1]

  fixed_mask = (
    _broadcast_per_structure(
      spec.fixed_mask,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.bool_,
      name="fixed_mask",
    )
    if spec.fixed_mask is not None
    else None
  )

  fixed_tokens = (
    _broadcast_per_structure(
      spec.fixed_tokens,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.int32,
      name="fixed_tokens",
    )
    if spec.fixed_tokens is not None
    else None
  )

  if spec.fixed_positions is not None:
    positions = jnp.asarray(spec.fixed_positions, dtype=jnp.int32).reshape(-1)
    if positions.size > 0:
      if int(jnp.min(positions)) < 0 or int(jnp.max(positions)) >= seq_len:
        msg = f"fixed_positions must be in [0, {seq_len - 1}]"
        raise ValueError(msg)
      legacy_mask = jnp.zeros((seq_len,), dtype=jnp.bool_).at[positions].set(True)
      legacy_mask = jnp.broadcast_to(legacy_mask[None, :], (batch_size, seq_len))
    else:
      legacy_mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    if fixed_mask is None:
      fixed_mask = legacy_mask
    elif not bool(jnp.all(fixed_mask == legacy_mask)):
      msg = "fixed_positions and fixed_mask disagree; provide consistent values."
      raise ValueError(msg)

  if fixed_mask is None and fixed_tokens is not None:
    msg = "fixed_tokens was provided without fixed_mask/fixed_positions."
    raise ValueError(msg)

  if fixed_mask is not None and fixed_tokens is None:
    aatype = jnp.asarray(batched_ensemble.aatype, dtype=jnp.int32)
    if aatype.ndim == 1:
      aatype = jnp.broadcast_to(aatype[None, :], (batch_size, seq_len))
    fixed_tokens = aatype

  if fixed_mask is None or fixed_tokens is None:
    return None, None

  fixed_mask_np = np.asarray(fixed_mask, dtype=bool)
  fixed_tokens_np = np.asarray(fixed_tokens, dtype=np.int32)
  invalid_tokens = (fixed_tokens_np < 0) | (fixed_tokens_np >= AMINO_ACID_VOCAB_SIZE)
  if np.any(invalid_tokens & fixed_mask_np):
    msg = f"fixed_tokens must be in [0, {AMINO_ACID_VOCAB_SIZE - 1}] at masked positions."
    raise ValueError(msg)

  if tie_map_for_vmap is not None:
    tie_map_np = np.asarray(tie_map_for_vmap, dtype=np.int32)
    for b_idx in range(batch_size):
      masked_positions = np.where(fixed_mask_np[b_idx])[0]
      if masked_positions.size == 0:
        continue
      for group_id in np.unique(tie_map_np[b_idx, masked_positions]):
        in_group = tie_map_np[b_idx] == group_id
        in_group_and_fixed = in_group & fixed_mask_np[b_idx]
        if not np.any(in_group_and_fixed):
          continue
        group_tokens = np.unique(fixed_tokens_np[b_idx, in_group_and_fixed])
        if group_tokens.size > 1:
          msg = (
            "Tied group contains conflicting fixed tokens. "
            f"batch={b_idx}, group_id={group_id}, tokens={group_tokens.tolist()}"
          )
          raise ValueError(msg)

  return fixed_mask.astype(jnp.bool_), fixed_tokens.astype(jnp.int32)


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  sampler_fn: Callable,
  *,
  canonical_structure_ids: Sequence[str] | None = None,
  batch_structure_ids: Sequence[str] | None = None,
  chunk_sample_start: int | None = None,
  chunk_sample_count: int | None = None,
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
  """Sample sequences for a batched ensemble of proteins."""
  grid_lineage = _resolve_grid_lineage(spec)
  base_key = _base_sampling_key(spec, grid_lineage=grid_lineage)
  default_num_samples = (
    int(grid_lineage["sample_count"]) if grid_lineage is not None else int(spec.num_samples)
  )
  target_num_samples = (
    int(chunk_sample_count) if chunk_sample_count is not None else default_num_samples
  )
  if target_num_samples <= 0:
    msg = "num_samples must be positive."
    raise ValueError(msg)
  chunk_size = (
    target_num_samples
    if chunk_sample_count is not None
    else (spec.samples_chunk_size or target_num_samples)
  )
  if chunk_size <= 0:
    msg = "samples_chunk_size must be positive when provided."
    raise ValueError(msg)

  tie_group_map = None
  num_groups = None

  if spec.tie_group_map is not None:
    tie_group_map = jnp.asarray(spec.tie_group_map, dtype=jnp.int32)
  elif spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
    tie_group_map = resolve_tie_groups(spec, batched_ensemble)
  if tie_group_map is not None:
    num_groups = int(jnp.max(tie_group_map)) + 1

  noise_array = (
    jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
    if spec.backbone_noise is not None
    else jnp.zeros(1)
  )

  temperature_array = jnp.asarray(spec.temperature, dtype=jnp.float32)

  sample_fn_with_params = partial(
    sampler_fn,
    _k_neighbors=48,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    iterations=spec.iterations,
    learning_rate=spec.learning_rate,
    multi_state_strategy=spec.multi_state_strategy,
    multi_state_temperature=spec.multi_state_temperature,
    num_groups=num_groups,
  )

  # Ensure tie_group_map and mapping have batch dimensions for vmap.
  batch_size = batched_ensemble.coordinates.shape[0]
  seq_len = batched_ensemble.coordinates.shape[1]

  tie_map_for_vmap = None
  if tie_group_map is not None:
    if tie_group_map.ndim == 1:
      tie_map_for_vmap = jnp.broadcast_to(
        jnp.atleast_2d(tie_group_map),
        (batch_size, tie_group_map.shape[0]),
      )
    else:
      tie_map_for_vmap = tie_group_map

  mapping_for_vmap = (
    jnp.asarray(spec.structure_mapping, dtype=jnp.int32)
    if spec.structure_mapping is not None
    else batched_ensemble.mapping
  )
  if mapping_for_vmap is not None:
    mapping_for_vmap = _broadcast_per_structure(
      mapping_for_vmap,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.int32,
      name="structure_mapping",
    )

  fixed_mask_for_vmap, fixed_tokens_for_vmap = _prepare_fixed_controls(
    spec,
    batched_ensemble,
    tie_map_for_vmap,
  )
  ligand_context = _prepare_ligand_context(
    spec,
    batched_ensemble,
    batch_size,
    seq_len,
    canonical_structure_ids=canonical_structure_ids,
    batch_structure_ids=batch_structure_ids,
  )
  state_weights = (
    jnp.asarray(spec.state_weights, dtype=jnp.float32)
    if spec.state_weights is not None
    else None
  )

  def sample_single_config(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise: float,
    temp: float,
    current_tie_map: jnp.ndarray | None,
    structure_mapping: jnp.ndarray | None = None,
    fixed_mask_local: jnp.ndarray | None = None,
    fixed_tokens_local: jnp.ndarray | None = None,
    Y: jnp.ndarray | None = None,
    Y_t: jnp.ndarray | None = None,
    Y_m: jnp.ndarray | None = None,
    xyz_37: jnp.ndarray | None = None,
    xyz_37_m: jnp.ndarray | None = None,
    chain_mask: jnp.ndarray | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample one sequence for one structure configuration."""
    return sample_fn_with_params(
      key,
      coords,
      mask,
      residue_ix,
      chain_ix,
      backbone_noise=noise,
      temperature=temp,
      tie_group_map=current_tie_map,
      structure_mapping=structure_mapping,
      fixed_mask=fixed_mask_local,
      fixed_tokens=fixed_tokens_local,
      state_weights=state_weights,
      state_mapping=structure_mapping,
      Y=Y,
      Y_t=Y_t,
      Y_m=Y_m,
      xyz_37=xyz_37,
      xyz_37_m=xyz_37_m,
      chain_mask=chain_mask,
    )

  def internal_sample(
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    keys_arr: PRNGKeyArray,
    current_tie_map: jnp.ndarray | None,
    structure_mapping: jnp.ndarray | None = None,
    fixed_mask_local: jnp.ndarray | None = None,
    fixed_tokens_local: jnp.ndarray | None = None,
    Y: jnp.ndarray | None = None,
    Y_t: jnp.ndarray | None = None,
    Y_m: jnp.ndarray | None = None,
    xyz_37: jnp.ndarray | None = None,
    xyz_37_m: jnp.ndarray | None = None,
    chain_mask: jnp.ndarray | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample mapping over keys (sequential) and noise/temp (vectorized)."""

    def map_over_noise_and_temp(
      k: PRNGKeyArray,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      def map_over_temp(n: float) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        return _safe_map(
          lambda t: sample_single_config(
            k,
            coords,
            mask,
            residue_ix,
            chain_ix,
            n,
            t,
            current_tie_map,
            structure_mapping,
            fixed_mask_local,
            fixed_tokens_local,
            Y,
            Y_t,
            Y_m,
            xyz_37,
            xyz_37_m,
            chain_mask,
          ),
          temperature_array,
          batch_size=spec.temperature_batch_size,
        )

      return _safe_map(map_over_temp, noise_array, batch_size=spec.noise_batch_size)

    if spec.samples_batch_size and spec.samples_batch_size > 0:
      return jax.lax.map(
        map_over_noise_and_temp,
        keys_arr,
        batch_size=spec.samples_batch_size,
      )
    return jax.lax.map(map_over_noise_and_temp, keys_arr)

  tie_map_in_axis = 0 if tie_map_for_vmap is not None else None
  mapping_in_axis = 0 if mapping_for_vmap is not None else None
  fixed_mask_axis = 0 if fixed_mask_for_vmap is not None else None
  fixed_tokens_axis = 0 if fixed_tokens_for_vmap is not None else None
  Y_axis = 0 if ligand_context["Y"] is not None else None
  Y_t_axis = 0 if ligand_context["Y_t"] is not None else None
  Y_m_axis = 0 if ligand_context["Y_m"] is not None else None
  xyz_37_axis = 0 if ligand_context["xyz_37"] is not None else None
  xyz_37_m_axis = 0 if ligand_context["xyz_37_m"] is not None else None
  chain_mask_axis = 0 if ligand_context["chain_mask"] is not None else None

  vmap_structures = jax.vmap(
    internal_sample,
    in_axes=(
      0,
      0,
      0,
      0,
      None,
      tie_map_in_axis,
      mapping_in_axis,
      fixed_mask_axis,
      fixed_tokens_axis,
      Y_axis,
      Y_t_axis,
      Y_m_axis,
      xyz_37_axis,
      xyz_37_m_axis,
      chain_mask_axis,
    ),
  )

  sequence_chunks: list[jax.Array] = []
  logit_chunks: list[jax.Array] = []
  total_chunks = (target_num_samples + chunk_size - 1) // chunk_size
  default_sample_offset = int(grid_lineage["sample_start"]) if grid_lineage is not None else 0
  sample_offset = int(chunk_sample_start) if chunk_sample_start is not None else default_sample_offset

  for chunk_iter in range(total_chunks):
    chunk_start = chunk_iter * chunk_size
    chunk_count = min(chunk_size, target_num_samples - chunk_start)
    sample_indices = range(sample_offset + chunk_start, sample_offset + chunk_start + chunk_count)
    keys = jnp.stack([jax.random.fold_in(base_key, sample_idx) for sample_idx in sample_indices], axis=0)

    chunk_sequences, chunk_logits, _ = vmap_structures(
      batched_ensemble.coordinates,
      batched_ensemble.mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      keys,
      tie_map_for_vmap,
      mapping_for_vmap,
      fixed_mask_for_vmap,
      fixed_tokens_for_vmap,
      ligand_context["Y"],
      ligand_context["Y_t"],
      ligand_context["Y_m"],
      ligand_context["xyz_37"],
      ligand_context["xyz_37_m"],
      ligand_context["chain_mask"],
    )
    sequence_chunks.append(chunk_sequences)
    logit_chunks.append(chunk_logits)

  sampled_sequences = jnp.concatenate(sequence_chunks, axis=1)
  sampled_logits = jnp.concatenate(logit_chunks, axis=1)

  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(sampled_logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    mask = batched_ensemble.mask
    if mask is None:
      mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    pseudo_perplexity = jnp.exp(nll / jnp.sum(mask, axis=-1))
    return sampled_sequences, sampled_logits, pseudo_perplexity
  return sampled_sequences, sampled_logits, None


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Sample new sequences for the given input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then samples new sequences for each structure.

  Args:
      spec: An optional SamplingSpecification object. If None, a default will be created using
      kwargs, options are provided as keyword arguments. The following options can be set:
        inputs: A single or sequence of inputs (files, PDB IDs, etc.).
        chain_id: Specific chain(s) to parse from the structure.
        model: The model number to load. If None, all models are loaded.
        altloc: The alternate location identifier to use.
        model_version: The model version to use.
        model_weights: The model weights to use.
        foldcomp_database: The FoldComp database to use for FoldComp IDs.
        random_seed: The random number generator key.
        backbone_noise: The amount of noise to add to the backbone.
        num_samples: The number of sequences to sample per structure/noise level.
        sampling_strategy: The sampling strategy to use.
        temperature: The sampling temperature.
        bias: An optional array to bias the logits.
        fixed_positions: An optional array of residue indices to keep fixed.
        iterations: Number of optimization iterations for "straight_through" sampling.
        learning_rate: Learning rate for "straight_through" sampling.
        batch_size: The number of structures to process in a single batch.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing sampled sequences, logits, and metadata.

  """
  if spec is None:
    spec = SamplingSpecification(**kwargs)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  if spec.average_node_features:
    return _sample_averaged_mode(spec, protein_iterator, model)

  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=_DEFAULT_DECODING_ORDER_FN,
    sampling_strategy=spec.sampling_strategy,
  )

  if spec.output_h5_path:
    return _sample_streaming(spec, protein_iterator, sampler_fn)

  all_sequences, all_pseudo_perplexities = [], []
  all_logits = [] if spec.return_logits else None
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0

  for batched_ensemble in protein_iterator:
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )
    sampled_sequences, logits, pseudo_perplexity = _sample_batch(
      spec,
      batched_ensemble,
      sampler_fn,
      canonical_structure_ids=canonical_structure_ids,
      batch_structure_ids=batch_structure_ids,
    )
    all_sequences.append(sampled_sequences)
    if spec.return_logits and all_logits is not None:
      all_logits.append(logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)
    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  max_len = max(arr.shape[-1] for arr in all_sequences)

  def pad_to_max(arr: jax.Array, target_len: int, axis: int = -1, pad_value: int = 0) -> jax.Array:
    """Pad the specified dimension of a JAX array to target_len."""
    diff = target_len - arr.shape[axis]
    if diff == 0:
      return arr
    padding_config = [(0, 0)] * arr.ndim
    # Handle negative axis
    axis = axis % arr.ndim
    padding_config[axis] = (0, diff)
    return jnp.pad(arr, padding_config, constant_values=pad_value)

  all_sequences_padded = [pad_to_max(seq, max_len, axis=-1, pad_value=0) for seq in all_sequences]

  all_masks = [
    pad_to_max(
      jnp.ones(seq.shape, dtype=jnp.int32),
      max_len,
      axis=-1,
      pad_value=0,
    )
    for seq in all_sequences
  ]

  results = {
    "sequences": jnp.concatenate(all_sequences_padded, axis=0),
    "mask": jnp.concatenate(all_masks, axis=0),
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }
  if spec.return_logits and all_logits is not None:
    all_logits_padded = [pad_to_max(logits, max_len, axis=-2, pad_value=0) for logits in all_logits]
    results["logits"] = jnp.concatenate(all_logits_padded, axis=0)
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = jnp.concatenate(all_pseudo_perplexities, axis=0)

  grid_lineage = _resolve_grid_lineage(spec)
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    chunk_size = int(spec.samples_chunk_size or grid_lineage["sample_count"])
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size,
    )
    sample_indices = _grid_sample_indices(grid_lineage)
    results["sample_indices"] = jnp.asarray(sample_indices, dtype=jnp.int32)
    results["metadata"]["lineage"] = {
      **grid_lineage,
      "manifest_row_hash": manifest_row_hash,
      "sample_indices": sample_indices.tolist(),
      "grid_iteration_ids": iteration_ids.tolist(),
      "grid_iteration_sample_start": iteration_starts.tolist(),
      "grid_iteration_sample_count": iteration_counts.tolist(),
    }

  return results


def _sample_streaming(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  sampler_fn: Callable,
) -> dict[str, Any]:
  """Sample new sequences and stream results to an HDF5 file."""
  grid_lineage = _resolve_grid_lineage(spec)
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  total_num_samples = (
    int(grid_lineage["sample_count"]) if grid_lineage is not None else int(spec.num_samples)
  )
  chunk_size = int(spec.samples_chunk_size or total_num_samples)
  sample_start = int(grid_lineage["sample_start"]) if grid_lineage is not None else 0
  with h5py.File(spec.output_h5_path, "w") as f:
    f.attrs["schema_version"] = GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION
    f.attrs["model_family"] = spec.model_family
    f.attrs["ligand_conditioning"] = int(spec.ligand_conditioning)
    f.attrs["sidechain_conditioning"] = int(spec.sidechain_conditioning)
    f.attrs["samples_chunk_size"] = chunk_size
    if grid_lineage is not None:
      manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
      f.attrs["job_id"] = str(grid_lineage["job_id"])
      f.attrs["chunk_id"] = int(grid_lineage["chunk_id"])
      f.attrs["sample_start"] = int(grid_lineage["sample_start"])
      f.attrs["sample_count"] = int(grid_lineage["sample_count"])
      f.attrs["manifest_row_hash"] = manifest_row_hash
      sample_indices = _grid_sample_indices(grid_lineage)
      iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
        grid_lineage,
        chunk_size=chunk_size,
      )
      f.create_dataset("sample_indices", data=sample_indices, dtype="i8")
      f.create_dataset("grid_iteration_ids", data=iteration_ids, dtype="i8")
      f.create_dataset("grid_iteration_sample_start", data=iteration_starts, dtype="i8")
      f.create_dataset("grid_iteration_sample_count", data=iteration_counts, dtype="i8")
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_idx,
        batch_size=batch_size,
      )
      if not spec.campaign_mode:
        sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch(
          spec,
          batched_ensemble,
          sampler_fn,
          canonical_structure_ids=canonical_structure_ids,
          batch_structure_ids=batch_structure_ids,
        )
        for i in range(sampled_sequences.shape[0]):
          grp = f.create_group(f"structure_{structure_idx}")
          grp.create_dataset("sequences", data=sampled_sequences[i], dtype="i4")
          if spec.return_logits:
            grp.create_dataset("logits", data=sampled_logits[i], dtype="f4")
          if pseudo_perplexity is not None:
            grp.create_dataset("pseudo_perplexity", data=pseudo_perplexity[i], dtype="f4")
          # Store metadata about the structure
          grp.attrs["structure_index"] = structure_idx
          grp.attrs["structure_id"] = batch_structure_ids[i]
          grp.attrs["num_samples"] = sampled_sequences.shape[1]
          grp.attrs["num_noise_levels"] = sampled_sequences.shape[2]
          grp.attrs["num_temperatures"] = sampled_sequences.shape[3]
          grp.attrs["sequence_length"] = sampled_sequences.shape[4]
          if grid_lineage is not None:
            grp.attrs["job_id"] = str(grid_lineage["job_id"])
            grp.attrs["chunk_id"] = int(grid_lineage["chunk_id"])
            grp.attrs["sample_start"] = int(grid_lineage["sample_start"])
            grp.attrs["sample_count"] = int(grid_lineage["sample_count"])
          resolved_structure_ids.append(batch_structure_ids[i])
          structure_idx += 1
      else:
        structure_groups: list[h5py.Group] = []
        for i in range(batch_size):
          grp = f.create_group(f"structure_{structure_idx}")
          grp.attrs["structure_index"] = structure_idx
          grp.attrs["structure_id"] = batch_structure_ids[i]
          grp.attrs["num_samples"] = total_num_samples
          grp.attrs["num_noise_levels"] = len(spec.backbone_noise)
          grp.attrs["num_temperatures"] = len(spec.temperature)
          grp.attrs["sequence_length"] = batched_ensemble.coordinates.shape[1]
          if grid_lineage is not None:
            grp.attrs["job_id"] = str(grid_lineage["job_id"])
            grp.attrs["chunk_id"] = int(grid_lineage["chunk_id"])
            grp.attrs["sample_start"] = int(grid_lineage["sample_start"])
            grp.attrs["sample_count"] = int(grid_lineage["sample_count"])
          structure_groups.append(grp)
          resolved_structure_ids.append(batch_structure_ids[i])
          structure_idx += 1

        for chunk_start in range(0, total_num_samples, chunk_size):
          chunk_count = min(chunk_size, total_num_samples - chunk_start)
          chunk_sample_start = sample_start + chunk_start
          sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch(
            spec,
            batched_ensemble,
            sampler_fn,
            canonical_structure_ids=canonical_structure_ids,
            batch_structure_ids=batch_structure_ids,
            chunk_sample_start=chunk_sample_start,
            chunk_sample_count=chunk_count,
          )

          for i, grp in enumerate(structure_groups):
            seq_chunk = np.asarray(sampled_sequences[i], dtype=np.int32)
            if "sequences" not in grp:
              grp.create_dataset(
                "sequences",
                shape=(0, *seq_chunk.shape[1:]),
                maxshape=(None, *seq_chunk.shape[1:]),
                chunks=True,
                dtype="i4",
              )
            seq_ds = grp["sequences"]
            seq_ds.resize(seq_ds.shape[0] + seq_chunk.shape[0], axis=0)
            seq_ds[-seq_chunk.shape[0] :] = seq_chunk

            if spec.return_logits:
              logits_chunk = np.asarray(sampled_logits[i], dtype=np.float32)
              if "logits" not in grp:
                grp.create_dataset(
                  "logits",
                  shape=(0, *logits_chunk.shape[1:]),
                  maxshape=(None, *logits_chunk.shape[1:]),
                  chunks=True,
                  dtype="f4",
                )
              logits_ds = grp["logits"]
              logits_ds.resize(logits_ds.shape[0] + logits_chunk.shape[0], axis=0)
              logits_ds[-logits_chunk.shape[0] :] = logits_chunk

            if pseudo_perplexity is not None:
              perplexity_chunk = np.asarray(pseudo_perplexity[i], dtype=np.float32)
              if "pseudo_perplexity" not in grp:
                grp.create_dataset(
                  "pseudo_perplexity",
                  shape=(0, *perplexity_chunk.shape[1:]),
                  maxshape=(None, *perplexity_chunk.shape[1:]),
                  chunks=True,
                  dtype="f4",
                )
              perplexity_ds = grp["pseudo_perplexity"]
              perplexity_ds.resize(perplexity_ds.shape[0] + perplexity_chunk.shape[0], axis=0)
              perplexity_ds[-perplexity_chunk.shape[0] :] = perplexity_chunk

      f.flush()

  results = {
    "output_h5_path": str(spec.output_h5_path),
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size,
    )
    results["metadata"]["lineage"] = {
      **grid_lineage,
      "manifest_row_hash": manifest_row_hash,
      "sample_indices": _grid_sample_indices(grid_lineage).tolist(),
      "grid_iteration_ids": iteration_ids.tolist(),
      "grid_iteration_sample_start": iteration_starts.tolist(),
      "grid_iteration_sample_count": iteration_counts.tolist(),
    }
  return results


def _create_decode_wrapper(base_decode_fn: Callable) -> Callable:
  """Create a custom decode wrapper that averages logits over structural features."""

  def wrapped(
    encoded_features: tuple,
    sequence: ProteinSequence,
    ar_mask_in: AutoRegressiveMask,
  ) -> Logits:
    avg_node, avg_edge, neighbors, mask, ar_mask_struct = encoded_features

    # Flatten batch dimensions
    neighbors_flat = neighbors.reshape(
      (-1, neighbors.shape[-2], neighbors.shape[-1]),
    )
    mask_flat = mask.reshape((-1, mask.shape[-1]))
    ar_mask_struct_flat = ar_mask_struct.reshape(
      (-1, ar_mask_struct.shape[-2], ar_mask_struct.shape[-1]),
    )

    def decode_single(
      n_idx: jnp.ndarray,
      m: jnp.ndarray,
      ar_m: jnp.ndarray,
    ) -> Logits:
      return base_decode_fn(
        (avg_node, avg_edge, n_idx, m, ar_m),
        sequence,
        ar_mask_in,
      )

    logits_batch = jax.vmap(decode_single)(neighbors_flat, mask_flat, ar_mask_struct_flat)
    return jnp.mean(logits_batch, axis=0)

  return wrapped


def _sample_averaged_mode(
  spec: SamplingSpecification,
  protein_iterator: Any,  # noqa: ANN401
  model: PrxteinMPNN,
) -> dict[str, Any]:
  """Run sampling in averaged node features mode."""
  if spec.output_h5_path:
    return _sample_streaming_averaged(spec, protein_iterator, model)

  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)
  all_sequences, all_logits, all_pseudo_perplexities = [], [], []

  for batched_ensemble in protein_iterator:
    sampled_sequences, logits, pseudo_perplexity = _sample_batch_averaged(
      spec,
      batched_ensemble,
      model,
      sample_fn,
      decode_fn,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)

  results = {
    "sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = jnp.concatenate(all_pseudo_perplexities, axis=0)

  return results


def _internal_sample_averaged(
  spec: SamplingSpecification,
  encoded_feat: tuple,
  keys_arr: PRNGKeyArray,
  sample_fn_with_params: Callable,
  tie_group_map: jnp.ndarray | None,
  num_groups: int | None,
) -> ProteinSequence:
  """Sample mapping over keys for averaged features."""
  decoding_order_keys = jax.random.split(jax.random.key(spec.random_seed + 1), spec.num_samples)

  temperature_array = jnp.asarray(spec.temperature, dtype=jnp.float32)

  def sample_single_sequence(
    key: PRNGKeyArray,
    decoding_order_key: PRNGKeyArray,
    encoded_feat: tuple,
    temperature: float,
  ) -> ProteinSequence:
    """Sample one sequence from averaged features."""
    seq_len = encoded_feat[0].shape[0]
    decoding_order, _ = _DEFAULT_DECODING_ORDER_FN(
      decoding_order_key,
      seq_len,
      tie_group_map,
      num_groups,
    )
    return sample_fn_with_params(key, encoded_feat, decoding_order, temperature=temperature)

  def sample_for_key(k: PRNGKeyArray, dok: PRNGKeyArray) -> ProteinSequence:
    return jax.vmap(
      lambda t: sample_single_sequence(k, dok, encoded_feat, t),
    )(temperature_array)

  vmap_sample_fn = jax.vmap(
    sample_for_key,
    in_axes=(0, 0),
    out_axes=0,
  )
  return vmap_sample_fn(keys_arr, decoding_order_keys)


def _compute_logits_averaged(
  spec: SamplingSpecification,
  averaged_encodings: tuple,
  sampled_sequences: ProteinSequence,
  decode_fn_wrapped: Callable,
) -> Logits:
  """Compute logits for the sampled sequences."""
  seq_len = sampled_sequences.shape[-1]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  if spec.average_encoding_mode == "inputs_and_noise":

    def get_logits_local_both(seq: ProteinSequence) -> Logits:
      return jax.vmap(lambda s: decode_fn_wrapped(averaged_encodings, s, ar_mask))(seq)

    vmap_logits = jax.vmap(get_logits_local_both)
    logits = vmap_logits(sampled_sequences[0])
    logits = jnp.expand_dims(logits, axis=0)
  else:

    def get_logits_local(seq: ProteinSequence, enc: tuple) -> Logits:
      return jax.vmap(lambda s: decode_fn_wrapped(enc, s, ar_mask))(seq)

    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    vmap_logits = jax.vmap(
      jax.vmap(get_logits_local, in_axes=(0, None)),
      in_axes=(0, (0, 0, struct_axis, struct_axis, struct_axis)),
    )
    logits = vmap_logits(sampled_sequences, averaged_encodings)

  return logits


def _sample_batch_averaged(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: PrxteinMPNN,
  sample_fn: Callable,  # noqa: ARG001
  decode_fn: Callable,  # noqa: ARG001
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
  """Sample sequences for a batched ensemble of proteins using averaged encodings."""
  keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

  tie_group_map = None
  num_groups = None

  if spec.tie_group_map is not None:
    tie_group_map = jnp.asarray(spec.tie_group_map, dtype=jnp.int32)
  elif spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
    tie_group_map = resolve_tie_groups(spec, batched_ensemble)
  if tie_group_map is not None:
    num_groups = int(jnp.max(tie_group_map)) + 1

  structure_mapping = (
    jnp.asarray(spec.structure_mapping, dtype=jnp.int32)
    if spec.structure_mapping is not None
    else batched_ensemble.mapping
  )

  averaged_encodings = get_averaged_encodings(
    batched_ensemble,
    model,
    spec.backbone_noise,
    spec.noise_batch_size,
    spec.random_seed,
    spec.average_encoding_mode,
    structure_mapping=structure_mapping,
  )

  # Create a new sample_fn with the wrapper
  _, sample_fn_wrapped, decode_fn_wrapped = make_encoding_sampling_split_fn(
    model,
    decode_fn_wrapper=_create_decode_wrapper,
  )

  sample_fn_with_params = partial(
    sample_fn_wrapped,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
    multi_state_strategy=spec.multi_state_strategy,
    multi_state_temperature=spec.multi_state_temperature,
  )

  if spec.average_encoding_mode == "inputs_and_noise":
    sampled_sequences = _internal_sample_averaged(
      spec,
      averaged_encodings,
      keys,
      sample_fn_with_params,
      tie_group_map,
      num_groups,
    )
    sampled_sequences = jnp.expand_dims(sampled_sequences, axis=0)
  else:
    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    def _call_internal(enc: tuple) -> ProteinSequence:
      return _internal_sample_averaged(
        spec,
        enc,
        keys,
        sample_fn_with_params,
        tie_group_map,
        num_groups,
      )

    vmap_sample_structures = jax.vmap(
      _call_internal,
      in_axes=((0, 0, struct_axis, struct_axis, struct_axis),),
    )
    sampled_sequences = vmap_sample_structures(
      averaged_encodings,
    )

  logits = _compute_logits_averaged(spec, averaged_encodings, sampled_sequences, decode_fn_wrapped)

  num_temps = len(cast("Sequence[float]", spec.temperature))
  # Reshape to (1, -1, num_temps, seq_len)
  seq_len = sampled_sequences.shape[-1]
  sampled_sequences = sampled_sequences.reshape((1, -1, num_temps, seq_len))
  logits = logits.reshape((1, -1, num_temps, seq_len, 21))

  if num_temps == 1:
    sampled_sequences = jnp.squeeze(sampled_sequences, axis=2)
    logits = jnp.squeeze(logits, axis=2)

  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    mask = batched_ensemble.mask
    if mask is None:
      mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    pseudo_perplexity = jnp.exp(nll / jnp.sum(mask, axis=-1))
    return sampled_sequences, logits, pseudo_perplexity
  return sampled_sequences, logits, None


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: PrxteinMPNN,
) -> dict[str, Any]:
  """Sample new sequences with averaged encodings and stream results to an HDF5 file."""
  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)

  with h5py.File(spec.output_h5_path, "w") as f:
    f.attrs["schema_version"] = "sampling_averaged_v1"
    f.attrs["model_family"] = spec.model_family
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch_averaged(
        spec,
        batched_ensemble,
        model,
        sample_fn,
        decode_fn,
      )
      for i in range(sampled_sequences.shape[0]):
        grp = f.create_group(f"structure_{structure_idx}")
        grp.create_dataset("sequences", data=sampled_sequences[i], dtype="i4")
        grp.create_dataset("logits", data=sampled_logits[i], dtype="f4")
        if pseudo_perplexity is not None:
          grp.create_dataset("pseudo_perplexity", data=pseudo_perplexity[i], dtype="f4")
        # Store metadata about the structure
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[1]
        grp.attrs["num_noise_levels"] = 1  # Averaged, so effectively 1 noise level
        grp.attrs["num_temperatures"] = (
          sampled_sequences.shape[2] if sampled_sequences.ndim == RANK_WITH_TEMPERATURE else 1
        )
        grp.attrs["sequence_length"] = sampled_sequences.shape[-1]
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "schema_version": "sampling_averaged_v1",
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
