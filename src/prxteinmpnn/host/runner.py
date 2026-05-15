"""Core user interface for the PrxteinMPNN package."""

# TODO(tech-debt): `.agents/TECHNICAL_DEBT.md` §14 + `TODO_io_callback.txt` (host-side I/O streaming).

from __future__ import annotations

import hashlib
import json
import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.types.bundles import (
    GeometryBundle,
    ConditioningBundle,
    LigandBundle,
    WaveScheduleBundle,
    InferenceBundle,
)
from prxteinmpnn.host._sampling_averaged import _sample_batch_averaged
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference import sample_autoregressive as sample_ar
from prxteinmpnn.inference import score_conditional
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.inference.logits import ArithmeticMeanLogits
from prxteinmpnn.host._sampling_grid_lineage import (
  _base_sampling_key,
  _resolve_grid_lineage,
)
from prxteinmpnn.host._sampling_helper import (
  RANK_WITH_TEMPERATURE,
  _DEFAULT_DECODING_ORDER_FN,
  _broadcast_per_structure,
  _canonical_structure_id,
  _canonical_structure_ids_for_spec,
  _dispatch_sampling_tensor_batch_io,
  _noop_sampling_chunk_io,
  _noop_sampling_structure_batch_io,
  _prepare_fixed_controls,
  _prepare_ligand_context,
  _structure_ids_for_batch,
)
from prxteinmpnn.host.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.host.streaming import (
    _sample_streaming,
    _sample_streaming_averaged,
    SAMPLING_SCHEMA_VERSION,
    GRID_SCHEMA_VERSION,
)
from prxteinmpnn.host.streaming_host import StreamingBatchHost
from prxteinmpnn.host.plan import (
    make_sampling_planner,
    extract_batch_sizes,
    compute_sample_keys,
    resolve_target_samples,
    resolve_chunk_size,
    resolve_sample_start,
)
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.tiling.planner import BatchPlan
from prxteinmpnn.utils.safe_map import safe_map as _safe_map

from .prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import SamplingSpecification, pop_deprecated_spec_kwargs

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from grain.python import IterDataset
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.data_structures import Protein
  from prxteinmpnn.types.arrays import (
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
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: ModelProtocol,
  *,
  canonical_structure_ids: Sequence[str] | None = None,
  batch_structure_ids: Sequence[str] | None = None,
  chunk_sample_start: int | None = None,
  chunk_sample_count: int | None = None,
  batch_idx: int = 0,
  structure_batch_count: int = -1,
  emit_structure_batch_io: bool = True,
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
  # 1. Plan batching
  plan = make_sampling_planner(spec)
  plan.log_summary()

  structures_bs, samples_bs, temps_bs, noises_bs = extract_batch_sizes(plan)

  # 2. Resolve Grids
  grid_lineage = _resolve_grid_lineage(spec)
  base_key = _base_sampling_key(spec, grid_lineage=grid_lineage)
  target_num_samples = resolve_target_samples(spec, chunk_sample_count, grid_lineage)

  noises = jnp.asarray(spec.backbone_noise)
  temperatures = jnp.asarray(spec.temperature)

  seq_len = batched_ensemble.coordinates.shape[1]
  batch_size = batched_ensemble.coordinates.shape[0]

  # Ensure tie_group_map and mapping have batch dimensions for vmap.
  tie_map_for_vmap = None
  if spec.tie_group_map is not None:
    tie_map_for_vmap = jnp.broadcast_to(
      jnp.atleast_2d(spec.tie_group_map),
      (batch_size, spec.tie_group_map.shape[0]),
    )

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
    batched_ensemble=batched_ensemble,
  )
  ligand_context = _prepare_ligand_context(
    spec,
    batched_ensemble=batched_ensemble,
    batch_size=batch_size,
    seq_len=seq_len,
    canonical_structure_ids=canonical_structure_ids,
    batch_structure_ids=batch_structure_ids,
  )
  state_weights = (
    jnp.asarray(spec.state_weights, dtype=jnp.float32) if spec.state_weights is not None else None
  )

  # 3. Inner Kernel Closure (Single Structure, Single Noise, Single Temp)
  def _call_kernel(key_samples, structure_idx, noise_val, temp_val):
      # Extract single structure from batch
      c = batched_ensemble.coordinates[structure_idx]
      m = batched_ensemble.mask[structure_idx]
      ri = batched_ensemble.residue_index[structure_idx]
      ci = batched_ensemble.chain_index[structure_idx]
      
      # Prepare controls
      fm = fixed_mask_for_vmap[structure_idx]
      ft = fixed_tokens_for_vmap[structure_idx]
      
      # Build bundle
      bundle, config, stage_set = build_inference_bundle(
          coords=c, mask=m, residue_index=ri, chain_index=ci,
          backbone_noise=noise_val,
          fixed_mask=fm, fixed_tokens=ft,
          bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
          state_weights=state_weights,
          y=ligand_context["y"][structure_idx] if ligand_context["y"] is not None else None,
          y_t=ligand_context["y_t"][structure_idx] if ligand_context["y_t"] is not None else None,
          y_m=ligand_context["y_m"][structure_idx] if ligand_context["y_m"] is not None else None,
          structure_mapping=mapping_for_vmap[structure_idx] if mapping_for_vmap is not None else None,
          temperature=temp_val,
          mode="sample_ar",
          strategy=spec.multi_state_strategy or "arithmetic_mean",
          strategy_temperature=spec.multi_state_temperature or 1.0,
          use_rolling_state=spec.use_rolling_state,
          inference=True
      )

      # Map over samples
      def _run_one_sample(k):
          res = sample_ar.kernel(model, k, bundle, config, stage_set)
          return res.sequence, res.logits
      
      return _safe_map(_run_one_sample, key_samples, batch_size=samples_bs)

  # 4. Nested Dispatch (Structures -> Noises -> Temps)
  # Compute deterministic sample keys
  sample_keys = compute_sample_keys(
      base_key,
      target_num_samples,
      chunk_sample_start=chunk_sample_start,
      grid_lineage_sample_start=grid_lineage["sample_start"] if grid_lineage is not None else None,
  )

  def _dispatch_structure(s_idx):
      def _dispatch_noise(n_val):
          def _dispatch_temp(t_val):
              return _call_kernel(sample_keys, s_idx, n_val, t_val)
          
          return _safe_map(_dispatch_temp, temperatures, batch_size=temps_bs)
      
      return _safe_map(_dispatch_noise, noises, batch_size=noises_bs)

  # Final batch map over structures
  # results: (batch, noise, temp, samples, seq_len)
  # logits: (batch, noise, temp, samples, seq_len, 21)
  sampled_sequences, sampled_logits = _safe_map(
      _dispatch_structure, 
      jnp.arange(batch_size), 
      batch_size=structures_bs
  )

  # 5. Post-process (transpose to expected output shape: [batch, samples, noise, temp, seq_len])
  # current: [B, D, T, N, L] -> desired: [B, N, D, T, L]
  sampled_sequences = jnp.transpose(sampled_sequences, (0, 3, 1, 2, 4))
  sampled_logits = jnp.transpose(sampled_logits, (0, 3, 1, 2, 4, 5))

  # 6. IO & Metadata
  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(sampled_logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    mask = batched_ensemble.mask
    if mask is None:
      mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    # mask is [B, L], nll is [B, N, D, T]
    # sum(mask, axis=-1) is [B]
    pseudo_perplexity = jnp.exp(nll / jnp.sum(mask, axis=-1)[:, None, None, None])
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
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = SamplingSpecification(**kw)

  protein_iterator, model = prep_protein_stream_and_model(spec)


  if spec.average_node_features:
    if spec.output_h5_path:
      return _sample_streaming_averaged(spec, protein_iterator, model, _sample_batch_averaged)
    return _sample_non_streaming_averaged(spec, protein_iterator, model)

  if spec.output_h5_path:
    return _sample_streaming(spec, protein_iterator, model, _sample_batch)

  # TODO(io_callback integration): Non-streaming path appends per-batch device arrays then concats;
  # prefer preallocated buffers or io_callback streaming (see prxteinmpnn/TODO_io_callback.txt).
  all_sequences, all_pseudo_perplexities = [], []
  all_logits = [] if spec.return_logits else None
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)
  grid_lineage = _resolve_grid_lineage(spec)

  for batch_idx, batched_ensemble in enumerate(protein_iterator):
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )
    sampled_sequences, logits, pseudo_perplexity = _sample_batch(
      spec,
      batched_ensemble,
      model,
      canonical_structure_ids=canonical_structure_ids,
      batch_structure_ids=batch_structure_ids,
      batch_idx=batch_idx,
      structure_batch_count=structure_batch_count,
    )
    all_sequences.append(sampled_sequences)
    if spec.return_logits and all_logits is not None:
      all_logits.append(logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)
    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  StreamingBatchHost.sink_barrier()
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

  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    total_num_samples_for_grid = resolve_target_samples(spec, grid_lineage=grid_lineage)
    chunk_size_for_grid = resolve_chunk_size(spec, total_num_samples_for_grid, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size_for_grid,
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








def _sample_non_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: PrxteinMPNN,
) -> dict[str, Any]:
  """Sample sequences with averaged encodings without streaming."""
  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)

  all_sequences, all_logits, all_pseudo_perplexities = [], [], []
  structure_batch_count_avg = StreamingBatchHost.structure_batch_count(protein_iterator)

  for batch_idx, batched_ensemble in enumerate(protein_iterator):
    sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch_averaged(
      spec,
      batched_ensemble,
      model,
      sample_fn,
      decode_fn,
      batch_idx,
      structure_batch_count_avg,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(sampled_logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)

  return {
    "sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "pseudo_perplexity": jnp.concatenate(all_pseudo_perplexities, axis=0) if all_pseudo_perplexities else None,
    "schema_version": "sampling_averaged_v1",
    "metadata": {
      "specification": spec,
    },
  }
