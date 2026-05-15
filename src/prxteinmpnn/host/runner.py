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
from prxteinmpnn.host.kernel_dispatch import _sample_batch
from prxteinmpnn.inference import score_conditional
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.inference.logits import ArithmeticMeanLogits
from prxteinmpnn.host._sampling_grid_lineage import (
  _base_sampling_key,
  _resolve_grid_lineage,
  _grid_manifest_row_hash,
  _grid_iteration_arrays,
  _grid_sample_indices,
)
from prxteinmpnn.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _structure_ids_for_batch,
)
from prxteinmpnn.host.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.host.logit_aggregation import (
    pad_to_max,
    aggregate_logits,
    aggregate_pseudo_perplexities,
)
from prxteinmpnn.host.streaming import (
    _sample_streaming,
    _sample_streaming_averaged,
    SAMPLING_SCHEMA_VERSION,
    GRID_SCHEMA_VERSION,
)
from prxteinmpnn.host.streaming_host import StreamingBatchHost
from prxteinmpnn.host.plan import (
    resolve_target_samples,
    resolve_chunk_size,
)
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.tiling.planner import BatchPlan

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
    results["logits"] = aggregate_logits(all_logits, max_len)
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = aggregate_pseudo_perplexities(all_pseudo_perplexities)

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
    "logits": aggregate_logits(all_logits),
    "pseudo_perplexity": aggregate_pseudo_perplexities(all_pseudo_perplexities) if all_pseudo_perplexities else None,
    "schema_version": "sampling_averaged_v1",
    "metadata": {
      "specification": spec,
    },
  }
