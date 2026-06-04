"""ArrayRecord and HDF5 streaming output paths for sampling."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from aminx.host._sampling_grid_lineage import (
  _grid_iteration_arrays,
  _grid_manifest_row_hash,
  _grid_sample_indices,
  _resolve_grid_lineage,
)
from aminx.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _structure_ids_for_batch,
)
from aminx.host.output_sinks import (
  streaming_tensor_sink_session,
  take_staging_sequences_logits,
)
from aminx.host.plan import (
  resolve_chunk_size,
  resolve_sample_start,
  resolve_target_samples,
)
from aminx.host.streaming_host import StreamingBatchHost
from aminx.io.designs import DesignArrayRecordWriter, DesignMetadata, DesignPayload
from aminx.run.specs import SamplingSpecification

if TYPE_CHECKING:
  from grain.python import IterDataset


SAMPLING_SCHEMA_VERSION = "sampling_v1"
GRID_SCHEMA_VERSION = "grid_v1"


def _sample_streaming(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  plan: Any,
  sample_batch_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Sample new sequences and stream results to an HDF5 or ArrayRecord file."""
  grid_lineage = _resolve_grid_lineage(spec)
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  total_num_samples = resolve_target_samples(spec, grid_lineage=grid_lineage)
  chunk_size = resolve_chunk_size(spec, total_num_samples, grid_lineage)
  sample_start = resolve_sample_start(grid_lineage)
  structure_batch_count_stream = StreamingBatchHost.structure_batch_count(protein_iterator)

  # Validate ArrayRecord and HDF5 parameter combinations
  if spec.use_arrayrecord and not spec.campaign_mode:
    msg = "use_arrayrecord=True requires campaign_mode=True."
    raise ValueError(msg)

  # Use ArrayRecord path for campaign mode if requested
  if spec.use_arrayrecord and spec.campaign_mode:
    return _sample_streaming_arrayrecord(
      spec,
      protein_iterator,
      plan,
      grid_lineage,
      canonical_structure_ids,
      sample_batch_fn,
    )

  # Deprecation warning for HDF5 path
  warnings.warn(
    "HDF5 output path is deprecated and will be removed in a future release. "
    "Use use_arrayrecord=True for async ArrayRecord output.",
    DeprecationWarning,
    stacklevel=3,
  )

  # Phase 5g PR4 tensor hook + streaming sink: ``_dispatch_sampling_tensor_batch_io`` stages host
  # sequences/logits under ``(batch_idx, chunk_start, chunk_count)``; ``take_staging_sequences_logits``
  # drains after ``jax.effects_barrier()`` (``TODO_io_callback.txt`` — perplexity stays return-path).
  with streaming_tensor_sink_session(), h5py.File(spec.output_h5_path, "w") as f:
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

    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_idx,
        batch_size=batch_size,
      )
      if not spec.campaign_mode:
        key_chunk_start = int(grid_lineage["sample_start"]) if grid_lineage is not None else 0
        key_chunk_count = total_num_samples
        _, _, pseudo_perplexity = sample_batch_fn(
          spec,
          batched_ensemble,
          plan,
          canonical_structure_ids=canonical_structure_ids,
          batch_structure_ids=batch_structure_ids,
          chunk_sample_start=key_chunk_start,
          chunk_sample_count=key_chunk_count,
          batch_idx=batch_idx,
          structure_batch_count=structure_batch_count_stream,
          emit_structure_batch_io=True,
        )
        StreamingBatchHost.sink_barrier()
        sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
          batch_idx,
          key_chunk_start,
          key_chunk_count,
        )
        for i in range(sampled_sequences_np.shape[0]):
          grp = f.create_group(f"structure_{structure_idx}")
          grp.create_dataset("sequences", data=sampled_sequences_np[i], dtype="i4")
          if spec.return_logits:
            grp.create_dataset("logits", data=sampled_logits_np[i], dtype="f4")
          if pseudo_perplexity is not None:
            grp.create_dataset("pseudo_perplexity", data=pseudo_perplexity[i], dtype="f4")
          # Store metadata about the structure
          grp.attrs["structure_index"] = structure_idx
          grp.attrs["structure_id"] = batch_structure_ids[i]
          grp.attrs["num_samples"] = sampled_sequences_np.shape[1]
          grp.attrs["num_noise_levels"] = sampled_sequences_np.shape[2]
          grp.attrs["num_temperatures"] = sampled_sequences_np.shape[3]
          grp.attrs["sequence_length"] = sampled_sequences_np.shape[4]
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

        chunks = list(StreamingBatchHost.iter_chunks(total_num_samples, chunk_size))
        for chunk_idx, (chunk_start, chunk_count) in enumerate(chunks):
          chunk_sample_start = sample_start + chunk_start
          is_last_chunk = chunk_idx == len(chunks) - 1
          _, _, pseudo_perplexity = sample_batch_fn(
            spec,
            batched_ensemble,
            plan,
            canonical_structure_ids=canonical_structure_ids,
            batch_structure_ids=batch_structure_ids,
            chunk_sample_start=chunk_sample_start,
            chunk_sample_count=chunk_count,
            batch_idx=batch_idx,
            structure_batch_count=structure_batch_count_stream,
            emit_structure_batch_io=is_last_chunk,
          )
          StreamingBatchHost.sink_barrier()
          sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
            batch_idx,
            chunk_sample_start,
            chunk_count,
          )

          for i, grp in enumerate(structure_groups):
            seq_chunk = sampled_sequences_np[i].astype(np.int32, copy=False)
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
              logits_chunk = sampled_logits_np[i].astype(np.float32, copy=False)
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

    # Flush only once at the end of sampling, not per batch, to reduce blocking I/O
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


def _sample_streaming_arrayrecord(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  plan: Any,
  grid_lineage: dict[str, int | str] | None,
  canonical_structure_ids: list[str],
  sample_batch_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Sample new sequences and stream results to ArrayRecord files (async path for campaign mode).

  Creates one .arrayrecord file per structure, using async thread-pool writes
  to avoid blocking the device on disk I/O.
  """
  resolved_structure_ids: list[str] = []
  total_num_samples = resolve_target_samples(spec, grid_lineage=grid_lineage)
  chunk_size = resolve_chunk_size(spec, total_num_samples, grid_lineage)
  sample_start = resolve_sample_start(grid_lineage)
  structure_idx = 0
  structure_batch_count_ar = StreamingBatchHost.structure_batch_count(protein_iterator)

  # Generate output base path (strip .h5 if present, use .arrayrecord)
  output_base = spec.output_h5_path
  if str(output_base).endswith(".h5"):
    output_base = Path(str(output_base)[:-3])

  structure_writers: list[tuple[int, DesignArrayRecordWriter]] = []
  try:
    with streaming_tensor_sink_session():
      for batch_idx, batched_ensemble in enumerate(protein_iterator):
        batch_size = batched_ensemble.coordinates.shape[0]
        batch_structure_ids = _structure_ids_for_batch(
          canonical_structure_ids,
          structure_offset=structure_idx,
          batch_size=batch_size,
        )

        # Create one ArrayRecord writer per structure in batch
        seq_len = int(batched_ensemble.coordinates.shape[1])
        n_states_rec = max(1, int(spec.run_spec.multistate.n_states))
        for i in range(batch_size):
          writer_path = Path(str(output_base) + f"_structure_{structure_idx}.arrayrecord")
          writer = DesignArrayRecordWriter.from_multistate_shapes(
            str(writer_path),
            n_canonical=seq_len,
            n_states=n_states_rec,
          )
          structure_writers.append((i, writer))
          resolved_structure_ids.append(batch_structure_ids[i])
          structure_idx += 1

        # Process samples in chunks
        chunks = list(StreamingBatchHost.iter_chunks(total_num_samples, chunk_size))
        for chunk_idx, (chunk_start, chunk_count) in enumerate(chunks):
          chunk_sample_start = sample_start + chunk_start
          is_last_chunk = chunk_idx == len(chunks) - 1
          _, _, _ = sample_batch_fn(
            spec,
            batched_ensemble,
            plan,
            canonical_structure_ids=canonical_structure_ids,
            batch_structure_ids=batch_structure_ids,
            chunk_sample_start=chunk_sample_start,
            chunk_sample_count=chunk_count,
            batch_idx=batch_idx,
            structure_batch_count=structure_batch_count_ar,
            emit_structure_batch_io=is_last_chunk,
          )
          StreamingBatchHost.sink_barrier()

          sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
            batch_idx,
            chunk_sample_start,
            chunk_count,
          )

          # Tensor host payloads drain here (Phase 5g sink unify); perplexity remains traced return path.
          for structure_batch_idx, writer in structure_writers:
            seq_chunk = sampled_sequences_np[structure_batch_idx].astype(np.uint8, copy=False)
            logits_chunk = (
              sampled_logits_np[structure_batch_idx].astype(np.float32, copy=False)
              if spec.return_logits
              else None
            )

            # Flatten samples-noise-temperature dimensions and write each design
            for sample_idx in range(seq_chunk.shape[0]):
              for noise_idx in range(seq_chunk.shape[1]):
                for temp_idx in range(seq_chunk.shape[2]):
                  sequence = seq_chunk[sample_idx, noise_idx, temp_idx, :]
                  logits = (
                    logits_chunk[sample_idx, noise_idx, temp_idx, :, :]
                    if logits_chunk is not None
                    else None
                  )
                  score = np.array([0.0], dtype=np.float32)  # Placeholder; adjust as needed
                  state_weights = np.ones(n_states_rec, dtype=np.float32) / float(n_states_rec)

                  metadata: DesignMetadata = {
                    "pool_type": "BackboneOnly",
                    "state_mapping": list(range(n_states_rec)),
                    "weight_strategy": "uniform",
                    "combination_algorithm": "none",
                    "structure_ids": [batch_structure_ids[structure_batch_idx]],
                    "parent_structure_idx": structure_idx - 1,
                  }

                  payload: DesignPayload = {
                    "sequence": sequence,
                    "logits": logits
                    if logits is not None
                    else np.zeros((writer.n_canonical, 21), dtype=np.float32),
                    "scores": score,
                    "state_weights": state_weights,
                    "metadata": metadata,
                  }
                  writer.write(payload)

  finally:
    # Close all writers (context manager will wait for pending async writes)
    for _, writer in structure_writers:
      writer.close()

  results = {
    "output_arrayrecord_paths": [
      str(Path(str(output_base) + f"_structure_{idx}.arrayrecord")) for idx in range(structure_idx)
    ],
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


def _original_sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: Any,
  sample_batch_averaged_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Sample new sequences with averaged encodings and stream results to an HDF5 file."""
  with h5py.File(spec.output_h5_path, "w") as f:
    f.attrs["schema_version"] = "sampling_averaged_v1"
    f.attrs["model_family"] = spec.model_family
    structure_idx = 0
    structure_batch_count_avg = StreamingBatchHost.structure_batch_count(protein_iterator)

    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      sampled_sequences, sampled_logits, pseudo_perplexity = sample_batch_averaged_fn(
        spec,
        batched_ensemble,
        model,
        batch_idx,
        structure_batch_count_avg,
      )
      StreamingBatchHost.sink_barrier()
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
          sampled_sequences.shape[2] if sampled_sequences.ndim == 4 else 1
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


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: Any,
  sample_batch_averaged_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Deprecated: Use _sample_streaming with a plan that has ArithmeticMeanEncodingFusion wired."""
  warnings.warn(
    "_sample_streaming_averaged is deprecated. Use _sample_streaming with a plan "
    "that has ArithmeticMeanEncodingFusion wired into stage_set.encoding_fusion.",
    DeprecationWarning,
    stacklevel=2,
  )
  return _original_sample_streaming_averaged(
    spec, protein_iterator, model, sample_batch_averaged_fn,
  )
