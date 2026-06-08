"""Core user interface for the Aminx package."""  # noqa: INP001

# COMP-NEW (2026-05-25): non-streaming path now drains via streaming_tensor_sink_session (§14 resolved).

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp

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
from aminx.host.kernel_dispatch import _sample_batch
from aminx.host.logit_aggregation import (
  aggregate_logits,
  aggregate_pseudo_perplexities,
  pad_to_max,
)
from aminx.host.output_sinks import (
  streaming_tensor_sink_session,
  take_staging_sequences_logits,
)
from aminx.host.plan import (
  make_inference_plan,
  resolve_chunk_size,
  resolve_target_samples,
)
from aminx.host.streaming import (
  GRID_SCHEMA_VERSION,
  SAMPLING_SCHEMA_VERSION,
  _sample_streaming,
)
from aminx.host.streaming_host import StreamingBatchHost
from aminx.run.specs import (
  InspectionSpecification,
  SamplingSpecification,
  ScoringSpecification,
  pop_deprecated_spec_kwargs,
)

from .prep import prep_protein_stream_and_model

logger = logging.getLogger(__name__)
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Sample new sequences for given input structures using high-performance Grain pipeline.

  Routes to streaming or in-memory sampling based on output configuration and averaging mode.
  Loads structures via Grain iterator, prepares model, processes in batches, and aggregates
  results with optional logits and pseudo-perplexity calculations.

  Parameters
  ----------
  spec : SamplingSpecification, optional
      SamplingSpecification object configuring inputs, model, and sampling strategy.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for SamplingSpecification if spec is None. Options include:
        inputs : str or list[str]
            Input structures (file paths, PDB IDs, etc.).
        chain_id : str, optional
            Specific chain(s) to parse.
        model : int, optional
            Model number to load. If None, all models used.
        altloc : str, optional
            Alternate location identifier.
        model_version : str, optional
            Model version (e.g., 'proteinmpnn_v1').
        model_weights : str, optional
            Path to model weights.
        foldcomp_database : str, optional
            FoldComp database for compressed structures.
        random_seed : int, optional
            Random seed for sampling.
        backbone_noise : float, optional
            Noise added to backbone coordinates.
        num_samples : int, optional
            Number of sequences per structure/noise level.
        sampling_strategy : str, optional
            Sampling strategy (e.g., 'autoregressive', 'straight_through').
        temperature : float, optional
            Sampling temperature.
        bias : jax.Array, optional
            Per-position or per-position-per-token logit bias. Shape: (L,) or (L, 21).
        fixed_positions : list[int], optional
            Residue indices to keep fixed.
        iterations : int, optional
            Optimization iterations for 'straight_through' strategy.
        learning_rate : float, optional
            Learning rate for 'straight_through' strategy.
        batch_size : int, optional
            Number of structures per batch.
        return_logits : bool, optional
            Whether to return logits for each position.
        output_h5_path : str, optional
            Path to streaming H5 output file.
        grid_mode : bool, optional
            Whether in grid sampling mode (affects metadata lineage).

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        sequences : jax.Array
            Sampled sequences, shape (B*N, L) where B is batch count,
            N is num_samples per structure, L is sequence length.
            Padded to max length across all sequences, masked with 'mask' key.
        mask : jax.Array
            Sequence validity mask (1 for valid, 0 for padding). Shape: (B*N, L).
        schema_version : str
            Schema version for results ('grid_v1' or 'sampling_v1').
        metadata : dict
            Metadata including specification, skipped_inputs, structure_ids,
            and optional lineage info (grid mode).
        logits : jax.Array, optional
            Per-position logits if return_logits=True. Shape: (B*N, L, 21).
        pseudo_perplexity : jax.Array, optional
            Per-sequence pseudo-perplexity scores if computed.

  Notes
  -----
  Streaming vs. in-memory: If output_h5_path is set, uses streaming I/O.
  Otherwise, concatenates per-batch results in memory. See _sample_streaming
  in streaming.py for streaming mode details.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = SamplingSpecification(**kw)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  # Construct inference plan once before routing to streaming or non-streaming path
  plan = make_inference_plan(model, spec)

  if spec.output_h5_path:
    return _sample_streaming(spec, protein_iterator, plan, _sample_batch)

  # Non-streaming path uses io_callback staging via streaming_tensor_sink_session;
  # drains per-batch via take_staging_sequences_logits.
  all_sequences, all_pseudo_perplexities = [], []
  all_logits = [] if spec.return_logits else None
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)
  grid_lineage = _resolve_grid_lineage(spec)

  with streaming_tensor_sink_session():
    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_offset,
        batch_size=batch_size,
      )
      target_for_batch = resolve_target_samples(spec, None, grid_lineage)
      _, _, pseudo_perplexity = _sample_batch(
        spec,
        batched_ensemble,
        plan,
        canonical_structure_ids=canonical_structure_ids,
        batch_structure_ids=batch_structure_ids,
        batch_idx=batch_idx,
        structure_batch_count=structure_batch_count,
      )
      StreamingBatchHost.sink_barrier()
      sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
        batch_idx,
        0,
        target_for_batch,
      )
      all_sequences.append(jnp.asarray(sampled_sequences_np))
      if spec.return_logits and all_logits is not None:
        all_logits.append(jnp.asarray(sampled_logits_np))
      if pseudo_perplexity is not None:
        all_pseudo_perplexities.append(pseudo_perplexity)
      resolved_structure_ids.extend(batch_structure_ids)
      structure_offset += batch_size
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


SCORING_SCHEMA_VERSION = "scoring_v1"
INSPECTION_SCHEMA_VERSION = "inspection_v1"


def score(  # noqa: PLR0915
  spec: ScoringSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Score sequences against input structures.

  Loads structures via Grain iterator, prepares model, and computes negative
  log-likelihood (NLL) scores for provided sequences. Optionally returns logits
  and decoding orders.

  Parameters
  ----------
  spec : ScoringSpecification, optional
      ScoringSpecification object configuring inputs, sequences, and scoring options.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for ScoringSpecification if spec is None.

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        scores : jax.Array
            Negative log-likelihood scores. Shape: (num_structures, num_sequences).
        mask : jax.Array
            Structure validity mask (1 for valid, 0 for padding). Shape: (num_structures,).
        schema_version : str
            Schema version for results ('scoring_v1').
        metadata : dict
            Metadata including specification, structure_ids, and skipped_inputs.
        logits : jax.Array, optional
            Per-position logits if return_logits=True. Shape: (num_structures, num_sequences, L, 21).
        decoding_orders : jax.Array, optional
            Decoding orders if return_decoding_orders=True. Shape: (num_structures, num_sequences, L).

  Raises
  ------
  NotImplementedError
      If output_h5_path is set (HDF5 streaming not yet implemented for scoring).
  ValueError
      If sequence length doesn't match structure length.

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = ScoringSpecification(**kw)

  if spec.output_h5_path:
    msg = "score runner: HDF5 streaming output not yet implemented; omit --output-h5-path for in-memory results"
    raise NotImplementedError(msg)

  from aminx.scoring.score import make_score_fn  # noqa: PLC0415
  from aminx.utils.aa_convert import string_to_protein_sequence  # noqa: PLC0415

  protein_iterator, model = prep_protein_stream_and_model(spec)

  # Build score function once
  score_fn = make_score_fn(model)  # type: ignore[arg-type]

  # Convert string sequences to integer indices
  sequence_indices_list = []
  for seq_str in spec.sequences_to_score:
    seq_idx = string_to_protein_sequence(seq_str)
    sequence_indices_list.append(seq_idx)

  all_scores, all_logits, all_decoding_orders = [], None, None
  if spec.return_logits:
    all_logits = []
  if spec.return_decoding_orders:
    all_decoding_orders = []

  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0

  # Prepare random key
  prng_key = jax.random.PRNGKey(spec.random_seed or 42)

  for _batch_idx, batched_ensemble in enumerate(protein_iterator):
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )

    batch_scores = []
    batch_logits = [] if spec.return_logits else None
    batch_decoding_orders = [] if spec.return_decoding_orders else None

    for struct_idx in range(batch_size):
      struct_coords = batched_ensemble.coordinates[struct_idx]
      struct_mask = batched_ensemble.mask[struct_idx]
      struct_residue_index = batched_ensemble.residue_index[struct_idx]
      struct_chain_index = batched_ensemble.chain_index[struct_idx]

      struct_len = struct_coords.shape[0]
      struct_scores = []
      struct_logits_list = [] if spec.return_logits else None
      struct_decoding_orders_list = [] if spec.return_decoding_orders else None

      for seq_idx in sequence_indices_list:
        # Validate sequence length matches structure length
        if seq_idx.shape[0] != struct_len:
          msg = (
            f"Sequence length mismatch for structure {batch_structure_ids[struct_idx]}: "
            f"structure has {struct_len} residues, but sequence has {seq_idx.shape[0]} residues"
          )
          raise ValueError(msg)

        # Generate deterministic key for this structure/sequence pair
        prng_key, subkey = jax.random.split(prng_key)

        # Score the sequence
        nll, logits, decoding_order = score_fn(  # type: ignore[misc]
          subkey,
          seq_idx,
          struct_coords,
          struct_mask,
          struct_residue_index,
          struct_chain_index,
          multi_state_strategy=spec.multi_state_strategy,
        )

        struct_scores.append(nll)
        if spec.return_logits and struct_logits_list is not None:
          struct_logits_list.append(logits)
        if spec.return_decoding_orders and struct_decoding_orders_list is not None:
          struct_decoding_orders_list.append(decoding_order)

      batch_scores.append(jnp.stack(struct_scores))
      if spec.return_logits and batch_logits is not None and struct_logits_list:
        batch_logits.append(jnp.stack(struct_logits_list))
      if spec.return_decoding_orders and batch_decoding_orders is not None and struct_decoding_orders_list:
        batch_decoding_orders.append(jnp.stack(struct_decoding_orders_list))

    all_scores.append(jnp.stack(batch_scores))
    if spec.return_logits and all_logits is not None and batch_logits:
      all_logits.append(jnp.stack(batch_logits))
    if spec.return_decoding_orders and all_decoding_orders is not None and batch_decoding_orders:
      all_decoding_orders.append(jnp.stack(batch_decoding_orders))

    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  # Concatenate results: shape (num_structures, num_sequences)
  scores_array = jnp.concatenate(all_scores, axis=0)

  results = {
    "scores": scores_array,
    "mask": jnp.ones(scores_array.shape[0], dtype=jnp.int32),
    "schema_version": SCORING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }

  if spec.return_logits and all_logits is not None:
    # Shape: (num_structures, num_sequences, L, 21)
    results["logits"] = jnp.concatenate(all_logits, axis=0)

  if spec.return_decoding_orders and all_decoding_orders is not None:
    # Shape: (num_structures, num_sequences, L)
    results["decoding_orders"] = jnp.concatenate(all_decoding_orders, axis=0)

  return results


def inspect(  # noqa: PLR0915
  spec: InspectionSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Inspect model encodings and features for input structures.

  Computes requested features (unconditional logits, encoded node features, etc.)
  for each input structure. Features are collected per-structure in the result dict.

  Parameters
  ----------
  spec : InspectionSpecification, optional
      InspectionSpecification object configuring inputs and inspection options.
      If None, a default specification is constructed from **kwargs.
  **kwargs : Any
      Keyword arguments for InspectionSpecification if spec is None.

  Returns
  -------
  dict[str, Any]
      Dictionary with keys:
        {feature_name}: list or jax.Array
            Per-feature results. Each feature is a list of arrays, one per structure.
        mask : jax.Array
            Structure validity mask. Shape: (num_structures,).
        schema_version : str
            Schema version for results ('inspection_v1').
        metadata : dict
            Metadata including specification, structure_ids, and skipped_inputs.

  Raises
  ------
  NotImplementedError
      If unsupported features are requested:
      - 'decoded_node_features'
      - distance_matrix or cross_input_similarity features

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = InspectionSpecification(**kw)

  # Validate unsupported features early
  if "decoded_node_features" in spec.inspection_features:
    msg = "inspect: 'decoded_node_features' requires decoder intermediate outputs not yet exposed"
    raise NotImplementedError(msg)
  if spec.distance_matrix:
    msg = "inspect: distance_matrix not yet implemented (calculate_*_distance_matrix are stubs)"
    raise NotImplementedError(msg)
  if spec.cross_input_similarity:
    msg = "inspect: cross_input_similarity metrics not yet implemented"
    raise NotImplementedError(msg)

  from aminx.inference.bundle_builder import build_inference_bundle  # noqa: PLC0415
  from aminx.inference.logits import make_stage_set  # noqa: PLC0415
  from aminx.inference.score_unconditional import kernel as score_unconditional  # noqa: PLC0415
  from aminx.sampling.conditional_logits import (  # noqa: PLC0415
    make_conditional_logits_fn,
    make_encoding_conditional_logits_split_fn,
  )

  protein_iterator, model = prep_protein_stream_and_model(spec)

  results_per_feature = {feat: [] for feat in spec.inspection_features}

  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  structure_offset = 0

  # Prepare random key
  prng_key = jax.random.PRNGKey(spec.random_seed or 42)

  for _batch_idx, batched_ensemble in enumerate(protein_iterator):
    batch_size = batched_ensemble.coordinates.shape[0]
    batch_structure_ids = _structure_ids_for_batch(
      canonical_structure_ids,
      structure_offset=structure_offset,
      batch_size=batch_size,
    )

    for struct_idx in range(batch_size):
      struct_coords = batched_ensemble.coordinates[struct_idx]
      struct_mask = batched_ensemble.mask[struct_idx]
      struct_residue_index = batched_ensemble.residue_index[struct_idx]
      struct_chain_index = batched_ensemble.chain_index[struct_idx]

      prng_key, subkey = jax.random.split(prng_key)

      for feature_name in spec.inspection_features:
        if feature_name == "unconditional_logits":
          # Build inference bundle and compute unconditional logits
          bundle, config = build_inference_bundle(
            coords=struct_coords,
            mask=struct_mask,
            residue_index=struct_residue_index,
            chain_index=struct_chain_index,
            sequence=None,  # Not needed for unconditional
            backbone_noise=0.0,
            ar_mask=None,
            structure_mapping=None,
            mode="score_unconditional",
            inference=True,
          )
          stage_set = make_stage_set()
          logits = score_unconditional(model, subkey, bundle, config, stage_set)  # type: ignore[arg-type]
          results_per_feature[feature_name].append(logits)

        elif feature_name == "conditional_logits":
          # Use native sequence from parsed structure
          if hasattr(batched_ensemble, "aatype") and batched_ensemble.aatype is not None:
            native_seq = batched_ensemble.aatype[struct_idx]
          else:
            msg = "Structure does not contain sequence information; cannot compute conditional_logits"
            raise ValueError(msg)

          cond_logits_fn = make_conditional_logits_fn(model)
          logits = cond_logits_fn(
            subkey,
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            native_seq,
          )
          results_per_feature[feature_name].append(logits)

        elif feature_name in ("encoded_node_features", "edge_features"):
          # Use the split function to compute encoder output
          encode_fn, _ = make_encoding_conditional_logits_split_fn(model)
          enc_output = encode_fn(
            struct_coords,
            struct_mask,
            struct_residue_index,
            struct_chain_index,
            backbone_noise=0.0,
            prng_key=subkey,
            structure_mapping=None,
          )

          if feature_name == "encoded_node_features":
            results_per_feature[feature_name].append(enc_output.node_features)
          else:  # edge_features
            results_per_feature[feature_name].append(enc_output.edge_features)

    resolved_structure_ids.extend(batch_structure_ids)
    structure_offset += batch_size

  results = {
    "mask": jnp.ones(len(resolved_structure_ids), dtype=jnp.int32),
    "schema_version": INSPECTION_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }

  # Add feature results
  results.update(results_per_feature)

  return results
