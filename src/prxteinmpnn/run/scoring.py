"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, fields, replace
from typing import TYPE_CHECKING, Any, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.run.averaging import get_averaged_encodings
from prxteinmpnn.run.output_sinks import active_scoring_sink
from prxteinmpnn.run.scoring_driver import ScoringDriver
from prxteinmpnn.run.streaming_host import StreamingBatchHost
from prxteinmpnn.scoring.score import score_sequence_with_encoding
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.batching import BatchPlan, BatchPlanner, estimate_memory_theoretical
from prxteinmpnn.utils.batching_registry import N_NOISES, N_STRUCTURES
from prxteinmpnn.utils.safe_map import safe_map

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification, ScoringSpecification, pop_deprecated_spec_kwargs

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.data_structures import Protein
  from prxteinmpnn.utils.types import Logits, ProteinSequence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


def _noop_scoring_structure_batch_io(batch_idx: object, batch_count: object) -> None:
  """Host structure-batch boundary hook for scoring (Phase **5g** PR2b / **PR2c**).

  ``jax.experimental.io_callback`` invokes this on the host with ``ordered=False``;
  completion order is **not** guaranteed to match program order. Only lightweight
  scalar markers cross here — scores / logits stay on device until list concat or
  HDF5 materialization (see ``TODO_io_callback.txt``).

  ``batch_count`` is ``-1`` when the protein iterator does not expose ``__len__``
  before the host loop (Grain / streaming iterators).

  Default implementation is a no-op so production pays minimal overhead; tests may
  monkeypatch this symbol **before** the first traced scoring call for that graph.
  """
  del batch_idx, batch_count


def _noop_scoring_tensor_batch_io(
  batch_idx: object,
  batch_count: object,
  scores_host: object,
  logits_host: object,
) -> None:
  """Host tensor-batch hook for scoring per-batch ``scores`` / ``logits`` (Phase **5g** PR3b).

  Invoked via ``jax.experimental.io_callback(..., ordered=False)``; completion order is **not**
  guaranteed to match program order. Operands are wrapped with ``stop_gradient`` at call sites.

  Default implementation is a no-op for minimal production overhead; tests may monkeypatch this
  symbol **before** the first traced scoring call for that graph.
  """
  del batch_idx, batch_count, scores_host, logits_host


def _dispatch_scoring_tensor_batch_io(
  batch_idx: object,
  batch_count: object,
  scores_host: object,
  logits_host: object,
) -> None:
  """Trace-stable entry for scoring tensor ``io_callback`` (active sink or noop)."""
  # TODO(phase5f-followup): activate scoring_tensor_sink_session() in score() so
  # active_scoring_sink() returns a real sink for tensor D2H streaming paths.
  sink = active_scoring_sink()
  if sink is not None:
    sink.on_scoring_scores_logits(batch_idx, batch_count, scores_host, logits_host)
    return
  _noop_scoring_tensor_batch_io(batch_idx, batch_count, scores_host, logits_host)


def _scoring_structure_and_tensor_batch_io(
  batch_idx: jax.Array,
  batch_count: jax.Array,
  scores: jax.Array,
  logits: jax.Array,
) -> None:
  """Emit scalar structure-batch marker then per-batch tensor hook (both ``ordered=False``)."""
  jax.experimental.io_callback(
    _noop_scoring_structure_batch_io,
    None,
    jax.lax.stop_gradient(batch_idx),
    jax.lax.stop_gradient(batch_count),
    ordered=False,
  )
  jax.experimental.io_callback(
    _dispatch_scoring_tensor_batch_io,
    None,
    jax.lax.stop_gradient(batch_idx),
    jax.lax.stop_gradient(batch_count),
    jax.lax.stop_gradient(scores),
    jax.lax.stop_gradient(logits),
    ordered=False,
  )


def _make_scoring_planner(
    spec: ScoringSpecification,
    param_bytes: float = 0.0,
    headroom: float = 0.80,
    activation_multiplier: float = 2.5,
) -> BatchPlan:
  """Create a BatchPlan for the scoring task.

  Args:
      spec: ScoringSpecification with batch_size and backbone_noise settings.
      param_bytes: Model parameter memory (default 0.0).
      headroom: Fraction of device memory to use as budget (default 0.80).
      activation_multiplier: Activation memory multiplier (default 2.5).

  Returns:
      A BatchPlan configured for the scoring workload.
  """
  try:
    import jax as jax_module
    limit = jax_module.devices()[0].memory_stats()["bytes_limit"]
  except Exception:
    limit = 4 * 1024**3
  budget = limit * headroom - param_bytes
  axes = [
      replace(N_STRUCTURES, cardinality=max(1, getattr(spec, "batch_size", 1) or 1)),
      replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
  ]
  planner = BatchPlanner(
      axes=axes,
      budget_bytes=budget,
      estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, activation_multiplier),
  )
  return planner.plan()


def score(
  spec: ScoringSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Score all provided sequences against all input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then scores all provided sequences against each structure.

  Args:
      spec: An optional ScoringSpecification object. If None, a default will be created using
      kwargs, options are provided as keyword arguments. The following options can be set:
        inputs: A single or sequence of inputs (files, PDB IDs, etc.).
        sequences_to_score: A list of protein sequences (strings) to score.
        chain_id: Specific chain(s) to parse from the structure.
        model: The model number to load. If None, all models are loaded.
        altloc: The alternate location identifier to use.
        model_version: The model version to use.
        model_weights: The model weights to use.
        foldcomp_database: The FoldComp database to use for FoldComp IDs.
        random_seed: The random number generator key.
        backbone_noise: The amount of noise to add to the backbone.
        ar_mask: An optional array of shape (L, L) to mask out certain residue pairs.
        batch_size: The number of structures to process in a single batch.
      **kwargs: Additional keyword arguments for structure loading.


  Returns:
      A dictionary containing scores, logits, and metadata.

  """
  if spec is None:
    kw = dict(kwargs)
    pop_deprecated_spec_kwargs(kw)
    spec = ScoringSpecification(**kw)

  _plan = _make_scoring_planner(spec)
  _plan.log_summary()
  if _plan.exceeded_budget():
    _batch_logger.warning(
        "score: BatchPlan exceeded budget even at minimum tiles.",
    )

  if spec.output_h5_path:
    return _score_streaming(spec, _plan)

  if not spec.sequences_to_score:
    msg = (
      "No sequences provided for scoring. `sequences_to_score` must be a non-empty list of strings."
    )
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.array(integer_sequences)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  if spec.average_node_features:
    all_scores, all_logits = _score_averaged_mode(
      spec,
      protein_iterator,
      model,
      batched_sequences,
    )
  else:
    all_scores, all_logits = _score_standard_mode(
      spec,
      protein_iterator,
      model,
      batched_sequences,
      _plan,
    )

  if not all_scores:
    return {}

  StreamingBatchHost.sink_barrier()
  return {
    "scores": jnp.concatenate(all_scores, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }


def _score_averaged_mode(
  spec: ScoringSpecification,
  protein_iterator: Any,  # noqa: ANN401
  model: PrxteinMPNN,
  batched_sequences: jax.Array,
) -> tuple[list[jnp.ndarray], list[Logits]]:
  """Run scoring in averaged node features mode."""
  spec_dict = asdict(spec)
  sampling_fields = {f.name for f in fields(SamplingSpecification)}
  filtered_spec = {k: v for k, v in spec_dict.items() if k in sampling_fields}
  pop_deprecated_spec_kwargs(filtered_spec)
  sampling_spec = SamplingSpecification(**filtered_spec)
  # Phase 5g follow-up: optional host-only accumulation (skip device concat) once sinks consume tensors end-to-end.
  all_scores, all_logits = [], []
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)

  for batch_idx, batched_ensemble in enumerate(protein_iterator):
    scores, logits = _score_batch_averaged(
      sampling_spec,
      batched_ensemble,
      model,
      batched_sequences,
      batch_idx,
      structure_batch_count,
    )
    all_scores.append(scores)
    all_logits.append(logits)

  return all_scores, all_logits


def _score_standard_mode(
  spec: ScoringSpecification,
  protein_iterator: Any,  # noqa: ANN401
  model: PrxteinMPNN,
  batched_sequences: jax.Array,
  plan: BatchPlan,
) -> tuple[list[jnp.ndarray], list[Logits]]:
  """Run scoring in standard mode."""
  score_single_pair = ScoringDriver(spec).build_score_single_pair(model)
  # Phase 5g PR3b: per-batch tensor io_callback; device-side lists and concat return unchanged.
  all_scores, all_logits = [], []
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)

  for batch_idx, batched_ensemble in enumerate(protein_iterator):
    tie_group_map = None
    if spec.tie_group_map is not None:
      tie_group_map = jnp.asarray(spec.tie_group_map, dtype=jnp.int32)
    elif spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
      tie_group_map = resolve_tie_groups(spec, batched_ensemble)

    max_len = batched_ensemble.coordinates.shape[1]
    if spec.ar_mask is None:
      current_ar_mask = 1 - jnp.eye(max_len, dtype=jnp.bool_)
    else:
      current_ar_mask = jnp.asarray(spec.ar_mask)

    batch_size = batched_ensemble.coordinates.shape[0]

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
    if mapping_for_vmap is not None and mapping_for_vmap.ndim == 1:
      mapping_for_vmap = jnp.broadcast_to(
        jnp.atleast_2d(mapping_for_vmap),
        (batch_size, mapping_for_vmap.shape[0]),
      )

    vmap_sequences = jax.vmap(
      score_single_pair,
      in_axes=(None, 0, None, None, None, None, None, None, None, None),
      out_axes=0,
    )
    vmap_noises = jax.vmap(
      vmap_sequences,
      in_axes=(None, None, None, None, None, None, 0, None, None, None),
      out_axes=0,
    )

    batch_idx_j = jnp.asarray(batch_idx, dtype=jnp.int32)
    batch_cnt_j = jnp.asarray(structure_batch_count, dtype=jnp.int32)

    def _score_one_structure(structure_batch_args):
      """Score one structure via closure over shared args."""
      (
          coords, mask, residue_idx, chain_idx, mapping_val, tie_map_val,
      ) = structure_batch_args
      return vmap_noises(  # noqa: B023
          jax.random.key(spec.random_seed),
          batched_sequences,
          coords,
          mask,
          residue_idx,
          chain_idx,
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          current_ar_mask,  # noqa: B023
          mapping_val,
          tie_map_val,
      )

    # Gather batched arguments (in_axes=0 from original vmap_structures)
    batched_structure_args = (
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        mapping_for_vmap,
        tie_map_for_vmap,
    )

    _structures_bs = plan.decision_for("n_structures").batch_size
    scores, logits, decoding_orders = safe_map(
        _score_one_structure,
        batched_structure_args,
        batch_size=_structures_bs,
    )

    _scoring_structure_and_tensor_batch_io(batch_idx_j, batch_cnt_j, scores, logits)

    all_scores.append(scores)
    all_logits.append(logits)

  return all_scores, all_logits


def _score_batch_averaged(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: PrxteinMPNN,
  sequences: ProteinSequence,
  batch_idx: int,
  structure_batch_count: int,
) -> tuple[jnp.ndarray, Logits]:
  """Score sequences for a batched ensemble of proteins using averaged encodings."""
  tie_group_map = None
  if spec.tie_group_map is not None:
    tie_group_map = jnp.asarray(spec.tie_group_map, dtype=jnp.int32)
  elif spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
    tie_group_map = resolve_tie_groups(spec, batched_ensemble)

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

  def score_single_sequence(seq: ProteinSequence, enc: tuple) -> tuple[jnp.ndarray, Logits]:
    avg_node, avg_edge, neighbors, mask, ar_mask_struct = enc

    # neighbors has shape (..., L, K). avg_node has shape (L, D).
    # Flatten batch dimensions of neighbors
    neighbors_flat = neighbors.reshape(
      (-1, neighbors.shape[-2], neighbors.shape[-1]),
    )
    mask_flat = mask.reshape((-1, mask.shape[-1]))
    ar_mask_struct_flat = ar_mask_struct.reshape(
      (-1, ar_mask_struct.shape[-2], ar_mask_struct.shape[-1]),
    )

    def score_one(
      n_idx: jnp.ndarray,
      m: jnp.ndarray,
      ar_m: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Logits, jnp.ndarray]:
      # score_sequence_with_encoding returns (score, logits, decoding_order)
      return cast(
        "tuple[jnp.ndarray, Logits, jnp.ndarray]",
        score_sequence_with_encoding(
          model,
          seq,
          (avg_node, avg_edge, n_idx, m, ar_m),
          tie_group_map=tie_group_map,
          multi_state_strategy=spec.multi_state_strategy,
          multi_state_temperature=spec.multi_state_temperature,
        ),
      )

    scores_batch, logits_batch, _ = jax.vmap(score_one)(
      neighbors_flat,
      mask_flat,
      ar_mask_struct_flat,
    )
    return jnp.mean(scores_batch, axis=0), jnp.mean(logits_batch, axis=0)

  if spec.average_encoding_mode == "inputs_and_noise":
    vmap_score = jax.vmap(score_single_sequence, in_axes=(0, None))
    scores, logits = vmap_score(sequences, averaged_encodings)
    scores = jnp.expand_dims(scores, axis=0)
    logits = jnp.expand_dims(logits, axis=0)
  else:
    # Determine structural axis to map over (the one NOT averaged)
    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    # vmap over sequences (axis 0)
    # vmap over outer batch (axis 0 of enc_node, axis struct_axis of enc_neighbors)

    vmap_score = jax.vmap(
      jax.vmap(score_single_sequence, in_axes=(0, None)),
      in_axes=(None, (0, 0, struct_axis, struct_axis, struct_axis)),
    )
    scores, logits = vmap_score(sequences, averaged_encodings)

  batch_idx_j = jnp.asarray(batch_idx, dtype=jnp.int32)
  batch_cnt_j = jnp.asarray(structure_batch_count, dtype=jnp.int32)
  _scoring_structure_and_tensor_batch_io(batch_idx_j, batch_cnt_j, scores, logits)
  return scores, logits


def _score_streaming(  # noqa: PLR0915
  spec: ScoringSpecification,
  plan: BatchPlan,
) -> dict[str, Any]:
  """Score sequences and stream results to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.array(integer_sequences)
  if batched_sequences.ndim == 1:
    batched_sequences = jnp.expand_dims(batched_sequences, 0)

  protein_iterator, model = prep_protein_stream_and_model(spec)
  score_single_pair = ScoringDriver(spec).build_score_single_pair(model)
  structure_batch_count = StreamingBatchHost.structure_batch_count(protein_iterator)

  with h5py.File(spec.output_h5_path, "w") as f:
    scores_ds = f.create_dataset(
      "scores",
      (0,),
      maxshape=(None,),
      chunks=True,
      dtype="f4",
    )
    logits_ds: h5py.Dataset | None = None

    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      tie_group_map = None
      if spec.tie_group_map is not None:
        tie_group_map = jnp.asarray(spec.tie_group_map, dtype=jnp.int32)
      elif spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
        tie_group_map = resolve_tie_groups(spec, batched_ensemble)

      max_len = batched_ensemble.coordinates.shape[1]
      if spec.ar_mask is None:
        current_ar_mask = 1 - jnp.eye(max_len, dtype=jnp.bool_)
      else:
        current_ar_mask = jnp.asarray(spec.ar_mask)

      batch_size = batched_ensemble.coordinates.shape[0]

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
      if mapping_for_vmap is not None and mapping_for_vmap.ndim == 1:
        mapping_for_vmap = jnp.broadcast_to(
          jnp.atleast_2d(mapping_for_vmap),
          (batch_size, mapping_for_vmap.shape[0]),
        )

      vmap_sequences = jax.vmap(
        score_single_pair,
        in_axes=(None, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
      )
      vmap_noises = jax.vmap(
        vmap_sequences,
        in_axes=(None, None, None, None, None, None, 0, None, None, None),
        out_axes=0,
      )

      batch_idx_j = jnp.asarray(batch_idx, dtype=jnp.int32)
      batch_cnt_j = jnp.asarray(structure_batch_count, dtype=jnp.int32)

      def _score_one_structure(structure_batch_args):
        coords, mask, residue_idx, chain_idx, mapping_val, tie_map_val = structure_batch_args
        return vmap_noises(  # noqa: B023
          jax.random.key(spec.random_seed),
          batched_sequences,
          coords,
          mask,
          residue_idx,
          chain_idx,
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          current_ar_mask,  # noqa: B023
          mapping_val,
          tie_map_val,
        )

      batched_structure_args = (
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        mapping_for_vmap,
        tie_map_for_vmap,
      )
      _structures_bs = plan.decision_for("n_structures").batch_size
      scores, logits, _ = safe_map(
        _score_one_structure,
        batched_structure_args,
        batch_size=_structures_bs,
      )
      _scoring_structure_and_tensor_batch_io(batch_idx_j, batch_cnt_j, scores, logits)

      StreamingBatchHost.sink_barrier()

      scores_ds.resize(scores_ds.shape[0] + scores.size, axis=0)
      scores_ds[-scores.size :] = scores.flatten()

      # Tensor hook above already D2H copies for observability; ``np.asarray`` below is a second
      # host transfer until streaming HDF5 is unified with the callback path (follow-up).
      logits_np = np.asarray(logits, dtype=np.float32)
      if logits_ds is None:
        logits_ds = f.create_dataset(
          "logits",
          (0, *logits_np.shape[1:]),
          maxshape=(None, *logits_np.shape[1:]),
          chunks=True,
          dtype="f4",
        )
      elif tuple(logits_ds.shape[1:]) != tuple(logits_np.shape[1:]):
        msg = (
          "Inconsistent logits shape across scoring batches: "
          f"expected trailing shape {tuple(logits_ds.shape[1:])}, "
          f"got {tuple(logits_np.shape[1:])}."
        )
        raise ValueError(msg)
      logits_ds.resize(logits_ds.shape[0] + logits_np.shape[0], axis=0)
      logits_ds[-logits_np.shape[0] :, ...] = logits_np

      f.flush()

    if logits_ds is None:
      f.create_dataset("logits", (0,), maxshape=(None,), chunks=True, dtype="f4")

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
