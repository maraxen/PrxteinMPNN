"""Kernel dispatch for sampling batch execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.experimental
import jax.numpy as jnp

from prxteinmpnn.host._sampling_grid_lineage import _base_sampling_key, _resolve_grid_lineage
from prxteinmpnn.host._sampling_helper import (
  _broadcast_per_structure,
  _dispatch_sampling_tensor_batch_io,
  _noop_sampling_structure_batch_io,
  _prepare_fixed_controls,
  _prepare_ligand_context,
)
from prxteinmpnn.host.logit_aggregation import compute_pseudo_perplexity
from prxteinmpnn.host.plan import (
  compute_sample_keys,
  extract_batch_sizes,
  make_sampling_planner,
  resolve_target_samples,
)
from prxteinmpnn.inference import sample_autoregressive as sample_ar
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.utils.safe_map import safe_map as _safe_map

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.types.arrays import (
    Logits,
    ProteinSequence,
  )
  from prxteinmpnn.types.stages import StageSet
  from prxteinmpnn.utils.data_structures import Protein


def resolve_kernel_fn(strategy: str) -> Callable:
  """Resolve a kernel callable from sampling_strategy string.

  Dispatches to the appropriate inference kernel based on the sampling strategy:
  - 'temperature': Autoregressive sampling kernel
  - 'straight_through': Straight-through estimator wrapped teacher-forced kernel
  - Default: Autoregressive sampling

  Args:
    strategy: Sampling strategy name (e.g., 'temperature', 'straight_through')

  Returns:
    A callable with signature (model, key, bundle, config, stage_set) -> SampleResult
  """
  if strategy == "temperature":
    return sample_ar.kernel
  if strategy == "straight_through":
    # For STE, wrap score_conditional kernel to return SampleResult compatible interface
    from prxteinmpnn.inference.sample_autoregressive import SampleResult
    from prxteinmpnn.inference.score_conditional import kernel as score_conditional_kernel

    def _ste_kernel_wrapper(model, prng_key, bundle, config, stage_set):
      """Wrap score_conditional to return SampleResult-compatible interface."""
      # Call score_conditional kernel (teacher-forced decoding)
      logits = score_conditional_kernel(model, prng_key, bundle, config, stage_set)
      # Compute sequence from logits via argmax (greedy decoding)
      sequence = logits.argmax(axis=-1).astype(jnp.int32)
      return SampleResult(sequence=sequence, logits=logits)

    return _ste_kernel_wrapper
  # Default to temperature strategy
  return sample_ar.kernel


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: ModelProtocol,
  *,
  stage_set: StageSet,
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

  # 3. Resolve the kernel function from spec.sampling_strategy
  _kernel_fn = resolve_kernel_fn(spec.sampling_strategy)

  # 4. Inner Kernel Closure (Single Structure, Single Noise, Single Temp)
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
      bundle, config = build_inference_bundle(
          coords=c, mask=m, residue_index=ri, chain_index=ci,
          backbone_noise=noise_val,
          fixed_mask=fm, fixed_tokens=ft,
          bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
          state_weights=state_weights,
          ligand_coords=ligand_context["y"][structure_idx] if ligand_context["y"] is not None else None,
          ligand_atom_types=ligand_context["y_t"][structure_idx] if ligand_context["y_t"] is not None else None,
          ligand_mask=ligand_context["y_m"][structure_idx] if ligand_context["y_m"] is not None else None,
          structure_mapping=mapping_for_vmap[structure_idx] if mapping_for_vmap is not None else None,
          temperature=temp_val,
          mode="sample_ar",
          use_rolling_state=spec.use_rolling_state,
          inference=True,
      )

      # Map over samples
      def _run_one_sample(k):
          res = _kernel_fn(model, k, bundle, config, stage_set)
          return res.sequence, res.logits

      return _safe_map(_run_one_sample, key_samples, batch_size=samples_bs)

  # 5. Nested Dispatch (Structures -> Noises -> Temps)
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
      batch_size=structures_bs,
  )

  # 6. Post-process (transpose to expected output shape: [batch, samples, noise, temp, seq_len])
  # current: [B, D, T, N, L] -> desired: [B, N, D, T, L]
  sampled_sequences = jnp.transpose(sampled_sequences, (0, 3, 1, 2, 4))
  sampled_logits = jnp.transpose(sampled_logits, (0, 3, 1, 2, 4, 5))

  # 7. io_callback emission — stage tensors to active sink (if any)
  _effective_chunk_start = chunk_sample_start if chunk_sample_start is not None else 0

  jax.experimental.io_callback(
      _dispatch_sampling_tensor_batch_io,
      None,
      jnp.int32(batch_idx),
      jnp.int32(structure_batch_count),
      jnp.int32(_effective_chunk_start),
      jnp.int32(target_num_samples),
      sampled_sequences,
      sampled_logits,
      ordered=False,
  )

  if emit_structure_batch_io:
    jax.experimental.io_callback(
        _noop_sampling_structure_batch_io,
        None,
        jnp.int32(batch_idx),
        jnp.int32(structure_batch_count),
        ordered=False,
    )

  # 8. IO & Metadata
  if spec.compute_pseudo_perplexity:
    mask = batched_ensemble.mask
    if mask is None:
      mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    pseudo_perplexity = compute_pseudo_perplexity(sampled_logits, sampled_sequences, mask)
    return sampled_sequences, sampled_logits, pseudo_perplexity

  return sampled_sequences, sampled_logits, None
