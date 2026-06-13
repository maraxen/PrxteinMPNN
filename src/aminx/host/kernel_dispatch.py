"""Kernel dispatch for sampling batch execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.experimental
import jax.numpy as jnp

from aminx.host._sampling_grid_lineage import _base_sampling_key, _resolve_grid_lineage
from aminx.host._sampling_helper import (
  _broadcast_per_structure,
  _dispatch_sampling_tensor_batch_io,
  _noop_sampling_structure_batch_io,
  _prepare_fixed_controls,
  _prepare_ligand_context,
)
from aminx.host.logit_aggregation import compute_pseudo_perplexity
from aminx.host.plan import (
  compute_sample_keys,
  extract_batch_sizes,
  make_sampling_planner,
  resolve_target_samples,
)
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.run.specs import SamplingSpecification
from aminx.utils.safe_map import safe_map as _safe_map

if TYPE_CHECKING:
  from collections.abc import Sequence

  from aminx.host.plan import InferencePlan
  from aminx.types.arrays import (
    Logits,
    ProteinSequence,
  )
  from aminx.utils.data_structures import Protein


def _dispatch_axis(strategy, body, xs, *, batch_size_fallback: int = 0):
  """Dispatch iteration over an axis using the declared AxisStrategy.

  Args:
      strategy: AxisStrategy from BatchPlanner — Vmap, SafeMap, Scan, or DedupGather.
      body: Function to apply at each element. For Vmap/SafeMap: body(x) -> y.
            For Scan: called as body(x) -> y; carry is threaded around it.
            For DedupGather: body(x) -> y; run on K unique elements then scatter to N.
      xs: Input array or pytree to map over (leading axis = the mapped axis).
      batch_size_fallback: Used only if strategy is not an AxisStrategy instance
          (backward compat with callers that don't have the strategy field yet).

  Returns:
      Stacked results, same leading shape as xs.

  """
  from aminx.tiling.strategy import DedupGather, SafeMap, Scan, Vmap
  from aminx.utils.safe_scan import safe_scan

  if isinstance(strategy, Vmap):
    return jax.vmap(body)(xs)
  if isinstance(strategy, SafeMap):
    return _safe_map(body, xs, batch_size=strategy.tile)
  if isinstance(strategy, Scan):
    # Wrap body to be scan-compatible: body is (x -> y), transition threads carry.
    # For default case (init=None, no real carry), this is a carry-passthrough.
    init = strategy.init if strategy.init is not None else jnp.array(0)

    def scan_body(carry, x):
      y = body(x)
      return carry, y  # carry-passthrough: carry is not updated

    _, ys = safe_scan(scan_body, xs, init=init)
    return ys
  if isinstance(strategy, DedupGather):
    unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)  # (K_bucket,)
    index_map = jnp.asarray(strategy.index_map, dtype=jnp.int32)  # (N,)
    xs_unique = strategy.dedup_fn(xs, unique_idx)  # in-trace gather
    ys_unique = _safe_map(body, xs_unique, batch_size=None)  # K_bucket runs
    return strategy.gather_fn(ys_unique, index_map)  # in-trace scatter
  # Fallback: treat as safe_map with batch_size_fallback
  return _safe_map(body, xs, batch_size=batch_size_fallback)


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  plan: InferencePlan,
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
  batch_plan = make_sampling_planner(spec)
  batch_plan.log_summary()

  structures_bs, samples_bs, temps_bs, noises_bs = extract_batch_sizes(batch_plan)

  # 2. Resolve Grids
  grid_lineage = _resolve_grid_lineage(spec)
  base_key = _base_sampling_key(spec, grid_lineage=grid_lineage)
  target_num_samples = resolve_target_samples(spec, chunk_sample_count, grid_lineage)

  noises = jnp.asarray(spec.backbone_noise)
  temperatures = jnp.asarray(spec.temperature)

  seq_len = batched_ensemble.coordinates.shape[1]
  batch_size = batched_ensemble.coordinates.shape[0]

  # The unified driver indexes these per-structure arrays by a (possibly traced)
  # structure index under vmap. Convert to JAX arrays so traced indexing lowers to a
  # gather instead of triggering numpy's __array__ on a tracer (TracerArrayConversionError).
  coords_for_vmap = jnp.asarray(batched_ensemble.coordinates)
  mask_for_vmap = jnp.asarray(batched_ensemble.mask)
  residue_index_for_vmap = jnp.asarray(batched_ensemble.residue_index)
  chain_index_for_vmap = jnp.asarray(batched_ensemble.chain_index)

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

  # 3. Compute deterministic sample keys
  sample_keys = compute_sample_keys(
    base_key,
    target_num_samples,
    chunk_sample_start=chunk_sample_start,
    grid_lineage_sample_start=grid_lineage["sample_start"] if grid_lineage is not None else None,
  )

  # 4. Dispatch — two paths based on whether encoding_fusion is wired (Python-level static check)
  # Check spec for unified driver flag (defaults to True since S5-D10; legacy path kept as fallback)
  _use_unified = getattr(spec, "use_unified_driver", False)

  if _use_unified and plan.stage_set.encoding_fusion is None:
    # -------------------------------------------------------------------------
    # Unified Path A: strategy-dispatched axis iteration using AxisDecision.strategy
    # Reads strategy from batch_plan.decision_for(axis_name) instead of batch_size ints.
    # -------------------------------------------------------------------------
    struct_decision = batch_plan.decision_for("n_structures")
    noise_decision = batch_plan.decision_for("n_noises")
    temp_decision = batch_plan.decision_for("n_temperatures")
    sample_decision = batch_plan.decision_for("n_samples")

    def _unified_call_kernel(key_samples, structure_idx, noise_val, temp_val):
      c = coords_for_vmap[structure_idx]
      m = mask_for_vmap[structure_idx]
      ri = residue_index_for_vmap[structure_idx]
      ci = chain_index_for_vmap[structure_idx]
      fm = fixed_mask_for_vmap[structure_idx]
      ft = fixed_tokens_for_vmap[structure_idx]

      bundle, config = build_inference_bundle(
        coords=c,
        mask=m,
        residue_index=ri,
        chain_index=ci,
        backbone_noise=noise_val,
        fixed_mask=fm,
        fixed_tokens=ft,
        bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
        tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
        state_weights=state_weights,
        ligand_coords=ligand_context["Y"][structure_idx]
        if ligand_context["Y"] is not None
        else None,
        ligand_atom_types=ligand_context["Y_t"][structure_idx]
        if ligand_context["Y_t"] is not None
        else None,
        ligand_mask=ligand_context["Y_m"][structure_idx]
        if ligand_context["Y_m"] is not None
        else None,
        atom_37=ligand_context["atom_37"][structure_idx]
        if ligand_context["atom_37"] is not None
        else None,
        atom_37_mask=ligand_context["atom_37_mask"][structure_idx]
        if ligand_context["atom_37_mask"] is not None
        else None,
        structure_mapping=mapping_for_vmap[structure_idx] if mapping_for_vmap is not None else None,
        temperature=temp_val,
        mode="sample_ar",
        inference=True,
      )

      encode_key = jax.random.fold_in(base_key, structure_idx)
      enc = plan.encode(bundle, encode_key, config)

      for _sink in plan.stage_set.encoder_sink:
        _sink(enc, jnp.int32(batch_idx), structure_idx, jnp.int32(0))

      def _run_one_sample(k):
        res = plan.decode(enc, bundle, k, config)
        return res.sequence, res.logits

      return _dispatch_axis(sample_decision.strategy, _run_one_sample, key_samples)

    def _unified_dispatch_structure(s_idx):
      def _dispatch_noise(n_val):
        def _dispatch_temp(t_val):
          return _unified_call_kernel(sample_keys, s_idx, n_val, t_val)

        return _dispatch_axis(temp_decision.strategy, _dispatch_temp, temperatures)

      return _dispatch_axis(noise_decision.strategy, _dispatch_noise, noises)

    sampled_sequences, sampled_logits = _dispatch_axis(
      struct_decision.strategy,
      _unified_dispatch_structure,
      jnp.arange(batch_size),
    )

  elif _use_unified and plan.stage_set.encoding_fusion is not None:
    # -------------------------------------------------------------------------
    # Unified Path B: strategy-dispatched encoding fusion + decoding
    # -------------------------------------------------------------------------
    struct_decision = batch_plan.decision_for("n_structures")
    noise_decision = batch_plan.decision_for("n_noises")
    temp_decision = batch_plan.decision_for("n_temperatures")
    sample_decision = batch_plan.decision_for("n_samples")

    def _unified_call_structure_fused(structure_idx):
      c = coords_for_vmap[structure_idx]
      m = mask_for_vmap[structure_idx]
      ri = residue_index_for_vmap[structure_idx]
      ci = chain_index_for_vmap[structure_idx]
      fm = fixed_mask_for_vmap[structure_idx]
      ft = fixed_tokens_for_vmap[structure_idx]

      def _build_bundle(noise_val, temperature_val=jnp.float32(1.0)):
        return build_inference_bundle(
          coords=c,
          mask=m,
          residue_index=ri,
          chain_index=ci,
          backbone_noise=noise_val,
          fixed_mask=fm,
          fixed_tokens=ft,
          bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
          state_weights=state_weights,
          ligand_coords=ligand_context["Y"][structure_idx]
          if ligand_context["Y"] is not None
          else None,
          ligand_atom_types=ligand_context["Y_t"][structure_idx]
          if ligand_context["Y_t"] is not None
          else None,
          ligand_mask=ligand_context["Y_m"][structure_idx]
          if ligand_context["Y_m"] is not None
          else None,
          atom_37=ligand_context["atom_37"][structure_idx]
          if ligand_context["atom_37"] is not None
          else None,
          atom_37_mask=ligand_context["atom_37_mask"][structure_idx]
          if ligand_context["atom_37_mask"] is not None
          else None,
          structure_mapping=mapping_for_vmap[structure_idx]
          if mapping_for_vmap is not None
          else None,
          temperature=temperature_val,
          mode="sample_ar",
          inference=True,
        )

      # Step 1: encode at each noise level using the noise axis strategy
      noise_indices = jnp.arange(len(spec.backbone_noise), dtype=jnp.int32)

      def _encode_at_noise(noise_and_idx):
        noise_val, noise_idx = noise_and_idx
        bundle, config = _build_bundle(noise_val)
        encode_key = jax.random.fold_in(base_key, structure_idx)
        enc = plan.encode(bundle, encode_key, config)
        for _sink in plan.stage_set.encoder_sink:
          _sink(enc, jnp.int32(batch_idx), structure_idx, noise_idx)
        return enc

      stacked_enc = _dispatch_axis(
        noise_decision.strategy,
        _encode_at_noise,
        (noises, noise_indices),
      )

      # Step 2: fuse D encoded outputs → K
      fused_enc = plan.stage_set.encoding_fusion(stacked_enc)

      # Build decode bundle at noise=0 (fusion already handled noise variation)
      decode_bundle, _ = _build_bundle(jnp.float32(0.0))

      # Step 3: decode K times × temperatures × samples
      def _call_decode_one_enc(enc_k):
        def _dispatch_temp(temp_val):
          _, t_config = _build_bundle(jnp.float32(0.0), temperature_val=temp_val)

          def _run_one_sample(k):
            res = plan.decode(enc_k, decode_bundle, k, t_config)
            return res.sequence, res.logits

          return _dispatch_axis(sample_decision.strategy, _run_one_sample, sample_keys)

        return _dispatch_axis(temp_decision.strategy, _dispatch_temp, temperatures)

      # Map decode over K fused encodings (use _safe_map for this axis as it has no AxisDecision)
      return _safe_map(_call_decode_one_enc, fused_enc, batch_size=None)

    sampled_sequences, sampled_logits = _dispatch_axis(
      struct_decision.strategy,
      _unified_call_structure_fused,
      jnp.arange(batch_size),
    )

  elif plan.stage_set.encoding_fusion is None:
    # -------------------------------------------------------------------------
    # Path A: no fusion — standard encode-per-(structure, noise, temp) topology
    # -------------------------------------------------------------------------
    def _call_kernel(key_samples, structure_idx, noise_val, temp_val):
      c = coords_for_vmap[structure_idx]
      m = mask_for_vmap[structure_idx]
      ri = residue_index_for_vmap[structure_idx]
      ci = chain_index_for_vmap[structure_idx]
      fm = fixed_mask_for_vmap[structure_idx]
      ft = fixed_tokens_for_vmap[structure_idx]

      bundle, config = build_inference_bundle(
        coords=c,
        mask=m,
        residue_index=ri,
        chain_index=ci,
        backbone_noise=noise_val,
        fixed_mask=fm,
        fixed_tokens=ft,
        bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
        tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
        state_weights=state_weights,
        ligand_coords=ligand_context["Y"][structure_idx]
        if ligand_context["Y"] is not None
        else None,
        ligand_atom_types=ligand_context["Y_t"][structure_idx]
        if ligand_context["Y_t"] is not None
        else None,
        ligand_mask=ligand_context["Y_m"][structure_idx]
        if ligand_context["Y_m"] is not None
        else None,
        atom_37=ligand_context["atom_37"][structure_idx]
        if ligand_context["atom_37"] is not None
        else None,
        atom_37_mask=ligand_context["atom_37_mask"][structure_idx]
        if ligand_context["atom_37_mask"] is not None
        else None,
        structure_mapping=mapping_for_vmap[structure_idx] if mapping_for_vmap is not None else None,
        temperature=temp_val,
        mode="sample_ar",
        inference=True,
      )

      encode_key = jax.random.fold_in(base_key, structure_idx)
      enc = plan.encode(bundle, encode_key, config)

      for _sink in plan.stage_set.encoder_sink:
        _sink(enc, jnp.int32(batch_idx), structure_idx, jnp.int32(0))

      def _run_one_sample(k):
        res = plan.decode(enc, bundle, k, config)
        return res.sequence, res.logits

      return _safe_map(_run_one_sample, key_samples, batch_size=samples_bs)

    def _dispatch_structure(s_idx):
      def _dispatch_noise(n_val):
        def _dispatch_temp(t_val):
          return _call_kernel(sample_keys, s_idx, n_val, t_val)

        return _safe_map(_dispatch_temp, temperatures, batch_size=temps_bs)

      return _safe_map(_dispatch_noise, noises, batch_size=noises_bs)

    # shape: (B, D, T, N, L) and (B, D, T, N, L, 21)
    sampled_sequences, sampled_logits = _safe_map(
      _dispatch_structure,
      jnp.arange(batch_size),
      batch_size=structures_bs,
    )

  else:
    # -------------------------------------------------------------------------
    # Path B: with fusion — encode D times per structure, fuse → K, decode K×T×N
    # -------------------------------------------------------------------------
    def _call_structure_fused(structure_idx):
      c = coords_for_vmap[structure_idx]
      m = mask_for_vmap[structure_idx]
      ri = residue_index_for_vmap[structure_idx]
      ci = chain_index_for_vmap[structure_idx]
      fm = fixed_mask_for_vmap[structure_idx]
      ft = fixed_tokens_for_vmap[structure_idx]

      def _build_bundle(noise_val, temperature_val=jnp.float32(1.0)):
        return build_inference_bundle(
          coords=c,
          mask=m,
          residue_index=ri,
          chain_index=ci,
          backbone_noise=noise_val,
          fixed_mask=fm,
          fixed_tokens=ft,
          bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          tie_group_map=tie_map_for_vmap[structure_idx] if tie_map_for_vmap is not None else None,
          state_weights=state_weights,
          ligand_coords=ligand_context["Y"][structure_idx]
          if ligand_context["Y"] is not None
          else None,
          ligand_atom_types=ligand_context["Y_t"][structure_idx]
          if ligand_context["Y_t"] is not None
          else None,
          ligand_mask=ligand_context["Y_m"][structure_idx]
          if ligand_context["Y_m"] is not None
          else None,
          atom_37=ligand_context["atom_37"][structure_idx]
          if ligand_context["atom_37"] is not None
          else None,
          atom_37_mask=ligand_context["atom_37_mask"][structure_idx]
          if ligand_context["atom_37_mask"] is not None
          else None,
          structure_mapping=mapping_for_vmap[structure_idx]
          if mapping_for_vmap is not None
          else None,
          temperature=temperature_val,
          mode="sample_ar",
          inference=True,
        )

      # Step 1: encode at each noise level
      noise_indices = jnp.arange(len(spec.backbone_noise), dtype=jnp.int32)

      def encode_at_noise(noise_and_idx):
        noise_val, noise_idx = noise_and_idx
        bundle, config = _build_bundle(noise_val)
        encode_key = jax.random.fold_in(base_key, structure_idx)
        enc = plan.encode(bundle, encode_key, config)
        for _sink in plan.stage_set.encoder_sink:
          _sink(enc, jnp.int32(batch_idx), structure_idx, noise_idx)
        return enc

      stacked_enc = _safe_map(encode_at_noise, (noises, noise_indices), batch_size=noises_bs)

      # Step 2: fuse D encoded outputs → K
      fused_enc = plan.stage_set.encoding_fusion(stacked_enc)

      # Build decode bundle at noise=0 (fusion already handled noise variation)
      decode_bundle, _ = _build_bundle(jnp.float32(0.0))

      # Step 3: decode K times, then over temps, then over samples
      def _call_decode_one_enc(enc_k):
        def _dispatch_temp(temp_val):
          _, t_config = _build_bundle(jnp.float32(0.0), temperature_val=temp_val)

          def _run_one_sample(k):
            res = plan.decode(enc_k, decode_bundle, k, t_config)
            return res.sequence, res.logits

          return _safe_map(_run_one_sample, sample_keys, batch_size=samples_bs)

        return _safe_map(_dispatch_temp, temperatures, batch_size=temps_bs)

      # Map decode over K fused encodings
      return _safe_map(_call_decode_one_enc, fused_enc, batch_size=None)

    # shape: (B, K, T, N, L) and (B, K, T, N, L, 21)
    sampled_sequences, sampled_logits = _safe_map(
      _call_structure_fused,
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
