"""Host-side factory for constructing InferenceBundle and associated configs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.inference.schedule_selector import DecodingSchedule, build_wave_schedule
from aminx.tiling.bucketing import BucketingConfig, select_bucket
from aminx.tiling.pad import pad_bundle
from aminx.types.bundles import (
  ConditioningBundle,
  GeometryBundle,
  InferenceBundle,
  LigandBundle,
  PackerBundle,
  WaveScheduleBundle,
)
from aminx.types.configs import InferenceConfig
from aminx.utils.autoregression import generate_ar_mask, generate_wave_ar_mask


def _scale_packer_bundle(packer: PackerBundle, num_states: int) -> PackerBundle:
  """Scale packer bundle to S-state conformation dimension by prepending and broadcasting S axis.

  If packer.sequence.ndim == 1 and num_states > 1, prepend and broadcast an S axis to all its JAX arrays
  to scale them to the conformation state dimension.
  """
  if packer.sequence.ndim == 1 and num_states > 1:
    def _broadcast(x: object) -> object:
      if isinstance(x, jax.Array) and x.ndim > 0:
        return jnp.broadcast_to(x[None, ...], (num_states, *x.shape))
      return x
    return jax.tree.map(_broadcast, packer)
  return packer


def build_inference_bundle(
  coords: jax.Array,  # (L, 4, 3) or (S, L, 4, 3)
  mask: jax.Array,  # (L,) or (S, L)
  residue_index: jax.Array,
  chain_index: jax.Array,
  *,
  backbone_noise: float | jax.Array = 0.0,
  backbone_noise_mode: str = "direct",
  sequence: jax.Array | None = None,
  ar_mask: jax.Array | None = None,
  bias: jax.Array | None = None,
  fixed_mask: jax.Array | None = None,
  fixed_tokens: jax.Array | None = None,
  tie_group_map: jax.Array | None = None,
  state_weights: jax.Array | None = None,
  ligand_coords: jax.Array | None = None,
  ligand_atom_types: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
  structure_mapping: jax.Array | None = None,
  physics_features: jax.Array | None = None,
  atom_37: jax.Array | None = None,
  atom_37_mask: jax.Array | None = None,
  chain_mask: jax.Array | None = None,
  packer: PackerBundle | None = None,
  temperature: float = 1.0,
  mode: str = "score_conditional",
  inference: bool = True,
  bucket_config: BucketingConfig | None = None,
  schedule: DecodingSchedule = "fixed_n_to_c",
  schedule_key: jax.Array | None = None,
  schedule_k_neighbors: int = 48,
  wave: WaveScheduleBundle | None = None,
) -> tuple[InferenceBundle, InferenceConfig]:
  """Single entry point for bundle construction from raw arrays.

  `schedule` selects the decoding-wave schedule (W0.3; see
  `aminx.inference.schedule_selector`): `fixed_n_to_c` (default, matches the
  prior hardcoded sequential behavior byte-for-byte), `random_ar`,
  `frozen_random_sigma`, `chromatic`, or `improper_coloring`. The resulting
  `WaveScheduleBundle` is always built from the *same* resolved `tie_group_map`
  used for `ConditioningBundle`, so the two are never inconsistent. When
  `ar_mask` is not explicitly supplied and `mode != "score_conditional"`, the
  autoregressive visibility mask is derived from this schedule via
  `generate_wave_ar_mask` (positions in earlier waves are visible; positions in
  the same wave but different tie groups are not).

  `wave` overrides `schedule` entirely with a caller-supplied `WaveScheduleBundle`
  (e.g. a custom decoding order built via `WaveScheduleBundle.from_tie_groups`
  for an experiment-specific counterfactual schedule). Like the four non-default
  `schedule` arms, this is HOST-SIDE ONLY if `wave` itself was built with
  host-only constructors -- jit/vmap-safety depends on how `wave` was built, not
  on this parameter. Its `group_ids` must live in `tie_group_map`'s id space
  (true for `from_tie_groups`/`from_colors`-built bundles; NOT true for
  `WaveScheduleBundle.empty()`, which ignores ties by design).
  """
  # 1. Resolve shapes
  if coords.ndim == 3:
    coords = coords[None, ...]
    mask = mask[None, ...]
    residue_index = residue_index[None, ...]
    chain_index = chain_index[None, ...]
    if structure_mapping is not None:
      structure_mapping = structure_mapping[None, ...]
    if ligand_coords is not None:
      ligand_coords = ligand_coords[None, ...]
      ligand_atom_types = ligand_atom_types[None, ...]
      ligand_mask = ligand_mask[None, ...]
    if physics_features is not None and physics_features.ndim == 2:
      physics_features = physics_features[None, ...]
    if atom_37 is not None and atom_37.ndim == 3:
      atom_37 = atom_37[None, ...]
    if atom_37_mask is not None and atom_37_mask.ndim == 2:
      atom_37_mask = atom_37_mask[None, ...]
    if chain_mask is not None and chain_mask.ndim == 1:
      chain_mask = chain_mask[None, ...]

  # After normalization, all arrays must be 4D (or 2D for mask/indices when already batched)
  # Coords: always (S, L, 4, 3)
  # Mask/residue_index/chain_index: always (S, L) after normalization
  # chain_mask: always (S, L) after normalization if provided
  assert coords.ndim == 4, f"Expected coords.ndim == 4 after normalization, got {coords.ndim}"
  assert mask.ndim == 2, f"Expected mask.ndim == 2 after normalization, got {mask.ndim}"
  assert residue_index.ndim == 2, (
    f"Expected residue_index.ndim == 2 after normalization, got {residue_index.ndim}"
  )
  assert chain_index.ndim == 2, (
    f"Expected chain_index.ndim == 2 after normalization, got {chain_index.ndim}"
  )

  num_states, seq_len = coords.shape[0], coords.shape[1]

  # 2. Geometry
  geo = GeometryBundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    n_states=num_states,
    n_canonical=seq_len,
    n_flat=seq_len,
    structure_mapping=structure_mapping,
    physics_features=physics_features,
    atom_37=atom_37,
    atom_37_mask=atom_37_mask,
    chain_mask=chain_mask,
  )

  # 3. Conditioning
  if state_weights is None:
    state_weights = jnp.ones(num_states) / num_states

  if tie_group_map is None:
    tie_group_map = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (num_states, seq_len))
  elif tie_group_map.ndim == 1:
    tie_group_map = jnp.broadcast_to(tie_group_map[None, :], (num_states, seq_len))

  # 2b. Wave schedule (W0.3).
  #
  # IMPORTANT (jit/vmap safety): `schedule="fixed_n_to_c"` (the default) stays
  # on `WaveScheduleBundle.empty(seq_len)` -- a pure, shape-only construction
  # with no data-dependent Python control flow -- because `build_inference_bundle`
  # is called from inside jit/vmap-traced contexts today (host/kernel_dispatch.py's
  # unified driver, sampling/sample.py's make_sample_sequences). The other four
  # arms (`build_wave_schedule` / `WaveScheduleBundle.from_tie_groups` /
  # `.from_colors` / the coloring adjacency builder) use `.tolist()`/Python
  # loops/sets and are HOST-SIDE ONLY -- only request them when calling
  # `build_inference_bundle` outside any jax.jit/vmap trace.
  wave_explicitly_supplied = wave is not None
  if wave is not None:
    pass
  elif schedule == "fixed_n_to_c":
    wave = WaveScheduleBundle.empty(seq_len)
  else:
    resolved_schedule_key = schedule_key if schedule_key is not None else jax.random.PRNGKey(0)
    if schedule in ("chromatic", "improper_coloring"):
      lig_coords_flat = ligand_coords[0].reshape(-1, 3) if ligand_coords is not None else None
      lig_mask_flat = ligand_mask[0].reshape(-1) if ligand_mask is not None else None
      wave = build_wave_schedule(
        schedule,
        key=resolved_schedule_key,
        tie_group_map=tie_group_map[0],
        coords=coords[0],
        mask=mask[0],
        k_neighbors=schedule_k_neighbors,
        structure_mapping=structure_mapping[0] if structure_mapping is not None else None,
        ligand_coords=lig_coords_flat,
        ligand_mask=lig_mask_flat,
      )
    else:
      wave = build_wave_schedule(schedule, key=resolved_schedule_key, tie_group_map=tie_group_map[0])

  if ar_mask is None:
    if mode == "score_conditional":
      # Default to full context minus self (all-ones except diagonal)
      ar_mask = 1.0 - jnp.eye(seq_len)
      ar_mask = jnp.broadcast_to(ar_mask[None, ...], (num_states, seq_len, seq_len))
    elif schedule == "fixed_n_to_c" and not wave_explicitly_supplied:
      # generate_ar_mask is pure-jnp (jit/vmap-safe) and tie-group-aware via
      # tie_group_map directly -- unlike `wave` (which stays untied/empty()
      # for jit-safety above), this is correct for tied positions: positions
      # sharing a tie group always see each other regardless of order.
      ar_mask_2d = generate_ar_mask(jnp.arange(seq_len, dtype=jnp.int32), tie_group_map=tie_group_map[0])
      ar_mask = jnp.broadcast_to(ar_mask_2d.astype(jnp.float32)[None, ...], (num_states, seq_len, seq_len))
    else:
      # wave was either built host-side above (from_tie_groups/from_colors) or
      # explicitly supplied by the caller -- both live in tie_group_map's id
      # space, so generate_wave_ar_mask is correct here (unlike for the
      # empty()-schedule branch above).
      ar_mask_2d = generate_wave_ar_mask(wave, tie_group_map[0])
      ar_mask = jnp.broadcast_to(ar_mask_2d[None, ...], (num_states, seq_len, seq_len))
  elif ar_mask.ndim == 2:
    ar_mask = jnp.broadcast_to(ar_mask[None, ...], (num_states, seq_len, seq_len))

  # Handle sequence: can be token indices or already one-hot
  if sequence is not None:
    if sequence.ndim == 1:
      # Token indices: convert to one-hot
      sequence_oh = jax.nn.one_hot(sequence, 21)
    elif sequence.ndim == 2 and sequence.shape[1] == 21:
      # Already one-hot
      sequence_oh = sequence
    else:
      raise ValueError(
        f"sequence must be shape (L,) for tokens or (L, 21) for one-hot, got {sequence.shape}",
      )
  else:
    sequence_oh = jnp.zeros((seq_len, 21))

  cond = ConditioningBundle(
    fixed_mask=fixed_mask if fixed_mask is not None else jnp.zeros(seq_len),
    fixed_tokens=fixed_tokens if fixed_tokens is not None else jnp.zeros(seq_len, dtype=jnp.int32),
    bias=bias if bias is not None else jnp.zeros((seq_len, 21)),
    tie_group_map=tie_group_map,
    state_weights=state_weights,
    sequence_oh=sequence_oh,
    ar_mask=ar_mask,
    temperature=jnp.array(temperature),
  )

  # 4. Ligand
  lig = LigandBundle(
    ligand_coords=ligand_coords if ligand_coords is not None else jnp.zeros((num_states, 0, 4, 3)),
    ligand_atom_types=ligand_atom_types
    if ligand_atom_types is not None
    else jnp.zeros((num_states, 0, 4), dtype=jnp.int32),
    ligand_mask=ligand_mask if ligand_mask is not None else jnp.zeros((num_states, 0, 4)),
  )

  # 5. Assemble Bundle
  if packer is not None:
    packer = _scale_packer_bundle(packer, num_states)

  bundle = InferenceBundle(
    geometry=geo,
    conditioning=cond,
    ligand=lig,
    wave=wave,
    packer=packer,
    backbone_noise=jnp.array(backbone_noise),
  )

  # 5b. Pad to bucket ceiling if requested (fixes per-seq-len XLA recompilation)
  if bucket_config is not None:
    target_length = select_bucket(seq_len, bucket_config)
    bundle = pad_bundle(bundle, target_length)

  # 6. Config
  config = InferenceConfig(
    mode=mode,
    backbone_noise_mode=backbone_noise_mode,
    inference=inference,
  )

  return bundle, config
