"""Genuine multi-state PoE-fused autoregressive sampling for a bead of k reference states.

Fixes a real gap found 2026-07-13 (tev_design task 260709_multistate-fusion-strategy-comparison,
Phase 3; see ../../.praxia/docs/decisions/260713_no-real-multistate-sampling-path-exists.md for
the full trace): `aminx campaign plan`/`run`'s real sampling dispatcher
(`host/kernel_dispatch.py::_sample_batch`) treats every `--inputs` path as an independent
`N_STRUCTURES` batch item and builds a single-state (`num_states=1`) bundle per call -- no
CLI-reachable path ever constructs a genuinely stacked `num_states>1` bundle for autoregressive
sampling, so `multi_state_strategy="product"` fusion never actually combines states in any real
campaign row.

This module is a new, separate, minimal-risk entry point (chosen over touching
`_sample_batch`/`kernel_dispatch.py` itself -- see the decision doc's option 1): it builds ONE
combined multi-state bundle from all of a bead's reference states (reusing the same structure
loading/padding/chain-filtering (`prep_protein_stream_and_model`) and bundle-building
(`build_inference_bundle`) machinery `_sample_batch` already uses per-structure, just without the
per-structure_idx slicing loop) and samples from it via `aminx.inference.sample_autoregressive`,
which already correctly wires `AutoregressiveMode` and genuinely fuses an `S>1` state axis
(confirmed by the existing `test_ar_decode_state_position_map_changes_fused_logits` test).

Does NOT touch `_sample_batch`/`kernel_dispatch.py`, `campaign.py`, or `bundle_builder.py` --
every existing campaign row (single-structure spike-in beads) is unaffected. Does not wire into
`aminx campaign plan/run`'s CLI or manifest/Zarr-writer machinery; that integration (and the more
general, xtrax-composable arbitrary-fusion-axis version of this idea) is real, separate follow-up
work, tracked as praxia debt #589.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.host._sampling_helper import _prepare_fixed_controls, _prepare_ligand_context
from aminx.host.prep import prep_protein_stream_and_model
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference.sample_autoregressive import kernel as _sample_autoregressive_kernel

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from aminx.model.mpnn import Aminx
  from aminx.run.specs import SamplingSpecification
  from aminx.types.bundles import InferenceBundle
  from aminx.types.configs import InferenceConfig
  from aminx.types.stages import StageSet


@eqx.filter_jit
def sample_states_fused(
  model: Aminx,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
  prng_key: PRNGKeyArray,
  n_samples: int,
) -> tuple[Int[Array, "n_samples L"], Float[Array, "n_samples L 21"]]:
  """Draw n_samples independent samples from an already-built, already-fused bundle.

  Thin vmap wrapper around `aminx.inference.sample_autoregressive.kernel` -- the genuine,
  already-tested `AutoregressiveMode` sampling path, which fuses `bundle`'s own state axis
  (`bundle.geometry.n_states`) via `stage_set.logit_transform`/`_realign_states_to_reference`
  exactly as `ConditionalDecode`'s teacher-forced scoring path does. Each sample gets its own
  PRNG key (`jax.random.split`); `bundle`/`config`/`stage_set` are shared and re-encoded once
  per sample by the kernel (matching `_sample_batch`'s own per-sample-key semantics).

  Parameters
  ----------
  model : Aminx
      Parameterized model.
  bundle : InferenceBundle
      Bundle to sample from -- may be single-state (num_states=1) or genuinely multi-state
      (num_states>1, pre-fused via a real `state_position_map`); this function does not care
      which, it just samples from whatever bundle it's given.
  config : InferenceConfig
      Paired with `bundle` (from the same `build_inference_bundle` call).
  stage_set : StageSet
      Fusion/decode configuration (e.g. from `make_stage_set(strategy="product", ...)`).
  prng_key : PRNGKeyArray
      Base key; split into `n_samples` independent per-sample keys.
  n_samples : int
      Number of independent samples to draw.

  Returns
  -------
  tuple
      (sequences, logits): sequences shape (n_samples, L) int32, logits shape
      (n_samples, L, 21) float32.

  """
  sample_keys = jax.random.split(prng_key, n_samples)

  def _one_sample(key: PRNGKeyArray) -> tuple[Any, Any]:
    result = _sample_autoregressive_kernel(model, key, bundle, config, stage_set)
    return result.sequence, result.logits

  sequences, logits = jax.vmap(_one_sample)(sample_keys)
  return sequences, logits


def sample_multistate_poe_bead(
  spec: SamplingSpecification,
  prng_key: PRNGKeyArray,
  n_samples: int,
) -> tuple[Int[Array, "n_samples L"], Float[Array, "n_samples L 21"]]:
  """Sample n_samples designs with genuine cross-state PoE fusion over spec.inputs' k states.

  Builds ONE combined multi-state bundle from all of `spec.inputs` (reusing the same
  structure-loading/padding pipeline `_sample_batch` uses per-structure) instead of `_sample_batch`'s
  real behavior of building `len(spec.inputs)` independent single-state bundles. `spec.state_position_map`
  (if set) is applied against this bundle's real `num_states=len(spec.inputs)` state axis, so
  `spec.multi_state_strategy` genuinely fuses across states -- unlike every existing campaign row.

  PRECONDITION: `spec.batch_size` must equal `len(spec.inputs)`, so the underlying protein
  dataset iterator yields exactly one batch covering every state (this function does not
  itself chunk or re-batch inputs; the caller is responsible for choosing a bead's k small
  enough that all states fit in one host batch, which the necklace campaign's k=4 always does).

  Only a single temperature and a single backbone-noise value are supported per call (the
  first entry of `spec.run_spec.sampling.temperature`/`backbone_noise` if either is a
  sequence) -- callers wanting a temperature/noise grid should call this once per value,
  mirroring how `build_necklace_p2_manifest.py` already treats those as manifest-row-level
  axes, not something this function's single call needs to dispatch internally.

  Parameters
  ----------
  spec : SamplingSpecification
      Real campaign-style spec -- same fields `_run_campaign_plan`/`run_manifest_row` already
      construct (`inputs`, `chain_id`, `checkpoint_id`, `multi_state_strategy`,
      `state_position_map`, `state_weights`, `tie_group_map`, etc).
  prng_key : PRNGKeyArray
      Base key for sampling (split internally into `n_samples` per-sample keys).
  n_samples : int
      Number of independent fused-PoE samples to draw for this bead.

  Returns
  -------
  tuple
      (sequences, logits) -- see `sample_states_fused`.

  Raises
  ------
  ValueError
      If `spec.inputs` has fewer than 2 entries, `spec.batch_size` doesn't match
      `len(spec.inputs)`, or the protein dataset iterator doesn't yield exactly one batch
      covering every input state (i.e. the precondition above was violated).

  """
  if not isinstance(spec.inputs, (list, tuple)) or len(spec.inputs) < 2:
    msg = (
      f"sample_multistate_poe_bead needs >=2 states in spec.inputs to fuse across "
      f"(got {spec.inputs!r}) -- for a single structure, use the existing campaign path."
    )
    raise ValueError(msg)
  n_states = len(spec.inputs)

  if spec.batch_size != n_states:
    msg = (
      f"spec.batch_size ({spec.batch_size}) must equal len(spec.inputs) ({n_states}) so the "
      "protein dataset iterator yields all states in one combined batch -- see this "
      "function's docstring precondition."
    )
    raise ValueError(msg)

  protein_iterator, model = prep_protein_stream_and_model(spec)
  batches = list(protein_iterator)
  if len(batches) != 1:
    msg = (
      f"expected exactly one combined batch covering all {n_states} states, got "
      f"{len(batches)} -- check spec.batch_size and the dataset iterator's actual "
      "batching behavior before trusting this function's output."
    )
    raise ValueError(msg)
  batched_ensemble = batches[0]
  if batched_ensemble.coordinates.shape[0] != n_states:
    msg = (
      f"batched_ensemble covers {batched_ensemble.coordinates.shape[0]} structures, "
      f"expected {n_states} (len(spec.inputs)) -- the dataset iterator split/dropped "
      "inputs unexpectedly."
    )
    raise ValueError(msg)

  seq_len = batched_ensemble.coordinates.shape[1]
  # _prepare_fixed_controls returns a (num_states, L) array -- shaped for
  # kernel_dispatch.py's per-structure_idx SLICING (each independent call gets its own
  # 1D (L,) row). build_inference_bundle's fixed_mask/fixed_tokens are design-level, not
  # per-state (its own default is 1D: jnp.zeros(seq_len)), and every row here is already
  # an identical broadcast of the same spec-level fixed positions (confirmed in
  # _prepare_fixed_controls's own body) -- take row 0 rather than passing the full 2D
  # array, which silently produces a wrong-shaped ConditioningBundle.fixed_mask/tokens
  # further downstream (confirmed 2026-07-13: this exact mistake broke AutoregressiveDecode's
  # wave-scan with a `Cannot broadcast to shape with fewer dimensions` error at do_sample's
  # seq_oh_stack construction -- not an aminx bug, a caller contract violation).
  #
  # The row-identity assertion below assumes fixed positions are the same across all states
  # for this call -- true for every real caller checked so far (tev_design's necklace P2
  # manifest builder never sets fixed_positions/fixed_mask at all; the prereg locks junction
  # placement identical across strata by design, see scripts/analysis/necklace_junction_placement.py).
  # _broadcast_per_structure/_prepare_fixed_controls DO support genuinely different fixed
  # positions per structure in general (that's their whole purpose for kernel_dispatch.py's
  # independent-structures case) -- a future caller with real per-state heterogeneous fixed
  # positions would hit the ValueError below, correctly, rather than this function silently
  # picking an arbitrary one of several different real fixed-position sets.
  fixed_mask_per_structure, fixed_tokens_per_structure = _prepare_fixed_controls(
    spec, batched_ensemble=batched_ensemble,
  )
  if not bool(jnp.all(fixed_mask_per_structure == fixed_mask_per_structure[0])) or not bool(
    jnp.all(fixed_tokens_per_structure == fixed_tokens_per_structure[0]),
  ):
    msg = (
      "_prepare_fixed_controls returned different fixed_mask/fixed_tokens rows per "
      "structure -- this function assumes fixed positions are design-level (identical "
      "across all states), per _prepare_fixed_controls's own broadcast contract; a "
      "genuinely per-state fixed_mask is not supported by this function."
    )
    raise ValueError(msg)
  fixed_mask = fixed_mask_per_structure[0]
  fixed_tokens = fixed_tokens_per_structure[0]
  ligand_context = _prepare_ligand_context(
    spec,
    batched_ensemble=batched_ensemble,
    batch_size=n_states,
    seq_len=seq_len,
  )

  state_weights = (
    jnp.asarray(spec.state_weights, dtype=jnp.float32) if spec.state_weights is not None else None
  )
  state_position_map = (
    jnp.asarray(spec.state_position_map) if spec.state_position_map is not None else None
  )
  if (
    state_position_map is not None
    and state_position_map.ndim == 2
    and state_position_map.shape[0] != n_states
  ):
    msg = (
      f"spec.state_position_map declares {state_position_map.shape[0]} states but this bead "
      f"has {n_states} (len(spec.inputs)) -- recompute state_position_map for this exact "
      "input set before calling this function (this is exactly the cardinality mismatch "
      "_realign_states_to_reference now rejects at fusion time, per the corruption fix)."
    )
    raise ValueError(msg)
  tie_group_map = jnp.asarray(spec.tie_group_map) if spec.tie_group_map is not None else None

  temperature_val = spec.run_spec.sampling.temperature
  temperature = float(temperature_val[0]) if isinstance(temperature_val, (list, tuple)) else float(
    temperature_val,
  )
  noise_val = spec.run_spec.sampling.backbone_noise
  backbone_noise = float(noise_val[0]) if isinstance(noise_val, (list, tuple)) else float(noise_val)
  bias = (
    jnp.asarray(spec.run_spec.sampling.bias, dtype=jnp.float32)
    if spec.run_spec.sampling.bias is not None
    else None
  )

  bundle, config = build_inference_bundle(
    coords=batched_ensemble.coordinates,
    mask=batched_ensemble.mask,
    residue_index=batched_ensemble.residue_index,
    chain_index=batched_ensemble.chain_index,
    backbone_noise=backbone_noise,
    fixed_mask=fixed_mask,
    fixed_tokens=fixed_tokens,
    bias=bias,
    tie_group_map=tie_group_map,
    state_weights=state_weights,
    state_position_map=state_position_map,
    ligand_coords=ligand_context["Y"],
    ligand_atom_types=ligand_context["Y_t"],
    ligand_mask=ligand_context["Y_m"],
    atom_37=ligand_context["atom_37"],
    atom_37_mask=ligand_context["atom_37_mask"],
    chain_mask=ligand_context["chain_mask"],
    structure_mapping=batched_ensemble.mapping,
    temperature=temperature,
    mode="sample_ar",
    inference=True,
  )

  stage_set = make_stage_set(
    strategy=spec.multi_state_strategy,
    strategy_temperature=getattr(spec, "multi_state_temperature", 1.0) or 1.0,
    state_weights=state_weights,
  )

  return sample_states_fused(model, bundle, config, stage_set, prng_key, n_samples)
