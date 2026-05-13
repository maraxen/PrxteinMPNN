"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.model_inputs import (
  AutoregressiveInputs,
  BackboneGeometry,
  ConditioningFeatures,
  LogitTransformFn,
  SamplingInputs,
  SamplingStaticConfig,
)
from prxteinmpnn.bundles import LigandBundle, ProteinBundle, WaveColorBundle
from prxteinmpnn.protocols import SamplerFn
from prxteinmpnn.registry import (
  MULTISTATE_MODE_STATE_VMAP_EXACT,
  SAMPLERS,
  combine_strategy_to_index,
  multistate_mode_descriptor,
)
from prxteinmpnn.run.decode_registry import DEFAULT_DECODE_FN_UID, register_decode_fn
from prxteinmpnn.sampling.state_vmap_prep import multistate_stack_payload_from_loose_ar_host
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  InputBias,
  Logits,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)


_AMINO_ACID_VOCAB = 21


def make_static_config_from_spec(
  spec: Any,
  *,
  decode_fn: Any | None = None,
) -> SamplingStaticConfig:
  """Build a :class:`SamplingStaticConfig` from a ``SamplingSpecification``.

  ``decode_fn`` overrides ``spec.decode_fn`` when provided.
  Registers the fn and stores its UID; falls back to the default arithmetic-mean fn.
  """
  fn = decode_fn if decode_fn is not None else getattr(spec, "decode_fn", None)
  if fn is not None:
    uid = register_decode_fn(fn)
  else:
    uid = DEFAULT_DECODE_FN_UID
  temperature = spec.temperature
  if hasattr(temperature, "__iter__"):
    temperature = float(next(iter(temperature)))
  return SamplingStaticConfig(
    decode_fn_uid=uid,
    n_samples=int(spec.num_samples),
    temperature=float(temperature),
    multistate_mode="tied",
    max_group_size=getattr(spec, "max_group_size", 1),
  )


def make_sampling_inputs_from_spec(
  spec: Any,
  coords: jnp.ndarray,
  mask: jnp.ndarray,
  residue_index: jnp.ndarray,
  chain_index: jnp.ndarray,
  *,
  multistate_stack: Any | None = None,
  wave_group_ids: jnp.ndarray | None = None,
  wave_group_positions: jnp.ndarray | None = None,
  wave_group_valid: jnp.ndarray | None = None,
  wave_position_valid: jnp.ndarray | None = None,
  fixed_tokens: jnp.ndarray | None = None,
  bias: jnp.ndarray | None = None,
  ar_mask: jnp.ndarray | None = None,
) -> SamplingInputs:
  """Build a :class:`SamplingInputs` pytree from a ``SamplingSpecification`` and raw arrays.

  All sub-payloads are resolved to concrete arrays here — nothing optional enters JIT.
  When ``multistate_stack`` is None, a trivial single-state payload is constructed from
  the provided backbone arrays.
  When wave args are None, a trivial single-wave covering all positions is constructed.
  """
  L = int(coords.shape[0])

  backbone = BackboneGeometry(
    coords=jnp.asarray(coords, dtype=jnp.float32),
    mask=jnp.asarray(mask, dtype=jnp.float32),
    residue_index=jnp.asarray(residue_index, dtype=jnp.int32),
    chain_index=jnp.asarray(chain_index, dtype=jnp.int32),
  )

  if multistate_stack is None:
    from prxteinmpnn.sampling.state_vmap_prep import multistate_stack_payload_from_loose_ar_host  # noqa: PLC0415
    rows = jnp.arange(L, dtype=jnp.int32)[None]  # (1, L)
    multistate_stack = multistate_stack_payload_from_loose_ar_host(
      coords=coords[None],
      mask=mask[None],
      residue_index=residue_index[None],
      chain_index=chain_index[None],
      tie_group_map=jnp.arange(L, dtype=jnp.int32)[None],
      fixed_mask=jnp.zeros((1, L), dtype=jnp.float32),
      fixed_tokens=jnp.zeros((1, L), dtype=jnp.int32),
      state_flat_rows=rows,
      n_canonical=L,
    )

  if wave_group_ids is None:
    wave_parallel = WaveColorBundle(
      wave_group_ids=jnp.arange(L, dtype=jnp.int32)[None, :],
      wave_group_positions=jnp.arange(L, dtype=jnp.int32)[None, :, None],
      wave_group_valid=jnp.ones((1, L), dtype=bool),
      wave_position_valid=jnp.ones((1, L, 1), dtype=bool),
    )
  else:
    wave_parallel = WaveColorBundle(
      wave_group_ids=jnp.asarray(wave_group_ids, dtype=jnp.int32),
      wave_group_positions=jnp.asarray(wave_group_positions, dtype=jnp.int32),
      wave_group_valid=jnp.asarray(wave_group_valid, dtype=bool),
      wave_position_valid=jnp.asarray(wave_position_valid, dtype=bool),
    )

  _fixed_tokens = (
    jnp.asarray(fixed_tokens, dtype=jnp.int32)
    if fixed_tokens is not None
    else jnp.zeros((L,), dtype=jnp.int32)
  )
  _bias = (
    jnp.asarray(bias, dtype=jnp.float32)
    if bias is not None
    else jnp.zeros((L, _AMINO_ACID_VOCAB), dtype=jnp.float32)
  )
  _ar_mask = (
    jnp.asarray(ar_mask, dtype=jnp.float32)
    if ar_mask is not None
    else jnp.eye(L, dtype=jnp.float32)
  )
  conditioning = ConditioningFeatures(
    fixed_tokens=_fixed_tokens,
    bias=_bias,
    ar_mask=_ar_mask,
  )

  return SamplingInputs(
    backbone=backbone,
    state_stack=multistate_stack,
    wave_parallel=wave_parallel,
    conditioning=conditioning,
  )


def make_sample_sequences(
  model: PrxteinMPNN | PrxteinLigandMPNN,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  use_concrete: bool = False,
  tau_start: float = 1.0,
  tau_end: float = 0.1,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
        Should accept (key, num_residues, tie_group_map, num_groups).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).

  Returns:
    A function that samples sequences from structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    >>> seq, logits, order = sample_fn(key, coords, mask, res_idx, chain_idx)
    >>>
    >>> # With tied positions
    >>> tie_map = jnp.array([0, 0, 1, 1, 2])  # Positions 0-1 tied, 2-3 tied
    >>> seq, logits, order = sample_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     tie_group_map=tie_map, num_groups=3
    ... )
    >>>
    >>> # For optimization
    >>> optimize_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    >>> seq, logits, order = optimize_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     iterations=100, learning_rate=0.01
    ... )

  """
  caps = model.capabilities
  supports_multi_state_temperature = caps.accepts_multi_state_temperature
  supports_state_weights = caps.accepts_state_weights
  supports_fixed_controls = caps.accepts_fixed_mask_and_tokens
  is_ligand_mpnn = caps.is_ligand_model

  if sampling_strategy == "straight_through":
    optimize_fn = make_optimize_sequence_fn(
      cast("PrxteinMPNN", model),
      decoding_order_fn,
      use_concrete=use_concrete,
      tau_start=tau_start,
      tau_end=tau_end,
    )

    @partial(
      jax.jit,
      static_argnames=(
        "num_groups",
        "multi_state_strategy",
        "multistate_mode",
        "max_group_size",
      ),
    )
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: StructureAtomicCoordinates,
      mask: AlphaCarbonMask,
      residue_index: ResidueIndex,
      chain_index: ChainIndex,
      bias: InputBias | None = None,
      fixed_positions: jnp.ndarray | None = None,
      fixed_mask: jnp.ndarray | None = None,
      fixed_tokens: jnp.ndarray | None = None,
      backbone_noise: BackboneNoise | None = None,
      iterations: Int | None = None,
      learning_rate: Float | None = None,
      temperature: Float | None = None,
      tie_group_map: jnp.ndarray | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal[
        "arithmetic_mean",
        "geometric_mean",
        "product",
      ] = "arithmetic_mean",
      multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
      structure_mapping: jax.Array | None = None,
      multi_state_temperature: Float = 1.0,
      max_group_size: int = 16,
      **kwargs: Any,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Optimize a sequence using straight-through estimation.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        bias: Not used in straight_through mode.
        fixed_positions: Not implemented yet.
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Number of optimization steps (default: 100).
        learning_rate: Learning rate for optimization (default: 0.01).
        temperature: Temperature for STE sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        multi_state_strategy: Unused in straight_through mode (kept for API compatibility).
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.
        multi_state_temperature: Unused in straight_through mode
          (kept for API compatibility).
        **kwargs: Additional arguments for LigandMPNN (Y, Y_t, Y_m) or weighting.

      Returns:
        Tuple of (optimized sequence, final logits, decoding order).

      """
      del bias, fixed_positions

      if iterations is None:
        iterations = jnp.array(100, dtype=jnp.int32)
      if learning_rate is None:
        learning_rate = jnp.array(0.01, dtype=jnp.float32)
      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      decoding_order, prng_key = decoding_order_fn(
        prng_key,
        structure_coordinates.shape[0],
        tie_group_map,
        num_groups,
      )

      optimized_sequence, final_logits, _ = optimize_fn(
        prng_key,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        iterations,
        learning_rate,
        temperature,
        backbone_noise,
        tie_group_map,
        num_groups,
        structure_mapping,
        multi_state_strategy=multi_state_strategy,
        multistate_mode=multistate_mode,
        multi_state_temperature=multi_state_temperature,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
        **kwargs,
      )

      return optimized_sequence, final_logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  if sampling_strategy == "temperature":

    @partial(
      jax.jit,
      static_argnames=("config", "batch_fn"),
    )
    def _sample_sequences_jitted(
      prng_key: PRNGKeyArray,
      inputs: SamplingInputs,
      config: SamplingStaticConfig,
      *,
      batch_fn: LogitTransformFn | None = None,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """JIT core for temperature sampling consuming unified ModelInputs."""
      # Extract backbone geometry
      coords = inputs.backbone.coords
      mask = inputs.backbone.mask
      residue_index = inputs.backbone.residue_index
      chain_index = inputs.backbone.chain_index

      # Extract conditioning features
      bias = inputs.conditioning.bias
      fixed_tokens = inputs.conditioning.fixed_tokens
      ar_mask = inputs.conditioning.ar_mask

      # Extract multistate info
      multistate_stack = inputs.state_stack
      multistate_mode = config.multistate_mode

      # Autoregressive specifics
      if isinstance(inputs, AutoregressiveInputs):
        wave_parallel = inputs.wave_parallel
        decoding_order = wave_parallel.decoding_order
      else:
        # Trivial decoding order for non-AR paths if they ever hit this
        decoding_order = jnp.arange(coords.shape[0])

      ms_route = multistate_mode_descriptor(multistate_mode)

      # Extract ligand context if present
      ligand_stack = inputs.ligand

      # State vmap exact path (Phase 2/3 unified)
      if ms_route.uses_stacked_exact_sample_wave:
        # For now, let's just delegate to the model's unified __call__
        # but the sampling logic needs to be aware of the wave-parallel scan.
        pass

      # Call model via unified __call__ interface
      # Note: In sampling, we usually use the internal _call_autoregressive
      # but we want to move towards model(key, inputs)
      return model(prng_key, inputs, config=config, batch_fn=batch_fn)

    def sample_sequences(
      prng_key: PRNGKeyArray,
      inputs: SamplingInputs,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Unified SamplerFn implementation."""
      # The config is captured from the factory scope or passed via inputs
      # (Ideally we'd have a way to carry static config in the Pytree but JAX
      # needs it in static_argnames).
      # For now, we assume the factory was called with a specific config.
      return _sample_sequences_jitted(
        prng_key,
        inputs,
        config=SamplingStaticConfig(
          decode_fn_uid=DEFAULT_DECODE_FN_UID, # Default
          n_samples=1, # Default
          temperature=1.0, # Default
        ),
      )

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)


SAMPLERS.register("make_sample_sequences")(make_sample_sequences)
