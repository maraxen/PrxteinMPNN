"""Factory for creating sequence sampling functions for PrxteinMPNN."""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model import PrxteinLigandMPNN, PrxteinMPNN
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

SamplerFn = Callable[..., tuple[ProteinSequence, Logits, DecodingOrder]]


def make_sample_sequences(
  model: PrxteinMPNN | PrxteinLigandMPNN,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
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
  model_params = inspect.signature(model.__call__).parameters
  supports_multi_state_temperature = "multi_state_temperature" in model_params
  supports_state_weights = "state_weights" in model_params
  supports_fixed_controls = "fixed_mask" in model_params and "fixed_tokens" in model_params
  is_ligand_mpnn = "Y" in model_params

  if sampling_strategy == "straight_through":
    optimize_fn = make_optimize_sequence_fn(model, decoding_order_fn)

    @partial(
      jax.jit,
      static_argnames=(
        "num_groups",
        "multi_state_strategy",
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
      structure_mapping: jax.Array | None = None,
      multi_state_temperature: Float = 1.0,
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
      del bias, fixed_positions, multi_state_strategy, multi_state_temperature
      if fixed_mask is not None or fixed_tokens is not None:
        msg = "fixed_mask/fixed_tokens are only supported with temperature sampling."
        raise ValueError(msg)

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

      # Note: STE optimize might need updates for LigandMPNN extra inputs
      # For now we'll pass structure_mapping which it supports.
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
        multi_state_temperature=multi_state_temperature,
        **kwargs,
      )

      return optimized_sequence, final_logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  if sampling_strategy == "temperature":

    @partial(
      jax.jit,
      static_argnames=(
        "num_groups",
        "multi_state_strategy",
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
      structure_mapping: jax.Array | None = None,
      multi_state_temperature: Float = 1.0,
      **kwargs: Any,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Sample a sequence from a structure using the ProteinMPNN model.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        bias: Optional bias to add to logits (N, 21).
        fixed_positions: Optional mask for positions to keep fixed (not implemented yet).
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Unused in temperature mode (for API compatibility).
        learning_rate: Unused in temperature mode (for API compatibility).
        temperature: Temperature for sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        multi_state_strategy: Strategy for combining logits across tied positions
          ("arithmetic_mean", "geometric_mean", "product").
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.
        multi_state_temperature: Temperature for geometric_mean multi-state combining.
        **kwargs: Additional arguments for LigandMPNN (Y, Y_t, Y_m) or weighting.


      Returns:
        Tuple of (sampled sequence, logits, decoding order).

      Example:
        >>> seq, logits, order = sample_sequences(
        ...     key, coords, mask, res_idx, chain_idx, temperature=0.1
        ... )

      """
      del iterations, learning_rate, fixed_positions
      if fixed_mask is None and fixed_positions is not None:
        fixed_mask = fixed_positions

      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      decoding_order, prng_key = decoding_order_fn(
        prng_key,
        structure_coordinates.shape[0],
        tie_group_map,
        num_groups,
      )
      autoregressive_mask = cast("Callable", generate_ar_mask)(
        decoding_order, None, tie_group_map, num_groups,
      )

      call_kwargs = {
        "decoding_approach": "autoregressive",
        "prng_key": prng_key,
        "ar_mask": autoregressive_mask,
        "temperature": temperature,
        "bias": bias,
        "backbone_noise": backbone_noise,
        "tie_group_map": tie_group_map,
        "multi_state_strategy": multi_state_strategy,
        "structure_mapping": structure_mapping,
      }

      if supports_multi_state_temperature:
        call_kwargs["multi_state_temperature"] = multi_state_temperature

      if supports_state_weights:
        call_kwargs["state_weights"] = kwargs.get("state_weights")
        call_kwargs["state_mapping"] = kwargs.get("state_mapping")

      if supports_fixed_controls:
        call_kwargs["fixed_mask"] = fixed_mask
        call_kwargs["fixed_tokens"] = fixed_tokens

      if is_ligand_mpnn:
        call_kwargs["Y"] = kwargs.get("Y")
        call_kwargs["Y_t"] = kwargs.get("Y_t")
        call_kwargs["Y_m"] = kwargs.get("Y_m")
        call_kwargs["xyz_37"] = kwargs.get("xyz_37")
        call_kwargs["xyz_37_m"] = kwargs.get("xyz_37_m")
        call_kwargs["chain_mask"] = kwargs.get("chain_mask")

      sampled_sequence, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        **call_kwargs,
      )

      one_hot_ndim = 2
      if sampled_sequence.ndim == one_hot_ndim:
        sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)

      return sampled_sequence, logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)
