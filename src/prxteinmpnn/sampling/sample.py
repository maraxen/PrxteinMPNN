"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
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
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
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
  if sampling_strategy == "straight_through":
    optimize_fn = make_optimize_sequence_fn(model, decoding_order_fn)

    @partial(jax.jit, static_argnames=("_k_neighbors", "num_groups"))
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: StructureAtomicCoordinates,
      mask: AlphaCarbonMask,
      residue_index: ResidueIndex,
      chain_index: ChainIndex,
      _k_neighbors: int = 48,
      bias: InputBias | None = None,
      fixed_positions: jnp.ndarray | None = None,
      backbone_noise: BackboneNoise | None = None,
      iterations: Int | None = None,
      learning_rate: Float | None = None,
      temperature: Float | None = None,
      tie_group_map: jnp.ndarray | None = None,
      num_groups: int | None = None,
      structure_mapping: jax.Array | None = None,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Optimize a sequence using straight-through estimation.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        _k_neighbors: Deprecated (kept for API compatibility).
        bias: Not used in straight_through mode.
        fixed_positions: Not implemented yet.
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Number of optimization steps (default: 100).
        learning_rate: Learning rate for optimization (default: 0.01).
        temperature: Temperature for STE sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.

      Returns:
        Tuple of (optimized sequence, final logits, decoding order).

      """
      del bias, fixed_positions, _k_neighbors  # Not used in optimization

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
      )

      return optimized_sequence, final_logits, decoding_order

    return sample_sequences

  if sampling_strategy == "temperature":

    @partial(jax.jit, static_argnames=("_k_neighbors", "num_groups", "multi_state_strategy"))
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: StructureAtomicCoordinates,
      mask: AlphaCarbonMask,
      residue_index: ResidueIndex,
      chain_index: ChainIndex,
      _k_neighbors: int = 48,
      bias: InputBias | None = None,
      fixed_positions: jnp.ndarray | None = None,
      backbone_noise: BackboneNoise | None = None,
      _iterations: Int | None = None,
      _learning_rate: Float | None = None,
      temperature: Float | None = None,
      tie_group_map: jnp.ndarray | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
      multi_state_alpha: float = 0.5,
      structure_mapping: jax.Array | None = None,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Sample a sequence from a structure using the ProteinMPNN model.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        _k_neighbors: Deprecated, model handles internally (kept for API compatibility).
        bias: Optional bias to add to logits (N, 21).
        fixed_positions: Optional mask for positions to keep fixed (not implemented yet).
        backbone_noise: Optional noise for backbone coordinates.
        _iterations: Unused in temperature mode (for API compatibility).
        _learning_rate: Unused in temperature mode (for API compatibility).
        temperature: Temperature for sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        multi_state_strategy: Strategy for combining logits across tied positions
          ("mean", "min", "product", "max_min").
        multi_state_alpha: Weight for min component when strategy="max_min" (0-1).
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.


      Returns:
        Tuple of (sampled sequence, logits, decoding order).

      Example:
        >>> seq, logits, order = sample_sequences(
        ...     key, coords, mask, res_idx, chain_idx, temperature=0.1
        ... )

      """
      del fixed_positions  # Not yet implemented

      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      decoding_order, prng_key = decoding_order_fn(
        prng_key,
        structure_coordinates.shape[0],
        tie_group_map,
        num_groups,
      )
      autoregressive_mask = generate_ar_mask(decoding_order, None, tie_group_map, num_groups)

      sampled_sequence, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach="autoregressive",
        prng_key=prng_key,
        ar_mask=autoregressive_mask,
        temperature=temperature,
        bias=bias,
        backbone_noise=backbone_noise,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
        multi_state_alpha=multi_state_alpha,
        structure_mapping=structure_mapping,
      )

      one_hot_ndim = 2
      if sampled_sequence.ndim == one_hot_ndim:
        sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)

      return sampled_sequence, logits, decoding_order

    return sample_sequences

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)


def make_encoding_sampling_split_fn(
  model: PrxteinMPNN,
) -> tuple[Callable, Callable]:
    """Create separate encoding and sampling functions.

    This splits the model into two parts:
    1. Encoding: Structure -> Encoder features
    2. Sampling: (Encoder features, Decoding Order) -> Sequence

    This is useful for applications where the encoding can be cached and reused.

    Args:
        model: A PrxteinMPNN Equinox model instance.

    Returns:
        Tuple of (encode_fn, sample_fn).
    """

    def encode_fn(
        prng_key: PRNGKeyArray,
        structure_coordinates: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_index: ResidueIndex,
        chain_index: ChainIndex,
        backbone_noise: BackboneNoise | None = None,
        structure_mapping: jax.Array | None = None,
    ) -> tuple:
        """Encode structure to get encoder features."""
        edge_features, neighbor_indices, initial_node_features, _ = model.features(
            prng_key,
            structure_coordinates,
            mask,
            residue_index,
            chain_index,
            backbone_noise,
            structure_mapping,
        )

        node_features, processed_edge_features = model.encoder(
            edge_features,
            neighbor_indices,
            mask,
            initial_node_features
        )

        return (node_features, processed_edge_features, neighbor_indices, mask)

    def sample_fn(
        prng_key: PRNGKeyArray,
        encoded_features: tuple,
        decoding_order: DecodingOrder,
        temperature: Float | None = None,
        bias: InputBias | None = None,
        tie_group_map: jnp.ndarray | None = None,
        num_groups: int | None = None,
        multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
        multi_state_alpha: float = 0.5,
    ) -> ProteinSequence:
        """Sample a sequence from encoded features."""
        node_features, processed_edge_features, neighbor_indices, mask = encoded_features

        if temperature is None:
            temperature = jnp.array(1.0, dtype=jnp.float32)

        ar_mask = generate_ar_mask(decoding_order, None, tie_group_map, num_groups)

        strategy_map = {"mean": 0, "min": 1, "product": 2, "max_min": 3}
        multi_state_strategy_idx = jnp.array(
            strategy_map[multi_state_strategy],
            dtype=jnp.int32,
        )

        sampled_sequence, _ = model._call_autoregressive(
            node_features,
            processed_edge_features,
            neighbor_indices,
            mask,
            ar_mask,
            jnp.zeros((mask.shape[0], model.w_s_embed.num_embeddings)),
            prng_key,
            temperature,
            bias if bias is not None else jnp.zeros((mask.shape[0], 21)),
            tie_group_map,
            multi_state_strategy_idx,
            multi_state_alpha,
        )

        one_hot_ndim = 2
        if sampled_sequence.ndim == one_hot_ndim:
            sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)

        return sampled_sequence

    return encode_fn, sample_fn