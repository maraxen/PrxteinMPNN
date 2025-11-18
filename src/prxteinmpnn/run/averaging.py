"""Contains the logic for averaging node features over multiple structures and/or noise levels."""
from collections.abc import Callable
from functools import partial
from typing import Literal, Tuple, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    BackboneCoordinates,
    BackboneNoise,
    ChainIndex,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
)
from prxteinmpnn.utils.decoding_order import DecodingOrder, DecodingOrderFn
from prxteinmpnn.utils.types import InputBias


def get_averaged_encodings(
    batched_ensemble: Protein,
    model: PrxteinMPNN,
    backbone_noise: Sequence[float] | float,
    noise_batch_size: int,
    random_seed: int,
    average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"],
) -> Tuple:
    """
    Compute averaged node and edge features from an ensemble of protein structures.

    This function encodes a batch of protein structures at multiple noise levels,
    then averages the resulting node and edge features based on the specified
    averaging mode.

    Args:
        batched_ensemble: A batch of protein structures.
        model: The PrxteinMPNN model.
        backbone_noise: The backbone noise levels to use.
        noise_batch_size: The batch size for noise levels.
        random_seed: The random seed to use.
        average_encoding_mode: The mode for averaging encodings.

    Returns:
        A tuple containing the averaged encodings and other necessary features for
        downstream tasks (sampling, scoring, etc.).
    """
    encode_fn, _, _ = make_encoding_sampling_split_fn(model)

    noise_array = (
        jnp.asarray(backbone_noise, dtype=jnp.float32)
        if backbone_noise is not None
        else jnp.zeros(1)
    )

    def encode_single_noise(
        key: PRNGKeyArray,
        coords: BackboneCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        noise: BackboneNoise,
        _encoder: Callable = encode_fn,
    ) -> tuple:
        """Encode one structure at one noise level."""
        return _encoder(
            key,
            coords,
            mask,
            residue_ix,
            chain_ix,
            backbone_noise=noise,
        )

    def mapped_encode_noise(
        key: PRNGKeyArray,
        coords: BackboneCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        noise_arr: BackboneNoise,
    ) -> tuple:
        """Compute encodings across all noise levels for a single structure."""
        return jax.lax.map(
            partial(
                encode_single_noise,
                key,
                coords,
                mask,
                residue_ix,
                chain_ix,
            ),
            noise_arr,
            batch_size=noise_batch_size,
        )

    vmap_encode_structures = jax.vmap(
        mapped_encode_noise,
        in_axes=(None, 0, 0, 0, 0, None),
    )

    encoded_features_per_noise = vmap_encode_structures(
        jax.random.key(random_seed),
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        noise_array,
    )

    (
        node_features,
        processed_edge_features,
        neighbor_indices,
        mask,
        ar_mask,
    ) = encoded_features_per_noise

    if average_encoding_mode == "inputs":
        averaging_axis = 0
    elif average_encoding_mode == "noise_levels":
        averaging_axis = 1
    else:  # "inputs_and_noise"
        averaging_axis = (0, 1)

    avg_node_features = jnp.mean(node_features, axis=averaging_axis)
    avg_processed_edge_features = jnp.mean(processed_edge_features, axis=averaging_axis)

    return (
        avg_node_features,
        avg_processed_edge_features,
        neighbor_indices,
        mask,
        ar_mask,
    )


def make_encoding_sampling_split_fn(
  model_parameters: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn | None = None,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  decode_fn_wrapper: Callable[[Callable], Callable] | None = None,
) -> tuple[Callable, Callable, Callable]:
  """Create separate encoding and sampling functions for averaged encodings.

  This splits the sampling process into two parts:
  1. Encoding: Structure -> Encoder features (can be averaged across noise levels)
  2. Sampling: (Encoder features, PRNGKey) -> Sequences

  This separation allows:
  - Averaging encoder features across multiple noise levels
  - Efficient reuse of encoder output for multiple samples
  - Lower memory usage when sampling many sequences

  Supports tied positions: when tie_group_map is provided, positions in the same
  group will be sampled together and receive identical amino acids.

  Args:
    model_parameters: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".
    decode_fn_wrapper: Optional wrapper for the decode function.

  Returns:
    Tuple of (encode_fn, sample_fn) where:
      - encode_fn: Computes encoder features from structure
      - sample_fn: Samples sequences from cached encoder features

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> encode_fn, sample_fn = make_encoding_sampling_split_fn(
    ...     model, sampling_strategy="temperature"
    ... )
    >>> # Encode once
    >>> encoding = encode_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     k_neighbors=48, backbone_noise=0.1
    ... )
    >>> # Sample multiple sequences using same encoding
    >>> seq1 = sample_fn(key1, encoding, order1, temperature=0.1)
    >>> seq2 = sample_fn(key2, encoding, order2, temperature=0.5)

  """
  del sampling_strategy, decoding_order_fn  # Currently only temperature sampling supported

  # Get the encoder/decoder split functions
  encode_logits_fn, decode_logits_fn = make_encoding_conditional_logits_split_fn(
    model_parameters,
  )

  if decode_fn_wrapper is not None:
    decode_logits_fn = decode_fn_wrapper(decode_logits_fn)

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def encode_fn(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple:
    """Encode structure to get encoder features.

    Args:
      prng_key: JAX random key for feature extraction.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      k_neighbors: Number of nearest neighbors (currently not used, for API compat).
      backbone_noise: Optional noise for backbone coordinates.

    Returns:
      Encoder features tuple that can be passed to sample_fn.

    """
    del k_neighbors

    return encode_logits_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
    )

  @partial(jax.jit, static_argnames=("num_groups",))
  def sample_fn(
    prng_key: PRNGKeyArray,
    encoded_features: tuple,
    decoding_order: DecodingOrder,
    bias: InputBias | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
  ) -> ProteinSequence:
    """Sample a sequence from cached encoder features.

    Args:
      prng_key: JAX random key for sampling.
      encoded_features: Encoder features from encode_fn.
      decoding_order: Decoding order for autoregressive sampling.
      bias: Optional bias to add to logits (N, 21).
      iterations: For STE optimization (not used in temperature sampling).
      learning_rate: For STE optimization (not used in temperature sampling).
      temperature: Temperature for sampling (default: 1.0).
      sampling_strategy: "temperature" or "straight_through" (currently only temperature).
      tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
      num_groups: Number of unique groups when using tied positions.

    Returns:
      Sampled sequence as integer array (N,).

    """
    del iterations, learning_rate, sampling_strategy  # Not used in current implementation

    if temperature is None:
      temperature = jnp.array(1.0, dtype=jnp.float32)

    autoregressive_mask = generate_ar_mask(decoding_order, tie_group_map)

    seq_length = autoregressive_mask.shape[0]
    _, prng_key = jax.random.split(prng_key)
    initial_seq = jax.random.randint(
      prng_key,
      shape=(seq_length,),
      minval=0,
      maxval=21,
      dtype=jnp.int8,
    )

    if tie_group_map is not None and num_groups is not None:

      def sample_group_step(
        group_idx: int,
        state: tuple[ProteinSequence, PRNGKeyArray],
      ) -> tuple[ProteinSequence, PRNGKeyArray]:
        """Sample one group of tied positions together."""
        sequence, key = state

        logits = decode_logits_fn(encoded_features, sequence, autoregressive_mask)
        group_mask = (tie_group_map == group_idx).astype(jnp.float32)[:, None]  # (N, 1)
        group_count = group_mask.sum(axis=0, keepdims=True)  # (1, 1)
        group_logit_sum = (logits * group_mask).sum(axis=0)  # (21,)
        avg_logits = group_logit_sum / (group_count.squeeze() + 1e-8)  # (21,)
        if bias is not None:
          group_bias = (bias * group_mask).sum(axis=0) / (group_count.squeeze() + 1e-8)
          avg_logits = avg_logits + group_bias
        scaled_logits = avg_logits / temperature
        key, subkey = jax.random.split(key)
        sampled_aa = jax.random.categorical(subkey, scaled_logits).astype(jnp.int8)
        group_member_mask = tie_group_map == group_idx  # (N,) boolean
        updated_seq = jnp.where(group_member_mask, sampled_aa, sequence)
        return updated_seq, key

      final_seq, _ = jax.lax.fori_loop(
        0,
        num_groups,
        sample_group_step,
        (initial_seq, prng_key),
      )

    else:

      def sample_step(
        i: int,
        state: tuple[ProteinSequence, PRNGKeyArray],
      ) -> tuple[ProteinSequence, PRNGKeyArray]:
        """Sample one position in the decoding order."""
        sequence, key = state
        logits = decode_logits_fn(encoded_features, sequence, autoregressive_mask)
        pos = decoding_order[i]
        if bias is not None:
          logits = logits.at[pos].add(bias[pos])
        scaled_logits = logits[pos] / temperature
        key, subkey = jax.random.split(key)
        sampled_aa = jax.random.categorical(subkey, scaled_logits).astype(jnp.int8)
        updated_seq = sequence.at[pos].set(sampled_aa)
        return updated_seq, key

      final_seq, _ = jax.lax.fori_loop(
        0,
        seq_length,
        sample_step,
        (initial_seq, prng_key),
      )

    return final_seq

  return encode_fn, sample_fn, decode_logits_fn
