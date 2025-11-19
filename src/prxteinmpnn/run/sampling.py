"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.run.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.decoding_order import random_decoding_order

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from collections.abc import Callable

  from grain.python import IterDataset
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.data_structures import Protein
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    BackboneCoordinates,
    BackboneNoise,
    ChainIndex,
    DecodingOrder,
    Logits,
    ProteinSequence,
    ResidueIndex,
  )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  sampler_fn: Callable,
) -> tuple[ProteinSequence, Logits]:
  """Sample sequences for a batched ensemble of proteins."""
  keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

  tie_group_map = None
  num_groups = None

  if spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
    tie_group_map = resolve_tie_groups(spec, batched_ensemble)
    num_groups = int(jnp.max(tie_group_map)) + 1

  noise_array = (
    jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
    if spec.backbone_noise is not None
    else jnp.zeros(1)
  )

  sample_fn_with_params = partial(
    sampler_fn,
    _k_neighbors=48,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    fixed_positions=(
      jnp.asarray(spec.fixed_positions, dtype=jnp.int32)
      if spec.fixed_positions is not None
      else None
    ),
    _iterations=spec.iterations,
    _learning_rate=spec.learning_rate,
    temperature=spec.temperature,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
    multi_state_strategy=spec.multi_state_strategy,
    multi_state_alpha=spec.multi_state_alpha,
  )

  def sample_single_noise(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise: BackboneNoise,
    structure_mapping: jnp.ndarray | None = None,
    _sampler: partial = sample_fn_with_params,  # Bind to avoid closure issues
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample one sequence for one structure at one noise level."""
    return _sampler(
      key,
      coords,
      mask,
      residue_ix,
      chain_ix,
      backbone_noise=noise,
      structure_mapping=structure_mapping,
    )

  def mapped_fn_noise(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise_arr: BackboneNoise,
    structure_mapping: jnp.ndarray | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Compute samples across all noise levels for a single structure/sample."""
    return jax.lax.map(
      partial(
        sample_single_noise,
        key,
        coords,
        mask,
        residue_ix,
        chain_ix,
        structure_mapping=structure_mapping,
      ),
      noise_arr,
      batch_size=spec.noise_batch_size,
    )

  def internal_sample(
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    keys_arr: PRNGKeyArray,
    noise_arr: BackboneNoise,
    structure_mapping: jnp.ndarray | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample mapping over keys and noise."""
    noise_map_fn = partial(
      mapped_fn_noise,
      coords=coords,
      mask=mask,
      residue_ix=residue_ix,
      chain_ix=chain_ix,
      noise_arr=noise_arr,
      structure_mapping=structure_mapping,
    )

    return jax.lax.map(
      noise_map_fn,
      keys_arr,
      batch_size=spec.samples_batch_size,
    )

  vmap_structures = jax.vmap(
    internal_sample,
    in_axes=(0, 0, 0, 0, None, None, 0 if batched_ensemble.mapping is not None else None),
  )

  sampled_sequences, sampled_logits, _ = vmap_structures(
    batched_ensemble.coordinates,
    batched_ensemble.mask,
    batched_ensemble.residue_index,
    batched_ensemble.chain_index,
    keys,
    noise_array,
    batched_ensemble.mapping,
  )
  return sampled_sequences, sampled_logits


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Sample new sequences for the given input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then samples new sequences for each structure.

  Args:
      spec: An optional SamplingSpecification object. If None, a default will be created using
      kwargs, options are provided as keyword arguments. The following options can be set:
        inputs: A single or sequence of inputs (files, PDB IDs, etc.).
        chain_id: Specific chain(s) to parse from the structure.
        model: The model number to load. If None, all models are loaded.
        altloc: The alternate location identifier to use.
        model_version: The model version to use.
        model_weights: The model weights to use.
        foldcomp_database: The FoldComp database to use for FoldComp IDs.
        random_seed: The random number generator key.
        backbone_noise: The amount of noise to add to the backbone.
        num_samples: The number of sequences to sample per structure/noise level.
        sampling_strategy: The sampling strategy to use.
        temperature: The sampling temperature.
        bias: An optional array to bias the logits.
        fixed_positions: An optional array of residue indices to keep fixed.
        iterations: Number of optimization iterations for "straight_through" sampling.
        learning_rate: Learning rate for "straight_through" sampling.
        batch_size: The number of structures to process in a single batch.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing sampled sequences, logits, and metadata.

  """
  if spec is None:
    spec = SamplingSpecification(**kwargs)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  if spec.average_node_features:
    if spec.output_h5_path:
      return _sample_streaming_averaged(spec, protein_iterator, model)

    _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)
    all_sequences, all_logits = [], []

    for batched_ensemble in protein_iterator:
      sampled_sequences, logits = _sample_batch_averaged(
        spec,
        batched_ensemble,
        model,
        sample_fn,
        decode_fn,
      )
      all_sequences.append(sampled_sequences)
      all_logits.append(logits)

    return {
      "sequences": jnp.concatenate(all_sequences, axis=0),
      "logits": jnp.concatenate(all_logits, axis=0),
      "metadata": {
        "specification": spec,
      },
    }

  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  if spec.output_h5_path:
    return _sample_streaming(spec, protein_iterator, sampler_fn)

  all_sequences, all_logits = [], []

  for batched_ensemble in protein_iterator:
    sampled_sequences, logits = _sample_batch(
      spec,
      batched_ensemble,
      sampler_fn,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)

  max_len = max(arr.shape[-1] for arr in all_sequences)

  def pad_to_max(arr: jax.Array, target_len: int, axis: int = -1, pad_value: int = 0) -> jax.Array:
    """Pad the specified dimension of a JAX array to target_len."""
    diff = target_len - arr.shape[axis]
    if diff == 0:
      return arr
    padding_config = [(0, 0)] * arr.ndim
    # Handle negative axis
    axis = axis % arr.ndim
    padding_config[axis] = (0, diff)
    return jnp.pad(arr, padding_config, constant_values=pad_value)

  all_sequences_padded = [pad_to_max(seq, max_len, axis=-1, pad_value=0) for seq in all_sequences]

  all_logits_padded = [pad_to_max(logits, max_len, axis=-2, pad_value=0) for logits in all_logits]

  all_masks = [
    pad_to_max(jnp.ones(seq.shape, dtype=jnp.int32), max_len, axis=-1, pad_value=0)
    for seq in all_sequences
  ]

  return {
    "sequences": jnp.concatenate(all_sequences_padded, axis=0),
    "logits": jnp.concatenate(all_logits_padded, axis=0),
    "mask": jnp.concatenate(all_masks, axis=0),  # It's good practice to return this
    "metadata": {
      "specification": spec,
    },
  }


def _sample_streaming(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  sampler_fn: Callable,
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample new sequences and stream results to an HDF5 file."""
  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits = _sample_batch(
        spec,
        batched_ensemble,
        sampler_fn,
      )
      for i in range(sampled_sequences.shape[0]):
        grp = f.create_group(f"structure_{structure_idx}")
        grp.create_dataset("sequences", data=sampled_sequences[i], dtype="i4")
        grp.create_dataset("logits", data=sampled_logits[i], dtype="f4")
        # Store metadata about the structure
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[1]
        grp.attrs["num_noise_levels"] = sampled_sequences.shape[2]
        grp.attrs["sequence_length"] = sampled_sequences.shape[3]
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
    },
  }


def _sample_batch_averaged(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: PrxteinMPNN,
  sample_fn: Callable,  # noqa: ARG001
  decode_fn: Callable,  # noqa: ARG001
) -> tuple[ProteinSequence, Logits]:
  """Sample sequences for a batched ensemble of proteins using averaged encodings."""
  keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

  tie_group_map = None
  num_groups = None

  if spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
    tie_group_map = resolve_tie_groups(spec, batched_ensemble)
    num_groups = int(jnp.max(tie_group_map)) + 1

  averaged_encodings = get_averaged_encodings(
    batched_ensemble,
    model,
    spec.backbone_noise,
    spec.noise_batch_size,
    spec.random_seed,
    spec.average_encoding_mode,
  )

  # Create a custom decode wrapper that averages logits over structural features
  def decode_wrapper(base_decode_fn: Callable) -> Callable:
    def wrapped(
      encoded_features: tuple, sequence: ProteinSequence, ar_mask_in: jnp.ndarray,
    ) -> Logits:
      avg_node, avg_edge, neighbors, mask, ar_mask_struct = encoded_features

      # Flatten batch dimensions
      neighbors_flat = neighbors.reshape((-1, neighbors.shape[-2], neighbors.shape[-1]))
      mask_flat = mask.reshape((-1, mask.shape[-1]))
      ar_mask_struct_flat = ar_mask_struct.reshape(
          (-1, ar_mask_struct.shape[-2], ar_mask_struct.shape[-1]),
      )

      def decode_single(
          n_idx: jnp.ndarray, m: jnp.ndarray, ar_m: jnp.ndarray,
      ) -> Logits:
        # Reconstruct tuple for base function
        # Note: ar_mask_in is the autoregressive mask for sampling (L, L)
        # ar_m is the structural mask (L, L) or similar.
        # decode_logits_fn takes (encoded, seq, ar_mask).
        # We pass the structural ar_mask as part of encoded features if needed?
        # Wait, base_decode_fn signature is (encoded_features, sequence, ar_mask).
        # encoded_features expected by base_decode_fn is
        # (node, edge, neighbors, mask, ar_mask_struct).
        return base_decode_fn(
            (avg_node, avg_edge, n_idx, m, ar_m),
            sequence,
            ar_mask_in,
        )

      logits_batch = jax.vmap(decode_single)(
          neighbors_flat, mask_flat, ar_mask_struct_flat,
      )
      return jnp.mean(logits_batch, axis=0)

    return wrapped

  # Create a new sample_fn with the wrapper
  _, sample_fn_wrapped, decode_fn_wrapped = make_encoding_sampling_split_fn(
      model,
      decode_fn_wrapper=decode_wrapper,
  )

  sample_fn_with_params = partial(
    sample_fn_wrapped,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    temperature=spec.temperature,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
  )

  def sample_single_sequence(
    key: PRNGKeyArray,
    decoding_order_key: PRNGKeyArray,
    encoded_feat: tuple,
    _sampler: partial = sample_fn_with_params,
  ) -> ProteinSequence:
    """Sample one sequence from averaged features."""
    # encoded_feat contains (avg_node, avg_edge, neighbors, ...).
    # neighbors has shape (..., L, K).
    # We need sequence length.
    seq_len = encoded_feat[0].shape[0]
    decoding_order, _ = random_decoding_order(
      decoding_order_key,
      seq_len,
      tie_group_map,
      num_groups,
    )
    return _sampler(key, encoded_feat, decoding_order)

  def internal_sample_averaged(
    encoded_feat: tuple,
    keys_arr: PRNGKeyArray,
  ) -> ProteinSequence:
    """Sample mapping over keys for averaged features."""
    decoding_order_keys = jax.random.split(jax.random.key(spec.random_seed + 1), spec.num_samples)

    vmap_sample_fn = jax.vmap(
      partial(sample_single_sequence, encoded_feat=encoded_feat),
      in_axes=(0, 0),
      out_axes=0,
    )
    return vmap_sample_fn(keys_arr, decoding_order_keys)

  if spec.average_encoding_mode == "inputs_and_noise":
    sampled_sequences = internal_sample_averaged(
      averaged_encodings,
      keys,
    )
    sampled_sequences = jnp.expand_dims(sampled_sequences, axis=0)
  else:
    # We need to map over the "outer" batch dimension (the one we are NOT averaging over).
    # If mode="inputs" (avg over axis 0), we keep axis 1 (noise).
    # averaged_encodings: avg_node (M, ...), neighbors (N, M, ...).
    # We want to map over M.
    # For neighbors, we map over axis 1.
    # If mode="noise_levels" (avg over axis 1), we keep axis 0 (inputs).
    # averaged_encodings: avg_node (N, ...), neighbors (N, M, ...).
    # We map over N (axis 0).

    # Wait, get_averaged_encodings returns tuple.
    # (avg_node, avg_edge, neighbors, mask, ar_mask).
    # avg_node has shape (Batch, ...).
    # neighbors has shape (N, M, ...).

    # We need to construct in_axes for the tuple.
    # tuple structure: (node, edge, neighbors, mask, ar_mask).
    # node/edge: always axis 0 (because they are averaged to (Batch, ...)).
    # neighbors/mask/ar_mask: axis 1 if "inputs", axis 0 if "noise_levels".

    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    vmap_sample_structures = jax.vmap(
      internal_sample_averaged,
      in_axes=((0, 0, struct_axis, struct_axis, struct_axis), None),
    )
    sampled_sequences = vmap_sample_structures(
      averaged_encodings,
      keys,
    )

  # Get logits for the sampled sequences using the SAME wrapped decode function
  seq_len = sampled_sequences.shape[-1]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  if spec.average_encoding_mode == "inputs_and_noise":
    def get_logits_local_both(seq: ProteinSequence) -> Logits:
      return decode_fn_wrapped(averaged_encodings, seq, ar_mask)

    vmap_logits = jax.vmap(get_logits_local_both)
    logits = vmap_logits(sampled_sequences[0])
    logits = jnp.expand_dims(logits, axis=0)
  else:
    def get_logits_local(seq: ProteinSequence, enc: tuple) -> Logits:
      return decode_fn_wrapped(enc, seq, ar_mask)

    # vmap over samples (axis 0 of seq)
    # vmap over outer batch (axis 0 of seqs, axis 0/1 of enc)
    # We want to map `get_logits_local` over Batch.
    # Inside, we map over NumSamples.

    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    vmap_logits = jax.vmap(
        jax.vmap(get_logits_local, in_axes=(0, None)),
        in_axes=(0, (0, 0, struct_axis, struct_axis, struct_axis)),
    )
    logits = vmap_logits(sampled_sequences, averaged_encodings)

  # Flatten batch dimensions into samples
  # We want (1, Batch*Samples, L)

  sampled_sequences = sampled_sequences.reshape((1, -1, seq_len))
  logits = logits.reshape((1, -1, seq_len, 21))

  return sampled_sequences, logits


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: PrxteinMPNN,
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample new sequences with averaged encodings and stream results to an HDF5 file."""
  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits = _sample_batch_averaged(
        spec,
        batched_ensemble,
        model,
        sample_fn,
        decode_fn,
      )
      for i in range(sampled_sequences.shape[0]):
        grp = f.create_group(f"structure_{structure_idx}")
        grp.create_dataset("sequences", data=sampled_sequences[i], dtype="i4")
        grp.create_dataset("logits", data=sampled_logits[i], dtype="f4")
        # Store metadata about the structure
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[1]
        grp.attrs["num_noise_levels"] = 1  # Averaged, so effectively 1 noise level
        grp.attrs["sequence_length"] = sampled_sequences.shape[2]
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
    },
  }
