"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.sampling.sample import make_encoding_sampling_split_fn, make_sample_sequences
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.decoding_order import random_decoding_order

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from collections.abc import Callable

  from grain.python import IterDataset
  from jaxtyping import PRNGKeyArray

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

    encode_fn, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)
    all_sequences, all_logits = [], []

    for batched_ensemble in protein_iterator:
      sampled_sequences, logits = _sample_batch_averaged(
        spec,
        batched_ensemble,
        encode_fn,
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

  return {
    "sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
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
  encode_fn: Callable,
  sample_fn: Callable,
  decode_fn: Callable,
) -> tuple[ProteinSequence, Logits]:
  """Sample sequences for a batched ensemble of proteins using averaged encodings."""
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

  def encode_single_noise(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise: BackboneNoise,
    structure_mapping: jnp.ndarray | None = None,
    _encoder: Callable = encode_fn,  # Bind to avoid closure issues
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
    structure_mapping: jnp.ndarray | None = None,
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
        structure_mapping=structure_mapping,
      ),
      noise_arr,
      batch_size=spec.noise_batch_size,
    )

  # Vmap over structures to get encodings for all structures and noise levels
  vmap_encode_structures = jax.vmap(
    mapped_encode_noise,
    in_axes=(None, 0, 0, 0, 0, None, 0 if batched_ensemble.mapping is not None else None),
  )

  # encoded_features_per_noise will have shape (num_structures, num_noise_levels, ...)
  encoded_features_per_noise = vmap_encode_structures(
    jax.random.key(spec.random_seed),  # Use a consistent key for encoding
    batched_ensemble.coordinates,
    batched_ensemble.mask,
    batched_ensemble.residue_index,
    batched_ensemble.chain_index,
    noise_array,
    batched_ensemble.mapping,
  )

  # Unpack and average features
  node_features, processed_edge_features, neighbor_indices, mask, ar_mask = encoded_features_per_noise
  
  if spec.average_encoding_mode == "inputs":
    averaging_axis = 0
    # Take features from the first structure
    neighbor_indices = neighbor_indices[0, :, ...]
    mask = mask[0, :, ...]
    ar_mask = ar_mask[0, :, ...]
  elif spec.average_encoding_mode == "noise_levels":
    averaging_axis = 1
    # Take features from the first noise level
    neighbor_indices = neighbor_indices[:, 0, ...]
    mask = mask[:, 0, ...]
    ar_mask = ar_mask[:, 0, ...]
  else:  # "inputs_and_noise"
    averaging_axis = (0, 1)
    # Take features from the first structure and noise level
    neighbor_indices = neighbor_indices[0, 0, ...]
    mask = mask[0, 0, ...]
    ar_mask = ar_mask[0, 0, ...]

  avg_node_features = jnp.mean(node_features, axis=averaging_axis)
  avg_processed_edge_features = jnp.mean(processed_edge_features, axis=averaging_axis)
  
  averaged_encodings = (avg_node_features, avg_processed_edge_features, neighbor_indices, mask, ar_mask)

  # Now sample from the averaged encodings
  sample_fn_with_params = partial(
    sample_fn,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    temperature=spec.temperature,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
  )

  def sample_single_sequence(
    key: PRNGKeyArray,
    decoding_order_key: PRNGKeyArray,
    encoded_feat: tuple,
    _sampler: partial = sample_fn_with_params,  # Bind to avoid closure issues
  ) -> ProteinSequence:
    """Sample one sequence from averaged features."""
    decoding_order, _ = random_decoding_order(
      decoding_order_key,
      encoded_feat[0].shape[0],  # Use sequence length from node_features
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
        out_axes=0
    )
    return vmap_sample_fn(keys_arr, decoding_order_keys)

  if spec.average_encoding_mode == "inputs_and_noise":
    sampled_sequences = internal_sample_averaged(
      averaged_encodings,
      keys,
    )
    sampled_sequences = jnp.expand_dims(sampled_sequences, axis=0)
  else:
    vmap_sample_structures = jax.vmap(
      internal_sample_averaged,
      in_axes=(0, None),
    )
    sampled_sequences = vmap_sample_structures(
      averaged_encodings,
      keys,
    )

  # Get logits for the sampled sequences
  seq_len = sampled_sequences.shape[-1]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  if spec.average_encoding_mode == "inputs_and_noise":
    def get_logits(seq):
        return decode_fn(averaged_encodings, seq, ar_mask)
    
    vmap_logits = jax.vmap(get_logits)
    logits = vmap_logits(sampled_sequences[0])
    logits = jnp.expand_dims(logits, axis=0)
  else:
    def get_logits(seq, enc):
        return decode_fn(enc, seq, ar_mask)

    vmap_logits = jax.vmap(jax.vmap(get_logits, in_axes=(0, None)), in_axes=(0, 0))
    logits = vmap_logits(sampled_sequences, averaged_encodings)

  return sampled_sequences, logits


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: PrxteinMPNN,
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample new sequences with averaged encodings and stream results to an HDF5 file."""
  from prxteinmpnn.sampling.sample import make_encoding_sampling_split_fn

  encode_fn, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits = _sample_batch_averaged(
        spec,
        batched_ensemble,
        encode_fn,
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
