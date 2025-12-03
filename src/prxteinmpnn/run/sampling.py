"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.run.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.safe_map import safe_map as _safe_map

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from grain.python import IterDataset
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.data_structures import Protein
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneCoordinates,
    ChainIndex,
    DecodingOrder,
    Logits,
    ProteinSequence,
    ResidueIndex,
  )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

RANK_WITH_TEMPERATURE = 4


def _sample_batch(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  sampler_fn: Callable,
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
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

  temperature_array = jnp.asarray(spec.temperature, dtype=jnp.float32)

  sample_fn_with_params = partial(
    sampler_fn,
    _k_neighbors=48,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    fixed_positions=(
      jnp.asarray(spec.fixed_positions, dtype=jnp.int32)
      if spec.fixed_positions is not None
      else None
    ),
    iterations=spec.iterations,
    learning_rate=spec.learning_rate,
    multi_state_strategy=spec.multi_state_strategy,
    num_groups=num_groups,
    md_config={
        "temperature": spec.md_temperature,
        "min_steps": spec.md_min_steps,
        "therm_steps": spec.md_therm_steps,
    } if spec.backbone_noise_mode == "md" else None,
  )

  def sample_single_config(
    key: PRNGKeyArray,
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    noise: float,
    temp: float,
    current_tie_map: jnp.ndarray | None,
    structure_mapping: jnp.ndarray | None = None,
    full_coords: jax.Array | None = None,
    md_p: dict[str, jax.Array] | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample one sequence for one structure configuration."""
    return sample_fn_with_params(
      key,
      coords,
      mask,
      residue_ix,
      chain_ix,
      backbone_noise=noise,
      temperature=temp,
      tie_group_map=current_tie_map,
      structure_mapping=structure_mapping,
      full_coordinates=full_coords,
      md_params=md_p,
    )

  def internal_sample(
    coords: BackboneCoordinates,
    mask: AlphaCarbonMask,
    residue_ix: ResidueIndex,
    chain_ix: ChainIndex,
    keys_arr: PRNGKeyArray,
    current_tie_map: jnp.ndarray | None,
    structure_mapping: jnp.ndarray | None = None,
    full_coords: jax.Array | None = None,
    md_p: dict[str, jax.Array] | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample mapping over keys (sequential) and noise/temp (vectorized)."""

    def map_over_noise_and_temp(
      k: PRNGKeyArray,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      def map_over_temp(n: float) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        return _safe_map(
          lambda t: sample_single_config(
            k,
            coords,
            mask,
            residue_ix,
            chain_ix,
            n,
            t,
            current_tie_map,
            structure_mapping,
            full_coords,
            md_p,
          ),
          temperature_array,
          batch_size=spec.temperature_batch_size,
        )

      return _safe_map(map_over_temp, noise_array, batch_size=spec.noise_batch_size)

    return _safe_map(
      map_over_noise_and_temp,
      keys_arr,
      batch_size=spec.samples_batch_size,
    )

  # Ensure tie_group_map and mapping have batch dimensions for vmap
  # tie_group_map comes from resolve_tie_groups with shape (n_residues,)
  # but vmap expects (batch_size, n_residues) when in_axes=0
  batch_size = batched_ensemble.coordinates.shape[0]
  
  tie_map_for_vmap = None
  if tie_group_map is not None:
    # Add batch dimension and broadcast: (n,) -> (1, n) -> (batch_size, n)
    tie_map_for_vmap = jnp.broadcast_to(
      jnp.atleast_2d(tie_group_map), 
      (batch_size, tie_group_map.shape[0])
    )
  
  mapping_for_vmap = batched_ensemble.mapping
  if batched_ensemble.mapping is not None and batched_ensemble.mapping.ndim == 1:
    # Add batch dimension and broadcast if needed: (n,) -> (1, n) -> (batch_size, n)
    mapping_for_vmap = jnp.broadcast_to(
      jnp.atleast_2d(batched_ensemble.mapping),
      (batch_size, batched_ensemble.mapping.shape[0])
    )
  
  tie_map_in_axis = 0 if tie_map_for_vmap is not None else None
  mapping_in_axis = 0 if mapping_for_vmap is not None else None

  vmap_structures = jax.vmap(
    internal_sample,
    in_axes=(0, 0, 0, 0, None, tie_map_in_axis, mapping_in_axis, 0, 0),
  )

  md_params = None
  if spec.backbone_noise_mode == "md":
      md_params = {
          "bonds": batched_ensemble.md_bonds,
          "bond_params": batched_ensemble.md_bond_params,
          "angles": batched_ensemble.md_angles,
          "angle_params": batched_ensemble.md_angle_params,
          "backbone_indices": batched_ensemble.md_backbone_indices,
          "exclusion_mask": batched_ensemble.md_exclusion_mask,
          "charges": batched_ensemble.charges,
          "sigmas": batched_ensemble.sigmas,
          "epsilons": batched_ensemble.epsilons,
      }

  sampled_sequences, sampled_logits, _ = vmap_structures(
    batched_ensemble.coordinates,
    batched_ensemble.mask,
    batched_ensemble.residue_index,
    batched_ensemble.chain_index,
    keys,
    tie_map_for_vmap,
    mapping_for_vmap,
    batched_ensemble.full_coordinates,
    md_params,
  )

  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(sampled_logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    pseudo_perplexity = jnp.exp(nll / jnp.sum(batched_ensemble.mask, axis=-1))
    return sampled_sequences, sampled_logits, pseudo_perplexity
  return sampled_sequences, sampled_logits, None


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
    return _sample_averaged_mode(spec, protein_iterator, model)

  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  if spec.output_h5_path:
    return _sample_streaming(spec, protein_iterator, sampler_fn)

  all_sequences, all_logits, all_pseudo_perplexities = [], [], []

  for batched_ensemble in protein_iterator:
    sampled_sequences, logits, pseudo_perplexity = _sample_batch(
      spec,
      batched_ensemble,
      sampler_fn,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)

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
    pad_to_max(
      jnp.ones(seq.shape, dtype=jnp.int32),
      max_len,
      axis=-1,
      pad_value=0,
    )
    for seq in all_sequences
  ]

  results = {
    "sequences": jnp.concatenate(all_sequences_padded, axis=0),
    "logits": jnp.concatenate(all_logits_padded, axis=0),
    "mask": jnp.concatenate(all_masks, axis=0),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = jnp.concatenate(all_pseudo_perplexities, axis=0)

  return results


def _sample_streaming(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  sampler_fn: Callable,
) -> dict[str, Any]:
  """Sample new sequences and stream results to an HDF5 file."""
  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch(
        spec,
        batched_ensemble,
        sampler_fn,
      )
      for i in range(sampled_sequences.shape[0]):
        grp = f.create_group(f"structure_{structure_idx}")
        grp.create_dataset("sequences", data=sampled_sequences[i], dtype="i4")
        grp.create_dataset("logits", data=sampled_logits[i], dtype="f4")
        if pseudo_perplexity is not None:
          grp.create_dataset("pseudo_perplexity", data=pseudo_perplexity[i], dtype="f4")
        # Store metadata about the structure
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[1]
        grp.attrs["num_noise_levels"] = sampled_sequences.shape[2]
        grp.attrs["num_temperatures"] = sampled_sequences.shape[3]
        grp.attrs["sequence_length"] = sampled_sequences.shape[4]
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }


def _create_decode_wrapper(base_decode_fn: Callable) -> Callable:
  """Create a custom decode wrapper that averages logits over structural features."""

  def wrapped(
    encoded_features: tuple,
    sequence: ProteinSequence,
    ar_mask_in: AutoRegressiveMask,
  ) -> Logits:
    avg_node, avg_edge, neighbors, mask, ar_mask_struct = encoded_features

    # Flatten batch dimensions
    neighbors_flat = neighbors.reshape(
      (-1, neighbors.shape[-2], neighbors.shape[-1]),
    )
    mask_flat = mask.reshape((-1, mask.shape[-1]))
    ar_mask_struct_flat = ar_mask_struct.reshape(
      (-1, ar_mask_struct.shape[-2], ar_mask_struct.shape[-1]),
    )

    def decode_single(
      n_idx: jnp.ndarray,
      m: jnp.ndarray,
      ar_m: jnp.ndarray,
    ) -> Logits:
      return base_decode_fn(
        (avg_node, avg_edge, n_idx, m, ar_m),
        sequence,
        ar_mask_in,
      )

    logits_batch = jax.vmap(decode_single)(neighbors_flat, mask_flat, ar_mask_struct_flat)
    return jnp.mean(logits_batch, axis=0)

  return wrapped


def _sample_averaged_mode(
  spec: SamplingSpecification,
  protein_iterator: Any,  # noqa: ANN401
  model: PrxteinMPNN,
) -> dict[str, Any]:
  """Run sampling in averaged node features mode."""
  if spec.output_h5_path:
    return _sample_streaming_averaged(spec, protein_iterator, model)

  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)
  all_sequences, all_logits, all_pseudo_perplexities = [], [], []

  for batched_ensemble in protein_iterator:
    sampled_sequences, logits, pseudo_perplexity = _sample_batch_averaged(
      spec,
      batched_ensemble,
      model,
      sample_fn,
      decode_fn,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)
    if pseudo_perplexity is not None:
      all_pseudo_perplexities.append(pseudo_perplexity)

  results = {
    "sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
  if all_pseudo_perplexities:
    results["pseudo_perplexity"] = jnp.concatenate(all_pseudo_perplexities, axis=0)

  return results


def _internal_sample_averaged(
  spec: SamplingSpecification,
  encoded_feat: tuple,
  keys_arr: PRNGKeyArray,
  sample_fn_with_params: Callable,
  tie_group_map: jnp.ndarray | None,
  num_groups: int | None,
) -> ProteinSequence:
  """Sample mapping over keys for averaged features."""
  decoding_order_keys = jax.random.split(jax.random.key(spec.random_seed + 1), spec.num_samples)

  temperature_array = jnp.asarray(spec.temperature, dtype=jnp.float32)

  def sample_single_sequence(
    key: PRNGKeyArray,
    decoding_order_key: PRNGKeyArray,
    encoded_feat: tuple,
    temperature: float,
  ) -> ProteinSequence:
    """Sample one sequence from averaged features."""
    seq_len = encoded_feat[0].shape[0]
    decoding_order, _ = random_decoding_order(
      decoding_order_key,
      seq_len,
      tie_group_map,
      num_groups,
    )
    return sample_fn_with_params(key, encoded_feat, decoding_order, temperature=temperature)

  def sample_for_key(k: PRNGKeyArray, dok: PRNGKeyArray) -> ProteinSequence:
    return jax.vmap(
      lambda t: sample_single_sequence(k, dok, encoded_feat, t),
    )(temperature_array)

  vmap_sample_fn = jax.vmap(
    sample_for_key,
    in_axes=(0, 0),
    out_axes=0,
  )
  return vmap_sample_fn(keys_arr, decoding_order_keys)


def _compute_logits_averaged(
  spec: SamplingSpecification,
  averaged_encodings: tuple,
  sampled_sequences: ProteinSequence,
  decode_fn_wrapped: Callable,
) -> Logits:
  """Compute logits for the sampled sequences."""
  seq_len = sampled_sequences.shape[-1]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  if spec.average_encoding_mode == "inputs_and_noise":

    def get_logits_local_both(seq: ProteinSequence) -> Logits:
      return jax.vmap(lambda s: decode_fn_wrapped(averaged_encodings, s, ar_mask))(seq)

    vmap_logits = jax.vmap(get_logits_local_both)
    logits = vmap_logits(sampled_sequences[0])
    logits = jnp.expand_dims(logits, axis=0)
  else:

    def get_logits_local(seq: ProteinSequence, enc: tuple) -> Logits:
      return jax.vmap(lambda s: decode_fn_wrapped(enc, s, ar_mask))(seq)

    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    vmap_logits = jax.vmap(
      jax.vmap(get_logits_local, in_axes=(0, None)),
      in_axes=(0, (0, 0, struct_axis, struct_axis, struct_axis)),
    )
    logits = vmap_logits(sampled_sequences, averaged_encodings)

  return logits


def _sample_batch_averaged(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: PrxteinMPNN,
  sample_fn: Callable,  # noqa: ARG001
  decode_fn: Callable,  # noqa: ARG001
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
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

  # Create a new sample_fn with the wrapper
  _, sample_fn_wrapped, decode_fn_wrapped = make_encoding_sampling_split_fn(
    model,
    decode_fn_wrapper=_create_decode_wrapper,
  )

  sample_fn_with_params = partial(
    sample_fn_wrapped,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
  )

  if spec.average_encoding_mode == "inputs_and_noise":
    sampled_sequences = _internal_sample_averaged(
      spec,
      averaged_encodings,
      keys,
      sample_fn_with_params,
      tie_group_map,
      num_groups,
    )
    sampled_sequences = jnp.expand_dims(sampled_sequences, axis=0)
  else:
    struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

    def _call_internal(enc: tuple) -> ProteinSequence:
      return _internal_sample_averaged(
        spec,
        enc,
        keys,
        sample_fn_with_params,
        tie_group_map,
        num_groups,
      )

    vmap_sample_structures = jax.vmap(
      _call_internal,
      in_axes=((0, 0, struct_axis, struct_axis, struct_axis),),
    )
    sampled_sequences = vmap_sample_structures(
      averaged_encodings,
    )

  logits = _compute_logits_averaged(spec, averaged_encodings, sampled_sequences, decode_fn_wrapped)

  num_temps = len(cast("Sequence[float]", spec.temperature))
  # Reshape to (1, -1, num_temps, seq_len)
  seq_len = sampled_sequences.shape[-1]
  sampled_sequences = sampled_sequences.reshape((1, -1, num_temps, seq_len))
  logits = logits.reshape((1, -1, num_temps, seq_len, 21))

  if num_temps == 1:
    sampled_sequences = jnp.squeeze(sampled_sequences, axis=2)
    logits = jnp.squeeze(logits, axis=2)

  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    pseudo_perplexity = jnp.exp(nll / jnp.sum(batched_ensemble.mask, axis=-1))
    return sampled_sequences, logits, pseudo_perplexity
  return sampled_sequences, logits, None


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: PrxteinMPNN,
) -> dict[str, Any]:
  """Sample new sequences with averaged encodings and stream results to an HDF5 file."""
  _, sample_fn, decode_fn = make_encoding_sampling_split_fn(model)

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      sampled_sequences, sampled_logits, pseudo_perplexity = _sample_batch_averaged(
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
        if pseudo_perplexity is not None:
          grp.create_dataset("pseudo_perplexity", data=pseudo_perplexity[i], dtype="f4")
        # Store metadata about the structure
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[1]
        grp.attrs["num_noise_levels"] = 1  # Averaged, so effectively 1 noise level
        grp.attrs["num_temperatures"] = (
          sampled_sequences.shape[2] if sampled_sequences.ndim == RANK_WITH_TEMPERATURE else 1
        )
        grp.attrs["sequence_length"] = sampled_sequences.shape[-1]
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
    },
  }
