"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.sampling.sample import (
  make_encoding_sampling_split_fn,
  make_sample_sequences,
)
from prxteinmpnn.utils.decoding_order import random_decoding_order

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    BackboneCoordinates,
  BackboneDihedrals,
    BackboneNoise,
    ChainIndex,
    DecodingOrder,
    Logits,
    ProteinSequence,
    ResidueIndex,
  )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


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

  if spec.output_h5_path:
    return _sample_streaming(spec)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)
  sampler_fn = make_sample_sequences(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  all_sequences, all_logits = [], []

  for batched_ensemble in protein_iterator:
    keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

    vmap_samples = jax.vmap(
      sampler_fn,
      in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None),
      out_axes=0,
    )
    vmap_noises = jax.vmap(
      vmap_samples,
      in_axes=(None, None, None, None, None, None, None, None, 0, None, None, None),
      out_axes=0,
    )
    vmap_structures = jax.vmap(
      vmap_noises,
      in_axes=(
        None,  # keys
        0,  # coordinates
        0,  # residue_mask
        0,  # residue_index
        0,  # chain_index
        0,  # dihedrals
        None,  # k_neighbors
        None,  # bias
        None,  # fixed_positions
        None,  # backbone_noise
        None,  # iterations
        None,  # learning_rate
        None,  # temperature
      ),
      out_axes=0,
    )
    sampled_sequences, logits, _ = vmap_structures(
      keys,
      batched_ensemble.coordinates,
      batched_ensemble.mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      batched_ensemble.dihedrals,
      48,
      jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
      jnp.asarray(spec.fixed_positions, dtype=jnp.int32)
      if spec.fixed_positions is not None
      else None,
      jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
      if spec.backbone_noise is not None
      else None,
      spec.iterations,
      spec.learning_rate,
      spec.temperature,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)

  if not all_sequences:
    return {"sampled_sequences": None, "logits": None, "metadata": None}

  return {
    "sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "specification": spec,
    },
  }


def _sample_streaming(
  spec: SamplingSpecification,
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample new sequences and stream results to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)

  if spec.average_encodings:
    return _sample_streaming_averaged(spec, protein_iterator, model_parameters)

  sampler_fn = make_sample_sequences(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

      noise_array = (
        jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
        if spec.backbone_noise is not None
        else jnp.zeros(1)
      )

      def sample_single_noise(
        key: PRNGKeyArray,
        coords: BackboneCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
        noise: BackboneNoise,
      ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        """Sample one sequence for one structure at one noise level."""
        return sampler_fn(
          key,
          coords,
          mask,
          residue_ix,
          chain_ix,
          dihedrals,
          48,
          jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          jnp.asarray(spec.fixed_positions, dtype=jnp.int32)
          if spec.fixed_positions is not None
          else None,
          noise,
          spec.iterations,
          spec.learning_rate,
          spec.temperature,
        )

      def mapped_fn_noise(
        key: PRNGKeyArray,
        coords: BackboneCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
        noise_array: BackboneNoise = noise_array,
      ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        """Compute samples across all noise levels for a single structure/sample."""
        return jax.lax.map(
          partial(
            sample_single_noise,
            key,  # Pass key as constant for the noise map
            coords,
            mask,
            residue_ix,
            chain_ix,
            dihedrals,
          ),
          noise_array,
          batch_size=spec.noise_batch_size,
        )

      def internal_sample(
        coords: BackboneCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
        keys: PRNGKeyArray = keys,
      ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        """Sample mapping over keys and noise."""
        noise_map_fn = partial(
          mapped_fn_noise,
          coords=coords,
          mask=mask,
          residue_ix=residue_ix,
          chain_ix=chain_ix,
          dihedrals=dihedrals,
        )

        return jax.lax.map(
          noise_map_fn,
          keys,
          batch_size=spec.samples_batch_size,
        )

      vmap_structures = jax.vmap(internal_sample)

      sampled_sequences, sampled_logits, _ = vmap_structures(
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        batched_ensemble.dihedrals,
      )

      # Store each structure in its own group to handle variable lengths
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


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: Any,  # noqa: ANN401
  model_parameters: Any,  # noqa: ANN401
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample with averaged encodings across noise levels."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  encode_fn, sample_fn = make_encoding_sampling_split_fn(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  noise_array = (
    jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
    if spec.backbone_noise is not None
    else jnp.zeros(1, dtype=jnp.float32)
  )

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      # Process each structure in the batch
      for struct_idx in range(batched_ensemble.coordinates.shape[0]):
        coords = batched_ensemble.coordinates[struct_idx]
        mask = batched_ensemble.mask[struct_idx]
        residue_ix = batched_ensemble.residue_index[struct_idx]
        chain_ix = batched_ensemble.chain_index[struct_idx]
        dihedrals = batched_ensemble.dihedrals[struct_idx]

        # Encode across all noise levels
        encode_single_noise = partial(
          encode_fn,
          jax.random.key(spec.random_seed),
          coords,
          mask,
          residue_ix,
          chain_ix,
          dihedrals,
          48,
        )

        # Get encodings for all noise levels
        encodings_and_orders = jax.lax.map(
          encode_single_noise,
          noise_array,
          batch_size=spec.noise_batch_size,
        )

        # Split encodings from decoding orders
        all_encodings = encodings_and_orders[0]  # First element is encodings
        # Use first decoding order and ensure it's int32 (same across noise levels)
        decoding_order = encodings_and_orders[1][0].astype(jnp.int32)

        # Average the encodings across noise levels
        # Note: We average node_features and edge_features, but use the first
        # noise level's neighbor_indices (following jacobian.py pattern)
        avg_node_features = jnp.mean(all_encodings["node_features"], axis=0)
        avg_edge_features = jnp.mean(all_encodings["edge_features"], axis=0)

        # use struct_idx to get neighbor_indices and mask since these are strict
        neighbor_indices = all_encodings["neighbor_indices"][struct_idx]
        mask = all_encodings["mask"][struct_idx]

        avg_encodings = {
          "node_features": avg_node_features,
          "edge_features": avg_edge_features,
          "neighbor_indices": neighbor_indices,
          "mask": mask,
        }

        # Sample multiple sequences using the averaged encoding
        keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

        sample_single = partial(
          sample_fn,
          encoded_features=avg_encodings,
          decoding_order=decoding_order,
          bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
          iterations=spec.iterations,
          learning_rate=spec.learning_rate,
          temperature=spec.temperature,
          sampling_strategy=spec.sampling_strategy,
        )

        # Sample in batches
        sampled_results = jax.lax.map(
          sample_single,
          keys,
          batch_size=spec.samples_batch_size,
        )

        sampled_sequences, sampled_logits = sampled_results

        # Add noise dimension for consistency
        sampled_sequences = jnp.expand_dims(sampled_sequences, axis=1)
        sampled_logits = jnp.expand_dims(sampled_logits, axis=1)

        # Store in HDF5
        grp = f.create_group(f"structure_{structure_idx}")
        grp.create_dataset("sequences", data=sampled_sequences, dtype="i4")
        grp.create_dataset("logits", data=sampled_logits, dtype="f4")
        grp.attrs["structure_index"] = structure_idx
        grp.attrs["num_samples"] = sampled_sequences.shape[0]
        grp.attrs["num_noise_levels"] = 1  # Averaged, so effectively 1 level
        grp.attrs["sequence_length"] = sampled_sequences.shape[2]
        grp.attrs["averaged_encodings"] = True
        structure_idx += 1

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
    },
  }
