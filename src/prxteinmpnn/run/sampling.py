"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from typing import Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.residue_constants import atom_order

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,
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
        num_workers: Number of parallel workers for data loading.
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
    residue_mask = batched_ensemble.atom_mask[:, :, atom_order["CA"]]
    keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

    vmap_samples = jax.vmap(
      sampler_fn,
      in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
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
      residue_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
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
  sampler_fn = make_sample_sequences(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  with h5py.File(spec.output_h5_path, "w") as f:
    seq_ds = f.create_dataset(
      "sequences",
      (0, 0, 0),
      maxshape=(None, None, None),
      chunks=True,
      dtype="i4",
    )
    logits_ds = f.create_dataset(
      "logits",
      (0, 0, 0, 0),
      maxshape=(None, None, None, None),
      chunks=True,
      dtype="f4",
    )

    for batched_ensemble in protein_iterator:
      residue_mask = batched_ensemble.atom_mask[:, :, atom_order["CA"]]
      keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

      vmap_samples = jax.vmap(
        sampler_fn,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
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
        residue_mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
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

      seq_ds.resize(seq_ds.shape[0] + sampled_sequences.shape[0], axis=0)
      seq_ds[-sampled_sequences.shape[0] :, :, :] = sampled_sequences

      logits_ds.resize(logits_ds.shape[0] + logits.shape[0], axis=0)
      logits_ds[-logits.shape[0] :, :, :, :] = logits

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
    },
  }
