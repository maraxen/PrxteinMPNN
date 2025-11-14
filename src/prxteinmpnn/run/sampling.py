"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.autoregression import resolve_tie_groups
from prxteinmpnn.utils.decoding_order import random_decoding_order

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

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

  protein_iterator, model = prep_protein_stream_and_model(spec)

  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=spec.sampling_strategy,
  )

  all_sequences, all_logits = [], []

  for batched_ensemble in protein_iterator:
    keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

    tie_group_map = None
    num_groups = None

    if spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
      tie_group_map = resolve_tie_groups(spec, batched_ensemble)
      num_groups = int(jnp.max(tie_group_map)) + 1

    vmap_samples = jax.vmap(
      sampler_fn,
      in_axes=(
        0,  # keys
        None,  # coordinates
        None,  # mask
        None,  # residue_index
        None,  # chain_index
        None,  # k_neighbors
        None,  # bias
        None,  # fixed_positions
        None,  # backbone_noise
        None,  # iterations
        None,  # learning_rate
        None,  # temperature
        None,  # tie_group_map
        None,  # num_groups
        None,  # multi_state_strategy
        None,  # multi_state_alpha
        None,  # interstructure mapping
      ),
      out_axes=0,
    )
    vmap_noises = jax.vmap(
      vmap_samples,
      in_axes=(
        None,  # keys
        None,  # coordinates
        None,  # mask
        None,  # residue_index
        None,  # chain_index
        None,  # k_neighbors
        None,  # bias
        None,  # fixed_positions
        0,  # backbone_noise
        None,  # iterations
        None,  # learning_rate
        None,  # temperature
        None,  # tie_group_map
        None,  # num_groups
        None,  # multi_state_strategy
        None,  # multi_state_alpha
        None,  # interstructure mapping
      ),
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
        None,  # tie_group_map
        None,  # num_groups
        None,  # multi_state_strategy
        None,  # multi_state_alpha
        0 if batched_ensemble.mapping is not None else None,  # interstructure mapping
      ),
      out_axes=0,
    )
    sampled_sequences, logits, _ = vmap_structures(
      keys,
      batched_ensemble.coordinates,
      batched_ensemble.mask,
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
      tie_group_map,
      num_groups,
      spec.multi_state_strategy,
      spec.multi_state_alpha,
      batched_ensemble.mapping,
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

  protein_iterator, model = prep_protein_stream_and_model(spec)

  if spec.average_encodings:
    msg = "average_encodings feature is temporarily disabled during Equinox migration"
    raise NotImplementedError(msg)

  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="temperature",
  )

  with h5py.File(spec.output_h5_path, "w") as f:
    structure_idx = 0

    for batched_ensemble in protein_iterator:
      keys = jax.random.split(jax.random.key(spec.random_seed), spec.num_samples)

      # Compute tie groups if needed
      tie_group_map = None
      num_groups = None

      # Handle inter-chain mode with tied positions
      if spec.pass_mode == "inter" and spec.tied_positions is not None:  # noqa: S105
        # For inter mode, loader has already concatenated structures with sequential chain IDs
        # Just resolve the tie groups on the concatenated protein
        tie_group_map = resolve_tie_groups(spec, batched_ensemble)
        num_groups = int(jnp.max(tie_group_map)) + 1

      noise_array = (
        jnp.asarray(spec.backbone_noise, dtype=jnp.float32)
        if spec.backbone_noise is not None
        else jnp.zeros(1)
      )

      # Create the sample function with bound parameters to avoid closure issues
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
  _spec: SamplingSpecification,
  _protein_iterator: Any,  # noqa: ANN401
  _model: Any,  # noqa: ANN401
) -> dict[str, str | dict[str, SamplingSpecification]]:
  """Sample with averaged encodings across noise levels.

  TODO: Re-implement this feature after Equinox migration is complete.
  This requires a way to separate encoding and sampling steps in the new architecture.
  See issue: TBD

  Args:
    _spec: Sampling specification (unused).
    _protein_iterator: Protein data iterator (unused).
    _model: PrxteinMPNN model (unused).

  Returns:
    Nothing, always raises NotImplementedError.

  Raises:
    NotImplementedError: This feature is temporarily disabled.

  """
  msg = "average_encodings feature is temporarily disabled during Equinox migration"
  raise NotImplementedError(msg)
