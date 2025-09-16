"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

mp.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Sequence
  from io import StringIO

  from jaxtyping import ArrayLike

  from prxteinmpnn.mpnn import ModelVersion, ModelWeights
  from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase
  from prxteinmpnn.utils.types import (
    AtomMask,
    ChainIndex,
    OneHotProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from prxteinmpnn.io import loaders
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.catjac import CombineCatJacTranformFn, make_combine_jac
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.residue_constants import atom_order

AlignmentStrategy = Literal["sequence", "structure"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs


def score(
  inputs: Sequence[str | StringIO] | str | StringIO,
  sequences_to_score: Sequence[str],
  chain_id: Sequence[str] | str | None = None,
  model: int | None = None,
  altloc: Literal["first", "all"] = "first",
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  ar_mask: None | ArrayLike = None,
  batch_size: int = 32,
  num_workers: int = 0,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str]] | None]:
  """Score all provided sequences against all input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then scores all provided sequences against each structure.

  Args:
      inputs: A single or sequence of inputs (files, PDB IDs, etc.).
      sequences_to_score: A list of protein sequences (strings) to score.
      chain_id: Specific chain(s) to parse from the structure.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
      backbone_noise: The amount of noise to add to the backbone.
      ar_mask: An optional array of shape (L, L) to mask out certain residue pairs.
      batch_size: The number of structures to process in a single batch.
      num_workers: Number of parallel workers for data loading.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing scores, logits, and metadata.

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  # Prepare sequences to be scored
  if not sequences_to_score:
    msg = (
      "No sequences provided for scoring. `sequences_to_score` must be a non-empty list of strings."
    )
    raise ValueError(msg)
  integer_sequences = [string_to_protein_sequence(s) for s in sequences_to_score]
  batched_sequences, _ = jnp.concatenate(integer_sequences)

  parse_kwargs = {"chain_id": chain_id, "model": model, "altloc": altloc, **kwargs}
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(inputs),
    batch_size=batch_size,
    foldcomp_database=foldcomp_database,
    parse_kwargs=parse_kwargs,
    num_workers=num_workers,
  )

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  score_single_pair = make_score_sequence(model_parameters=model_parameters)

  all_scores, all_logits = [], []

  for batched_ensemble in protein_iterator:
    max_len = batched_ensemble.coordinates.shape[1]
    current_ar_mask = (
      1 - jnp.eye(max_len, dtype=jnp.bool_) if ar_mask is None else jnp.asarray(ar_mask)
    )

    vmap_sequences = jax.vmap(
      score_single_pair,
      in_axes=(None, 0, None, None, None, None, None, None, None),
      out_axes=0,
    )
    vmap_noises = jax.vmap(
      vmap_sequences,
      in_axes=(None, None, None, None, None, None, None, 0, None),
      out_axes=0,
    )
    vmap_structures = jax.vmap(
      vmap_noises,
      in_axes=(None, None, 0, 0, 0, 0, None, None, None),
      out_axes=0,
    )
    scores, logits, _ = vmap_structures(
      jax.random.key(rng_key),
      batched_sequences,
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      48,
      backbone_noise,
      current_ar_mask,
    )
    all_scores.append(scores)
    all_logits.append(logits)

  if not all_scores:
    return {"scores": None, "logits": None, "metadata": None}

  return {
    "scores": jnp.concatenate(all_scores, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "backbone_noise_levels": backbone_noise,
    },
  }


def sample(
  inputs: Sequence[str | StringIO] | str | StringIO,
  chain_id: Sequence[str] | str | None = None,
  model: int | None = None,
  altloc: Literal["first", "all"] = "first",
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  num_samples: int = 1,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  temperature: float = 0.1,
  bias: ArrayLike | None = None,
  fixed_positions: ArrayLike | None = None,
  iterations: int | None = None,
  learning_rate: float | None = None,
  batch_size: int = 32,
  num_workers: int = 0,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str]] | None]:
  """Sample new sequences for the given input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then samples new sequences for each structure.

  Args:
      inputs: A single or sequence of inputs (files, PDB IDs, etc.).
      chain_id: Specific chain(s) to parse from the structure.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
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
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  parse_kwargs = {"chain_id": chain_id, "model": model, "altloc": altloc, **kwargs}
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(inputs),
    batch_size=batch_size,
    foldcomp_database=foldcomp_database,
    parse_kwargs=parse_kwargs,
    num_workers=num_workers,
  )

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  sampler_fn = make_sample_sequences(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=sampling_strategy,
  )

  all_sequences, all_logits = [], []

  for batched_ensemble in protein_iterator:
    residue_mask = batched_ensemble.atom_mask[:, :, atom_order["CA"]]
    keys = jax.random.split(jax.random.key(rng_key), num_samples)

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
      bias,
      fixed_positions,
      backbone_noise,
      iterations,
      learning_rate,
      temperature,
    )
    all_sequences.append(sampled_sequences)
    all_logits.append(logits)

  if not all_sequences:
    return {"sampled_sequences": None, "logits": None, "metadata": None}

  return {
    "sampled_sequences": jnp.concatenate(all_sequences, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "backbone_noise_levels": backbone_noise,
    },
  }


def categorical_jacobian(
  inputs: Sequence[str | StringIO] | str | StringIO,
  chain_id: Sequence[str] | str | None = None,
  model: int | None = None,
  altloc: Literal["first", "all"] = "first",
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  batch_size: int = 1,
  noise_batch_size: int = 1,
  jacobian_batch_size: int = 16,
  combine_batch_size: int = 4,
  num_workers: int = 0,
  combine_fn: CombineCatJacTranformFn | Literal["add", "subtract"] = "add",
  combine_fn_kwargs: dict[str, Any] | None = None,
  combine_weights: jax.Array | None = None,
  *,
  combine: bool = False,
  **kwargs: Any,  # noqa: ANN401
) -> dict[
  str,
  jax.Array
  | dict[
    str,
    jax.Array | Sequence[str] | str | int | dict[str, Any] | FoldCompDatabase | None,
  ]
  | None,
]:
  """Compute the Jacobian of the model's logits with respect to the input sequence.

  Args:
      inputs: A single or sequence of inputs (files, PDB IDs, etc.).
      chain_id: Specific chain(s) to parse from the structure.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
      backbone_noise: The amount of noise to add to the backbone.
      mode: "full" to compute the full Jacobian, "diagonal" for only diagonal blocks.
      batch_size: The number of structures to process in a single batch.
      noise_batch_size: Batch size for noise levels in Jacobian computation.
      jacobian_batch_size: Inner batch size for Jacobian computation.
      num_workers: Number of parallel workers for data loading.
      calculate_cross_diff: Whether to calculate cross-protein differences.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing the Jacobian tensor and metadata.

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  parse_kwargs = {"chain_id": chain_id, "model": model, "altloc": altloc, **kwargs}
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(inputs),
    batch_size=batch_size,
    foldcomp_database=foldcomp_database,
    parse_kwargs=parse_kwargs,
    num_workers=num_workers,
  )

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  conditional_logits_fn = make_conditional_logits_fn(model_parameters=model_parameters)

  all_jacobians = []
  all_sequences = []
  for batched_ensemble in protein_iterator:

    def compute_jacobian_for_structure(
      coords: jax.Array,
      atom_mask: jax.Array,
      residue_ix: jax.Array,
      chain_ix: jax.Array,
      one_hot_sequence: jax.Array,
      noise: jax.Array,
    ) -> jax.Array:
      length = one_hot_sequence.shape[0]
      residue_mask = atom_mask[:, 0]
      one_hot_flat = one_hot_sequence.flatten()
      input_dim = one_hot_flat.shape[0]

      def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
        one_hot_2d = one_hot_flat.reshape(length, 21)
        logits, _, _ = conditional_logits_fn(
          jax.random.key(rng_key),
          coords,
          one_hot_2d,
          residue_mask,
          residue_ix,
          chain_ix,
          None,
          48,
          noise,
        )
        return logits.flatten()

      def jvp_fn(tangent: jax.Array) -> jax.Array:
        return jax.jvp(logit_fn, (one_hot_flat,), (tangent,))[1]

      def chunked_jacobian(idx: jax.Array) -> jax.Array:
        tangent = jax.nn.one_hot(idx, num_classes=input_dim, dtype=one_hot_flat.dtype)
        return jvp_fn(tangent)

      jacobian_flat = jax.lax.map(
        chunked_jacobian,
        jnp.arange(input_dim),
        batch_size=jacobian_batch_size,
      )
      return jacobian_flat.reshape(length, 21, length, 21)

    def mapped_fn(
      coords: StructureAtomicCoordinates,
      atom_mask: AtomMask,
      residue_ix: ResidueIndex,
      chain_ix: ChainIndex,
      one_hot_sequence: OneHotProteinSequence,
    ) -> jax.Array:
      """Compute Jacobians for a single structure across multiple noise levels."""
      return jax.lax.map(
        partial(
          compute_jacobian_for_structure,
          coords,
          atom_mask,
          residue_ix,
          chain_ix,
          one_hot_sequence,
        ),
        backbone_noise,
        batch_size=noise_batch_size,
      )

    jacobians_batch = jax.vmap(mapped_fn)(
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      batched_ensemble.one_hot_sequence,
    )

    all_jacobians.append(jacobians_batch)
    all_sequences.append(batched_ensemble.one_hot_sequence)

  if not all_jacobians:
    return {"categorical_jacobians": None, "metadata": None}

  jacobians = jnp.concatenate(all_jacobians, axis=0)
  apc_jacobians = jax.vmap(jax.vmap(apc_corrected_frobenius_norm))(jacobians)

  combine_jacs_fn = make_combine_jac(
    combine_fn=combine_fn,
    fn_kwargs=combine_fn_kwargs,
    batch_size=combine_batch_size,
  )

  combined_jacs = (
    combine_jacs_fn(jacobians, jnp.concatenate(all_sequences, axis=0), combine_weights)
    if combine
    else None
  )

  return {
    "categorical_jacobians": jacobians,
    "apc_corrected_jacobians": apc_jacobians,
    "combined": combined_jacs,
    "metadata": {
      "backbone_noise_levels": backbone_noise,
      "chain_id": chain_id,
      "model": model,
      "altloc": altloc,
      "model_version": model_version,
      "model_weights": model_weights,
      "combine_function": combine_fn if isinstance(combine_fn, str) else combine_fn.__name__,
      "combine_function_kwargs": combine_fn_kwargs,
      "combine_weights": combine_weights,
      "foldcomp_database": foldcomp_database,
      "jacobian_batch_size": jacobian_batch_size,
      "noise_batch_size": noise_batch_size,
      "combine_batch_size": combine_batch_size,
      "num_workers": num_workers,
      "rng_key": rng_key,
    },
  }
