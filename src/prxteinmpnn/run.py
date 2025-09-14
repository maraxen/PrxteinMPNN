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

from prxteinmpnn.io.process import load
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.batching import (
  batch_and_pad_proteins,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.residue_constants import atom_order

AlignmentStrategy = Literal["sequence", "structure"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def tuple_to_protein(t: ProteinTuple) -> Protein:
  """Convert a ProteinTuple to a Protein dataclass."""
  return Protein(
    coordinates=jnp.array(t[0]).astype(jnp.float32),
    aatype=jnp.array(t[1]).astype(jnp.int8),
    atom_mask=jnp.array(t[2]).astype(jnp.float16),
    residue_index=jnp.array(t[3]).astype(jnp.int32),
    chain_index=jnp.array(t[4]).astype(jnp.int32),
    dihedrals=None if t[5] is None else jnp.array(t[5]).astype(jnp.float32),
    one_hot_sequence=jax.nn.one_hot(jnp.array(t[1]), 21, dtype=jnp.float32),
  )


def compute_cross_protein_jacobian_diffs(
  jacobians: jax.Array,
  mapping: jax.Array,
) -> jax.Array:
  """Compute cross-protein Jacobian differences using alignment mapping.

  Args:
    jacobians: Jacobian tensors of shape (batch_size, noise_levels, L, 21, L, 21)
    mapping: Cross-protein mapping of shape (num_pairs, max_length, 2)

  Returns:
    Cross-protein Jacobian differences where mapping is valid, NaN otherwise.
    Shape: (num_pairs, noise_levels, L, 21, L, 21)

  """
  batch_size, noise_levels, max_len, _, _, _ = jacobians.shape
  n_pairs = mapping.shape[0] if mapping.size > 0 else 0

  if n_pairs == 0:
    return jnp.empty((0, noise_levels, max_len, 21, max_len, 21))

  def compute_pair_diff(
    pair_idx: jax.Array,
    protein_indices: jax.Array,
  ) -> jax.Array:
    """Compute Jacobian difference for a single protein pair."""
    pair_mapping = mapping[pair_idx]  # (max_length, 2)
    protein_i_idx, protein_j_idx = protein_indices

    # Check if the pair is valid (both proteins exist)
    pair_valid = protein_j_idx < batch_size

    # Use safe indexing to avoid out of bounds access
    safe_j_idx = jnp.minimum(protein_j_idx, batch_size - 1)

    # Get the Jacobians for both proteins
    jac_i = jacobians[protein_i_idx]  # (noise_levels, max_len, 21, max_len, 21)
    jac_j = jacobians[safe_j_idx]  # (noise_levels, max_len, 21, max_len, 21)

    # Compute base difference
    diff = jac_i - jac_j

    # Create validity mask based on alignment
    # Check both that the position is not -1 AND that it's within valid range
    valid_mask = (
      (pair_mapping[:, 0] != -1)
      & (pair_mapping[:, 1] != -1)
      & (pair_mapping[:, 0] < max_len)
      & (pair_mapping[:, 1] < max_len)
      & (pair_mapping[:, 0] >= 0)
      & (pair_mapping[:, 1] >= 0)
    )

    # Create a position-wise mask for the Jacobian
    def create_position_mask(pos_i: int, pos_j: int) -> jax.Array:
      """Check if both positions are valid in the alignment."""
      return valid_mask[pos_i] & valid_mask[pos_j]

    # Apply mask to all position pairs
    position_indices = jnp.arange(max_len)
    position_mask = jax.vmap(
      lambda i: jax.vmap(lambda j: create_position_mask(i, j))(position_indices),
    )(position_indices)

    # Broadcast position mask over noise levels and amino acid dimensions
    # position_mask is (max_len, max_len), we need (noise_levels, max_len, 21, max_len, 21)
    # Also combine with pair validity
    mask_broadcast = pair_valid & position_mask[None, :, None, :, None]

    # Apply mask: where valid, use difference; where invalid, use NaN
    return jnp.where(mask_broadcast, diff, jnp.nan)

  pair_indices = jnp.arange(n_pairs)
  rows, cols = jnp.triu_indices(batch_size, k=1)
  protein_indices = jnp.stack([rows, cols], axis=-1)
  jax.debug.print("protein_indices: {}", protein_indices)

  return jax.vmap(compute_pair_diff)(pair_indices, protein_indices)


async def score(
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
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str]] | None]:
  """Score all provided sequences against all input structures.

  This function streams and processes structures asynchronously, then uses a
  memory-efficient JAX map
  to score all provided sequences against each structure.

  Args:
      inputs: An async stream of structures (files, PDB IDs, etc.).
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
        If None, a full autoregressive mask will be used.
      batch_size: The batch size for processing structures.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing scores, logits, and metadata. The scores will
      have a shape of (num_structures, num_noise_levels, num_sequences).

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  _proteins, sources = await load(
    inputs,
    model=model,
    altloc=altloc,
    chain_id=chain_id,
    foldcomp_database=foldcomp_database,
    **kwargs,
  )

  proteins = [tuple_to_protein(p) for p in _proteins]

  # Convert string sequences to integer arrays
  if sequences_to_score:
    integer_sequences = [string_to_protein_sequence(s) for s in sequences_to_score]
  else:
    integer_sequences = None

  batched_ensemble, batched_sequences = batch_and_pad_proteins(
    proteins,
    sequences_to_score=integer_sequences,
  )

  if batched_sequences is None:
    msg = "sequences_to_score must be provided to the score function."
    raise ValueError(msg)

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  score_single_pair = make_score_sequence(model_parameters=model_parameters)

  if ar_mask is None:
    ar_mask = 1 - jnp.eye(batched_ensemble.aatype.shape[1], dtype=jnp.bool_)

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

  mapped_fn = partial(
    vmap_noises,
    prng_key=jax.random.key(rng_key),  # type: ignore[arg-type]
    sequence=batched_sequences,  # type: ignore[arg-type]
    k_neighbors=48,  # type: ignore[arg-type]
    ar_mask=ar_mask,  # type: ignore[arg-type]
    backbone_noise=backbone_noise,  # type: ignore[arg-type]
  )

  scores, logits, _ = jax.lax.map(
    mapped_fn,
    (
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
    ),
    batch_size=batch_size,
  )

  return {
    "scores": scores,
    "logits": logits,
    "mapping": batched_ensemble.mapping,
    "metadata": {
      "protein_sources": sources,
      "backbone_noise_levels": backbone_noise,
    },
  }


async def sample(
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
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str]] | None]:
  """Sample new sequences for the given input structures.

  This function streams and processes structures asynchronously, then uses a
  memory-efficient JAX map to sample new sequences for each structure.

  Args:
      inputs: An async stream of structures (files, PDB IDs, etc.).
      chain_id: Specific chain(s) to parse from the structure.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
      backbone_noise: The amount of noise to add to the backbone. Can be a single
          float or a list/array of floats to sample with multiple noise levels.
      num_samples: The number of sequences to sample per structure/noise level.
      sampling_strategy: The sampling strategy to use ("temperature" or "straight_through").
      temperature: The sampling temperature.
      bias: An optional array to bias the logits.
      fixed_positions: An optional array of residue indices to keep fixed.
      iterations: Number of optimization iterations for "straight_through" sampling.
      learning_rate: Learning rate for "straight_through" sampling.
      batch_size: The batch size for processing structures.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing sampled sequences, logits, and metadata. The sequences
      will have a shape of (num_structures, num_noise_levels, num_samples, length).

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  _proteins, sources = await load(
    inputs,
    model=model,
    altloc=altloc,
    chain_id=chain_id,
    foldcomp_database=foldcomp_database,
    **kwargs,
  )

  proteins = [tuple_to_protein(p) for p in _proteins]

  batched_ensemble, _ = batch_and_pad_proteins(proteins)

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  sampler_fn = make_sample_sequences(
    model_parameters=model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy=sampling_strategy,
  )

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

  residue_mask = batched_ensemble.atom_mask[:, :, atom_order["CA"]]

  mapped_fn = partial(
    vmap_noises,
    keys,
    k_neighbors=48,
    bias=bias,
    fixed_positions=fixed_positions,
    backbone_noise=backbone_noise,
    iterations=iterations,
    learning_rate=learning_rate,
    temperature=temperature,
  )

  sampled_sequences, logits, _ = jax.lax.map(
    mapped_fn,
    (
      batched_ensemble.coordinates,
      residue_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
    ),
    batch_size=batch_size,
  )

  return {
    "sampled_sequences": sampled_sequences,
    "logits": logits,
    "mapping": batched_ensemble.mapping,
    "metadata": {
      "protein_sources": sources,
      "backbone_noise_levels": backbone_noise,
    },
  }


async def categorical_jacobian(
  inputs: Sequence[str | StringIO] | str | StringIO,
  chain_id: Sequence[str] | str | None = None,
  model: int | None = None,
  altloc: Literal["first", "all"] = "first",
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  mode: Literal["full", "diagonal"] = "full",
  outer_batch_size: int = 1,
  inner_batch_size: int = 16,
  *,
  calculate_cross_diff: bool = False,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str] | str] | None]:
  """Compute the Jacobian of the model's logits with respect to the input sequence.

  This function calculates the derivative of the output logits at all positions
  with respect to the one-hot encoded input sequence at all positions.

  Args:
      inputs: An async stream of structures (files, PDB IDs, etc.).
      chain_id: Specific chain(s) to parse from the structure.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
      backbone_noise: The amount of noise to add to the backbone.
      mode: "full" to compute the full Jacobian, "diagonal" for only diagonal blocks.
      outer_batch_size: The batch size for processing structures.
      calculate_cross_diff: Whether to calculate cross-protein differences using mapping.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing the Jacobian tensor and metadata. The shape of the
      Jacobian will be (num_structures, num_noise_levels, L, 21, L, 21).

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  logger.info("Computing categorical Jacobian in %s mode.", mode)
  _proteins, sources = await load(
    inputs,
    model=model,
    altloc=altloc,
    chain_id=chain_id,
    foldcomp_database=foldcomp_database,
    **kwargs,
  )
  logger.info("Loaded protein stream, batching and padding proteins.")

  proteins = [tuple_to_protein(p) for p in _proteins]

  batched_ensemble, _ = batch_and_pad_proteins(
    proteins,
    calculate_cross_diff=calculate_cross_diff,
  )
  logger.info("Batched and padded proteins, loading model.")

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  logger.info("Loaded model parameters.")

  conditional_logits_fn = make_conditional_logits_fn(model_parameters=model_parameters)
  logger.info("Created scoring function.")

  def compute_jacobian_for_structure(
    coords: jax.Array,
    atom_mask: jax.Array,
    residue_ix: jax.Array,
    chain_ix: jax.Array,
    one_hot_sequence: jax.Array,
    noise: jax.Array,
  ) -> jax.Array:
    """Compute the full categorical Jacobian for a single protein structure and noise level."""
    length = one_hot_sequence.shape[0]
    residue_mask = atom_mask[:, 0]
    one_hot_flat = one_hot_sequence.flatten()
    input_dim = one_hot_flat.shape[0]

    def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
      """Compute logits from flattened one-hot sequence."""
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
      return logits.flatten()  # Shape: (length * 21,)

    def jvp_fn(tangent: jax.Array) -> jax.Array:
      """Compute the Jacobian-vector product for a single tangent vector."""
      return jax.jvp(logit_fn, (one_hot_flat,), (tangent,))[1]

    def chunked_jacobian(idx: jax.Array) -> jax.Array:
      """Compute the Jacobian matrix of logits with respect to the input one-hot sequence."""
      tangent = jax.nn.one_hot(idx, num_classes=input_dim, dtype=one_hot_flat.dtype)
      return jvp_fn(tangent)

    jacobian_flat = jax.lax.map(
      chunked_jacobian,
      jnp.arange(input_dim),
      batch_size=inner_batch_size,
    )

    return jacobian_flat.reshape(length, 21, length, 21)

  vmap_over_noise = jax.vmap(
    compute_jacobian_for_structure,
    in_axes=(None, None, None, None, None, 0),
    out_axes=(0, 0),
  )

  def mapped_fn(
    input_tuple: tuple[
      StructureAtomicCoordinates,
      AtomMask,
      ResidueIndex,
      ChainIndex,
      OneHotProteinSequence,
    ],
  ) -> jax.Array:
    """Map over a single structure, vector mapping over noise levels."""
    return vmap_over_noise(*input_tuple, backbone_noise)

  jacobians: jax.Array = jax.lax.map(
    mapped_fn,
    (
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      batched_ensemble.one_hot_sequence,  # type: ignore[arg-type]
    ),
    batch_size=outer_batch_size,
  )
  logger.info("Computed categorical Jacobians.")
  apc_jacobians = jax.vmap(
    jax.vmap(
      apc_corrected_frobenius_norm,
      in_axes=0,
      out_axes=0,
    ),
    in_axes=0,
    out_axes=0,
  )(jacobians)

  cross_protein_diffs = None
  if calculate_cross_diff and batched_ensemble.mapping is not None:
    cross_protein_diffs = compute_cross_protein_jacobian_diffs(
      apc_jacobians,
      batched_ensemble.mapping,
    )

    if cross_protein_diffs.size > 0:
      not_nan_mask = ~jnp.isnan(cross_protein_diffs).all(axis=(1, 3, 5))
      valid_residues = not_nan_mask.any(axis=(0, 2)) | not_nan_mask.any(axis=(0, 1))
      valid_indices = jnp.where(valid_residues, size=valid_residues.sum().item())[0]

      if valid_indices.size > 0:
        cross_protein_diffs = cross_protein_diffs[:, :, valid_indices][:, :, :, :, valid_indices, :]
      else:
        cross_protein_diffs = jnp.empty(
          (
            cross_protein_diffs.shape[0],
            cross_protein_diffs.shape[1],
            0,
            21,
            0,
            21,
          ),
        )
    logger.info("Computed cross-protein Jacobian differences.")

  return {
    "categorical_jacobians": jacobians,
    "cross_protein_diffs": cross_protein_diffs,
    "mapping": batched_ensemble.mapping,
    "metadata": {
      "protein_sources": sources,
      "backbone_noise_levels": backbone_noise,
      "computation_mode": mode,
    },
  }
