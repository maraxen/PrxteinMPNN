"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

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
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


from prxteinmpnn.io.process import load
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.batching import (
  batch_and_pad_proteins,
)

AlignmentStrategy = Literal["sequence", "structure"]

logger = logging.getLogger(__name__)
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


async def score(
  inputs: Sequence[str | StringIO] | str | StringIO,
  sequences_to_score: Sequence[str],
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  ar_mask: None | ArrayLike = None,
  batch_size: int = 32,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str]]]:
  """Score all provided sequences against all input structures.

  This function streams and processes structures asynchronously, then uses a
  memory-efficient JAX map to perform the scoring on a GPU or TPU.

  Args:
      inputs: An async stream of structures (files, PDB IDs, etc.).
      sequences_to_score: A list of protein sequences (strings) to score.
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

  protein_stream = load(inputs, foldcomp_database=foldcomp_database, **kwargs)

  batched_proteins, sources, batched_sequences = await batch_and_pad_proteins(
    protein_stream,
    sequences_to_score=sequences_to_score,
  )

  if batched_sequences is None:
    msg = "sequences_to_score must be provided to the score function."
    raise ValueError(msg)

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  score_single_pair = make_score_sequence(model_parameters=model_parameters)

  if ar_mask is None:
    ar_mask = 1 - jnp.eye(batched_proteins.aatype.shape[1], dtype=jnp.bool_)

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
      batched_proteins.coordinates,
      batched_proteins.atom_mask,
      batched_proteins.residue_index,
      batched_proteins.chain_index,
    ),
    batch_size=batch_size,
  )

  return {
    "scores": scores,
    "logits": logits,
    "metadata": {
      "protein_sources": sources,
      "backbone_noise_levels": backbone_noise,
    },
  }


async def categorical_jacobian(
  inputs: Sequence[str | StringIO] | str | StringIO,
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  foldcomp_database: FoldCompDatabase | None = None,
  rng_key: int = 0,
  backbone_noise: float | list[float] | ArrayLike = 0.0,
  mode: Literal["full", "diagonal"] = "full",
  batch_size: int = 32,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, jax.Array | dict[str, jax.Array | list[str] | str] | None]:
  """Compute the Jacobian of the model's logits with respect to the input sequence.

  This function calculates the derivative of the output logits at all positions
  with respect to the one-hot encoded input sequence at all positions.

  ⚠️ **Warning on Memory Usage**: The full Jacobian tensor for a protein of
  length L is of shape (L, 21, L, 21), which can too large to fit in
  memory, especially if you are examining it across many structures. As an option,
  this function can compute only the diagonal blocks of the Jacobian, i.e.,
  (∂ logits[i] / ∂ seq[i]), resulting in a much more manageable shape of
  (L, 21, 21). To compute this diagonal matrix, set `mode="diagonal"`.

  Args:
      inputs: An async stream of structures (files, PDB IDs, etc.).
      model_version: The model version to use.
      model_weights: The model weights to use.
      foldcomp_database: The FoldComp database to use for FoldComp IDs.
      rng_key: The random number generator key.
      backbone_noise: The amount of noise to add to the backbone.
      mode: "full" to compute the full Jacobian, "diagonal" for only diagonal blocks.
      batch_size: The batch size for processing structures.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing the Jacobian tensor and metadata. The shape of the
      Jacobian will be (num_structures, num_noise_levels, L, 21, 21) if
      `compute_diagonal_only` is True, or (num_structures, num_noise_levels,
      L, 21, L, 21) if False.

  """
  if isinstance(backbone_noise, float):
    backbone_noise = jnp.array([backbone_noise])
  else:
    backbone_noise = jnp.asarray(backbone_noise)

  logger.info("Computing categorical Jacobian in %s mode.", mode)

  protein_stream = load(inputs, foldcomp_database=foldcomp_database, **kwargs)

  logger.info("Loaded protein stream, batching and padding proteins.")

  batched_proteins, sources, _ = await batch_and_pad_proteins(protein_stream)

  logger.info("Batched and padded proteins, loading model.")

  model_parameters = get_mpnn_model(model_version=model_version, model_weights=model_weights)
  logger.info("Loaded model parameters.")
  score_single_pair = make_score_sequence(model_parameters=model_parameters)
  logger.info("Created scoring function.")

  def compute_jacobian_for_structure(
    coords: jax.Array,
    atom_mask: jax.Array,
    residue_ix: jax.Array,
    chain_ix: jax.Array,
    one_hot_sequence: jax.Array,
    noise: jax.Array,
  ) -> jax.Array:
    """Compute the Jacobian for a single protein structure and noise level."""
    jax.debug.print(
      "Shapes: coords {}, mask {}, res_ix {}, chain_ix {}, seq {}, noise {}",
      coords.shape,
      atom_mask.shape,
      residue_ix.shape,
      chain_ix.shape,
      one_hot_sequence.shape,
      noise.shape,
    )

    def logit_fn(one_hot_sequence: jax.Array) -> jax.Array:
      ar_mask = 1 - jnp.eye(one_hot_sequence.shape[0], dtype=jnp.bool_)

      _, logits, _ = score_single_pair(
        jax.random.key(rng_key),
        one_hot_sequence,
        coords,  # type: ignore[arg-type]
        atom_mask,
        residue_ix,
        chain_ix,
        48,
        noise,
        ar_mask,
      )
      return logits

    jax.debug.print("Computing Jacobian for structure of length {}", one_hot_sequence.shape[0])

    if mode == "diagonal":

      def get_logit_at_pos(one_hot_at_pos: jax.Array, pos: int) -> jax.Array:
        """Return logits at `pos` when only `seq[pos]` is changed."""
        modified_sequence = one_hot_sequence.at[pos].set(one_hot_at_pos)
        return logit_fn(modified_sequence)[pos]

      return jax.vmap(
        jax.jacrev(get_logit_at_pos, argnums=0),
        in_axes=(0, 0),
      )(one_hot_sequence, jnp.arange(one_hot_sequence.shape[0]))

    jax.debug.print("Computing full Jacobian.")
    return jax.jacrev(logit_fn)(one_hot_sequence)

  vmap_over_noise = jax.vmap(
    compute_jacobian_for_structure,
    in_axes=(None, None, None, None, None, 0),
    out_axes=0,
  )

  def mapped_fn(
    input_tuple: tuple[
      StructureAtomicCoordinates,
      AtomMask,
      ResidueIndex,
      ChainIndex,
      ProteinSequence,
    ],
  ) -> jax.Array:
    """Map over a single structure, vmapping over noise levels."""
    return vmap_over_noise(*input_tuple, backbone_noise)

  jacobians: jax.Array = jax.lax.map(
    mapped_fn,
    (
      batched_proteins.coordinates,
      batched_proteins.atom_mask,
      batched_proteins.residue_index,
      batched_proteins.chain_index,
      batched_proteins.one_hot_sequence,
    ),
    batch_size=batch_size,
  )

  return {
    "categorical_jacobians": jacobians,
    "jacobian_diffs": jnp.diff(jacobians, axis=1) if jacobians.shape[1] > 1 else None,
    "metadata": {
      "protein_sources": sources,
      "backbone_noise_levels": backbone_noise,
      "computation_mode": mode,
    },
  }
