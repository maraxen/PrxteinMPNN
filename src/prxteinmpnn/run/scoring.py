"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from typing import Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence

from .prep import prep_protein_stream_and_model
from .specs import ScoringSpecification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def score(
  spec: ScoringSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Score all provided sequences against all input structures.

  This function uses a high-performance Grain pipeline to load and process
  structures, then scores all provided sequences against each structure.

  Args:
      spec: An optional ScoringSpecification object. If None, a default will be created using
      kwargs, options are provided as keyword arguments. The following options can be set:
        inputs: A single or sequence of inputs (files, PDB IDs, etc.).
        sequences_to_score: A list of protein sequences (strings) to score.
        chain_id: Specific chain(s) to parse from the structure.
        model: The model number to load. If None, all models are loaded.
        altloc: The alternate location identifier to use.
        model_version: The model version to use.
        model_weights: The model weights to use.
        foldcomp_database: The FoldComp database to use for FoldComp IDs.
        random_seed: The random number generator key.
        backbone_noise: The amount of noise to add to the backbone.
        ar_mask: An optional array of shape (L, L) to mask out certain residue pairs.
        batch_size: The number of structures to process in a single batch.
      **kwargs: Additional keyword arguments for structure loading.


  Returns:
      A dictionary containing scores, logits, and metadata.

  """
  if spec is None:
    spec = ScoringSpecification(**kwargs)

  if spec.output_h5_path:
    return _score_streaming(spec)

  if not spec.sequences_to_score:
    msg = (
      "No sequences provided for scoring. `sequences_to_score` must be a non-empty list of strings."
    )
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.concatenate(integer_sequences)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)
  score_single_pair = make_score_sequence(model_parameters=model_parameters)

  all_scores, all_logits = [], []

  for batched_ensemble in protein_iterator:
    max_len = batched_ensemble.coordinates.shape[1]
    current_ar_mask = (
      1 - jnp.eye(max_len, dtype=jnp.bool_) if spec.ar_mask is None else jnp.asarray(spec.ar_mask)
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
    scores, logits, _decoding_orders = vmap_structures(
      jax.random.key(spec.random_seed),
      batched_sequences,
      batched_ensemble.coordinates,
      batched_ensemble.mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      48,
      jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
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
      "specification": spec,
    },
  }


def _score_streaming(
  spec: ScoringSpecification,
) -> dict[str, str | dict[str, ScoringSpecification]]:
  """Score sequences and stream results to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.concatenate(integer_sequences)
  if batched_sequences.ndim == 1:
    batched_sequences = jnp.expand_dims(batched_sequences, 0)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)
  score_single_pair = make_score_sequence(model_parameters=model_parameters)

  with h5py.File(spec.output_h5_path, "w") as f:
    scores_ds = f.create_dataset(
      "scores",
      (0,),
      maxshape=(None,),
      chunks=True,
      dtype="f4",
    )
    logits_ds = f.create_dataset(
      "logits",
      (0, 0, 0),
      maxshape=(None, None, None),
      chunks=True,
      dtype="f4",
    )

    for batched_ensemble in protein_iterator:
      max_len = batched_ensemble.coordinates.shape[1]
      current_ar_mask = (
        1 - jnp.eye(max_len, dtype=jnp.bool_) if spec.ar_mask is None else jnp.asarray(spec.ar_mask)
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
        jax.random.key(spec.random_seed),
        batched_sequences,
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        48,
        jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
        current_ar_mask,
      )

      scores_ds.resize(scores_ds.shape[0] + scores.size, axis=0)
      scores_ds[-scores.size :] = scores.flatten()

      logits_ds.resize(logits_ds.shape[0] + logits.shape[0], axis=0)
      logits_ds[-logits.shape[0] :, :, :] = logits

      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "specification": spec,
    },
  }
