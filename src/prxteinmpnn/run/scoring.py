"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.run.averaging import get_averaged_encodings
from prxteinmpnn.scoring.score import make_score_sequence, score_sequence_with_encoding
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence

from .prep import prep_protein_stream_and_model
from .specs import SamplingSpecification, ScoringSpecification

if TYPE_CHECKING:
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.utils.data_structures import Logits, Protein, ProteinSequence

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
  batched_sequences = jnp.array(integer_sequences)

  protein_iterator, model = prep_protein_stream_and_model(spec)

  if spec.average_node_features:
      spec_dict = asdict(spec)
      sampling_fields = {f.name for f in fields(SamplingSpecification)}
      filtered_spec = {k: v for k, v in spec_dict.items() if k in sampling_fields}
      sampling_spec = SamplingSpecification(**filtered_spec)
      all_scores, all_logits = [], []
      for batched_ensemble in protein_iterator:
          scores, logits = _score_batch_averaged(
              sampling_spec,
              batched_ensemble,
              model,
              batched_sequences,
          )
          all_scores.append(scores)
          all_logits.append(logits)
  else:
    score_single_pair = make_score_sequence(model=model)
    all_scores, all_logits = [], []

    for batched_ensemble in protein_iterator:
        max_len = batched_ensemble.coordinates.shape[1]
        if spec.ar_mask is None:
            current_ar_mask = 1 - jnp.eye(max_len, dtype=jnp.bool_)
        else:
            current_ar_mask = jnp.asarray(spec.ar_mask)

        vmap_sequences = jax.vmap(
            score_single_pair,
            in_axes=(None, 0, None, None, None, None, None, None, None, None),
            out_axes=0,
        )
        vmap_noises = jax.vmap(
            vmap_sequences,
            in_axes=(None, None, None, None, None, None, None, 0, None, None),
            out_axes=0,
        )
        vmap_structures = jax.vmap(
            vmap_noises,
            in_axes=(None, None, 0, 0, 0, 0, None, None, None, None),
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
            batched_ensemble.mapping,
        )
        all_scores.append(scores)
        all_logits.append(logits)

  if not all_scores:
    return {}

  return {
    "scores": jnp.concatenate(all_scores, axis=0),
    "logits": jnp.concatenate(all_logits, axis=0),
    "metadata": {
      "specification": spec,
    },
  }


def _score_batch_averaged(
    spec: SamplingSpecification,
    batched_ensemble: Protein,
    model: PrxteinMPNN,
    sequences: ProteinSequence,
) -> tuple[jnp.ndarray, Logits]:
    """Score sequences for a batched ensemble of proteins using averaged encodings."""
    averaged_encodings = get_averaged_encodings(
        batched_ensemble,
        model,
        spec.backbone_noise,
        spec.noise_batch_size,
        spec.random_seed,
        spec.average_encoding_mode,
    )

    def score_single_sequence(seq: ProteinSequence, enc: tuple) -> tuple[jnp.ndarray, Logits]:
        avg_node, avg_edge, neighbors, mask, ar_mask_struct = enc

        # neighbors has shape (..., L, K). avg_node has shape (L, D).
        # Flatten batch dimensions of neighbors
        neighbors_flat = neighbors.reshape(
            (-1, neighbors.shape[-2], neighbors.shape[-1]),
        )
        mask_flat = mask.reshape((-1, mask.shape[-1]))
        ar_mask_struct_flat = ar_mask_struct.reshape(
            (-1, ar_mask_struct.shape[-2], ar_mask_struct.shape[-1]),
        )

        def score_one(
            n_idx: jnp.ndarray, m: jnp.ndarray, ar_m: jnp.ndarray,
        ) -> tuple[jnp.ndarray, Logits, jnp.ndarray]:
             # score_sequence_with_encoding returns (score, logits, decoding_order)
             return score_sequence_with_encoding(
                 model,
                 seq,
                 (avg_node, avg_edge, n_idx, m, ar_m),
             )

        scores_batch, logits_batch, _ = jax.vmap(score_one)(
            neighbors_flat, mask_flat, ar_mask_struct_flat,
        )
        return jnp.mean(scores_batch, axis=0), jnp.mean(logits_batch, axis=0)

    if spec.average_encoding_mode == "inputs_and_noise":
        vmap_score = jax.vmap(score_single_sequence, in_axes=(0, None))
        scores, logits = vmap_score(sequences, averaged_encodings)
        scores = jnp.expand_dims(scores, axis=0)
        logits = jnp.expand_dims(logits, axis=0)
    else:
        # Determine structural axis to map over (the one NOT averaged)
        struct_axis = 1 if spec.average_encoding_mode == "inputs" else 0

        # vmap over sequences (axis 0)
        # vmap over outer batch (axis 0 of enc_node, axis struct_axis of enc_neighbors)

        vmap_score = jax.vmap(
            jax.vmap(score_single_sequence, in_axes=(0, None)),
            in_axes=(None, (0, 0, struct_axis, struct_axis, struct_axis)),
        )
        scores, logits = vmap_score(sequences, averaged_encodings)

    return scores, logits


def _score_streaming(
  spec: ScoringSpecification,
) -> dict[str, str | dict[str, ScoringSpecification]]:
  """Score sequences and stream results to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.array(integer_sequences)
  if batched_sequences.ndim == 1:
    batched_sequences = jnp.expand_dims(batched_sequences, 0)

  protein_iterator, model = prep_protein_stream_and_model(spec)
  score_single_pair = make_score_sequence(model=model)

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
      max_len = batched_ensemble.coordinates.shape[0]
      if spec.ar_mask is None:
        current_ar_mask = 1 - jnp.eye(max_len, dtype=jnp.bool_)
      else:
        current_ar_mask = jnp.asarray(spec.ar_mask)

      vmap_sequences = jax.vmap(
        score_single_pair,
        in_axes=(None, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
      )
      vmap_noises = jax.vmap(
        vmap_sequences,
        in_axes=(None, None, None, None, None, None, None, 0, None, None),
        out_axes=0,
      )
      vmap_structures = jax.vmap(
        vmap_noises,
        in_axes=(None, None, 0, 0, 0, 0, None, None, None, None),
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
        batched_ensemble.mapping,
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
