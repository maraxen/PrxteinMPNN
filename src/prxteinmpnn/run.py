"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

mp.set_start_method("spawn", force=True)

import h5py
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset
  from jaxtyping import ArrayLike

  from prxteinmpnn.mpnn import ModelVersion, ModelWeights
  from prxteinmpnn.utils.catjac import CombineCatJacPairFn
  from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase
  from prxteinmpnn.utils.types import (
    AtomMask,
    ChainIndex,
    ModelParameters,
    OneHotProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from prxteinmpnn.io import loaders
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.sampling.conditional_logits import ConditionalLogitsFn, make_conditional_logits_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.aa_convert import string_to_protein_sequence
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.catjac import combine_jacobians_h5_stream, make_combine_jac
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.residue_constants import atom_order

AlignmentStrategy = Literal["sequence", "structure"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs


@dataclass
class RunSpecification:
  """Configuration for running the model.

  Attributes:
      inputs: A sequence of input file paths or StringIO objects, or a single input.
      model_weights: The model weights to use (default is "original").
      model_version: The model version to use (default is "v_48_020.pkl").
      batch_size: The batch size to use (default is 32).
      backbone_noise: The backbone noise levels to use (default is (0.0,)).
                      Can be a single float or a sequence of floats.
      foldcomp_database: An optional path to a FoldComp database (default is None).
      num_workers: The number of worker processes for data loading (default is 0).
      ar_mask: An optional array-like mask for autoregressive positions (default is None).
      random_seed: The random seed to use (default is 42).
      chain_id: An optional chain ID to use (default is None).
      model: An optional model ID to use (default is None).
      altloc: The alternate location to use (default is "first").

  """

  inputs: Sequence[str | StringIO] | str | StringIO
  model_weights: ModelWeights = "original"
  model_version: ModelVersion = "v_48_020.pkl"
  batch_size: int = 32
  backbone_noise: Sequence[float] | float = (0.0,)
  foldcomp_database: FoldCompDatabase | None = None
  num_workers: int = 0
  ar_mask: None | ArrayLike = None
  random_seed: int = 42
  chain_id: Sequence[str] | str | None = None
  model: int | None = None
  altloc: Literal["first", "all"] = "first"

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    if isinstance(self.backbone_noise, float):
      object.__setattr__(self, "backbone_noise", (self.backbone_noise,))


@dataclass
class ScoringSpecification(RunSpecification):
  """Configuration for scoring sequences.

  Attributes:
      sequences_to_score: A sequence of amino acid sequences to score.
      temperature: The temperature for scoring (default is 1.0).
      return_logits: Whether to return the raw logits (default is False).
      return_decoding_orders: Whether to return decoding orders (default is False).
      return_all_scores: Whether to return scores for all sequences (default is False).
      score_batch_size: The batch size for scoring sequences (default is 16).

  """

  sequences_to_score: Sequence[str] = ()
  temperature: float = 1.0
  return_logits: bool = False
  return_decoding_orders: bool = False
  return_all_scores: bool = False
  score_batch_size: int = 16

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if not self.sequences_to_score:
      msg = (
        "No sequences provided for scoring."
        "`sequences_to_score` must be a non-empty list of strings."
      )
      raise ValueError(msg)


@dataclass
class SamplingSpecification(RunSpecification):
  """Configuration for sampling sequences."""

  num_samples: int = 1
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature"
  temperature: float = 0.1
  bias: ArrayLike | None = None
  fixed_positions: ArrayLike | None = None
  iterations: int | None = None
  learning_rate: float | None = None

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if self.sampling_strategy == "straight_through" and (
      self.iterations is None or self.learning_rate is None
    ):
      msg = "For 'straight_through' sampling, 'iterations' and 'learning_rate' must be provided."
      raise ValueError(msg)


@dataclass
class JacobianSpecification(RunSpecification):
  """Configuration for computing categorical Jacobians."""

  noise_batch_size: int = 1
  jacobian_batch_size: int = 16
  combine: bool = False
  combine_batch_size: int = 8
  combine_weights: ArrayLike | None = None
  combine_fn: CombineCatJacPairFn | Literal["add", "subtract"] = "add"
  combine_fn_kwargs: dict[str, Any] | None = None
  output_h5_path: str | Path | None = None
  compute_apc: bool = True
  combine_results: bool = False

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))


def _prep_protein_stream_and_model(spec: RunSpecification) -> tuple[IterDataset, ModelParameters]:
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
  }
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
    parse_kwargs=parse_kwargs,
    num_workers=spec.num_workers,
  )
  model_parameters = get_mpnn_model(
    model_version=spec.model_version,
    model_weights=spec.model_weights,
  )
  return protein_iterator, model_parameters


def score(
  spec: ScoringSpecification | None = None,
  **kwargs: Any,
) -> dict[str, jax.Array | dict[str, ScoringSpecification] | None]:
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
        num_workers: Number of parallel workers for data loading.
      **kwargs: Additional keyword arguments for structure loading.


  Returns:
      A dictionary containing scores, logits, and metadata.

  """
  if spec is None:
    spec = ScoringSpecification(**kwargs)

  if not spec.sequences_to_score:
    msg = (
      "No sequences provided for scoring. `sequences_to_score` must be a non-empty list of strings."
    )
    raise ValueError(msg)

  integer_sequences = [string_to_protein_sequence(s) for s in spec.sequences_to_score]
  batched_sequences = jnp.concatenate(integer_sequences)

  protein_iterator, model_parameters = _prep_protein_stream_and_model(spec)
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
    scores, logits, decoding_orders = vmap_structures(
      jax.random.key(spec.random_seed),
      batched_sequences,
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
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


def sample(
  spec: SamplingSpecification | None = None,
  **kwargs: Any,
) -> dict[str, jax.Array | dict[str, SamplingSpecification] | None]:
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

  protein_iterator, model_parameters = _prep_protein_stream_and_model(spec)
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


def categorical_jacobian(
  spec: JacobianSpecification | None = None,
  **kwargs: Any,
) -> dict[
  str,
  jax.Array
  | dict[
    str,
    JacobianSpecification,
  ]
  | None,
]:
  """Compute the Jacobian of the model's logits with respect to the input sequence.

  Args:
      spec: An optional JacobianConfig object. If None, a default will be created using
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
        batch_size: The number of structures to process in a single batch.
        noise_batch_size: Batch size for noise levels in Jacobian computation.
        jacobian_batch_size: Inner batch size for Jacobian computation.
        combine_batch_size: Batch size for combining Jacobians.
        num_workers: Number of parallel workers for data loading.
        combine_fn: Function or string specifying how to combine Jacobian pairs (e.g., "add",
        "subtract").
        combine_fn_kwargs: Optional dictionary of keyword arguments for the combine function.
        combine_weights: Optional weights to use when combining Jacobians.
        combine: Whether to combine Jacobians across samples.
        output_h5_path: Optional path to an HDF5 file for streaming output.
        compute_apc: Whether to compute APC-corrected Frobenius norm.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing the Jacobian tensor and metadata.

  """
  if spec is None:
    spec = JacobianSpecification(**kwargs)

  protein_iterator, model_parameters = _prep_protein_stream_and_model(spec)
  conditional_logits_fn = make_conditional_logits_fn(model_parameters=model_parameters)

  if spec.output_h5_path:
    result = _categorical_jacobian_streaming(spec, protein_iterator, conditional_logits_fn)
    if spec.combine_results:
      if not spec.output_h5_path:
        msg = "output_h5_path must be provided for streaming."
        raise ValueError(msg)
      if not spec.combine_weights is not None:
        msg = "combine_weights must be provided for streaming."
        raise ValueError(msg)
      if isinstance(spec.combine_fn, str):
        msg = "combine_fn must be a callable for streaming."
        raise TypeError(msg)
      combine_jacobians_h5_stream(
        h5_path=spec.output_h5_path,
        combine_fn=spec.combine_fn,
        fn_kwargs=spec.combine_fn_kwargs or {},
        batch_size=spec.combine_batch_size,
        weights=jnp.asarray(spec.combine_weights),
      )
    return result
  return _categorical_jacobian_in_memory(spec, protein_iterator, conditional_logits_fn)


def _compute_jacobian_batches(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn,
) -> Generator[tuple[jax.Array, jax.Array], None, None]:
  """Generate and yield Jacobian batches."""
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
          jax.random.key(spec.random_seed),
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
        batch_size=spec.jacobian_batch_size,
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
        jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
        batch_size=spec.noise_batch_size,
      )

    jacobians_batch = jax.vmap(mapped_fn)(
      batched_ensemble.coordinates,
      batched_ensemble.atom_mask,
      batched_ensemble.residue_index,
      batched_ensemble.chain_index,
      batched_ensemble.one_hot_sequence,
    )
    yield jacobians_batch, batched_ensemble.one_hot_sequence


def _categorical_jacobian_in_memory(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,
) -> dict[str, jax.Array | dict[str, JacobianSpecification] | None]:
  """Compute Jacobians and store them in memory."""
  all_jacobians, all_sequences = [], []
  for jacobians_batch, one_hot_sequence_batch in _compute_jacobian_batches(
    spec,
    protein_iterator,
    conditional_logits_fn,
  ):
    all_jacobians.append(jacobians_batch)
    all_sequences.append(one_hot_sequence_batch)

  if not all_jacobians:
    return {"categorical_jacobians": None, "metadata": None}

  jacobians = jnp.concatenate(all_jacobians, axis=0)
  apc_jacobians = (
    jax.vmap(jax.vmap(apc_corrected_frobenius_norm))(jacobians) if spec.compute_apc else None
  )

  combine_jacs_fn = make_combine_jac(
    combine_fn=spec.combine_fn,
    fn_kwargs=spec.combine_fn_kwargs,
    batch_size=spec.combine_batch_size,
  )
  # merge batch and noise dimensions
  reshaped_jacobians = jacobians.reshape((-1, *jacobians.shape[2:]))
  all_sequences_arr = jnp.concatenate(all_sequences, axis=0)
  if not isinstance(spec.backbone_noise, Sequence):
    msg = "backbone_noise must be a sequence."
    raise TypeError(msg)
  tiled_sequences = jnp.tile(
    all_sequences_arr,
    (len(spec.backbone_noise), 1, 1),
  )

  if spec.combine_weights is None and spec.combine:
    spec.combine_weights = jnp.ones((tiled_sequences.shape[0],), dtype=jnp.float32)

  combined_jacs = (
    combine_jacs_fn(
      reshaped_jacobians,
      tiled_sequences,
      jnp.asarray(spec.combine_weights, dtype=jnp.float32),
    )
    if spec.combine
    else None
  )

  return {
    "categorical_jacobians": jacobians,
    "apc_corrected_jacobians": apc_jacobians,
    "combined": combined_jacs,
    "metadata": {
      "spec": spec,
    },
  }


_EXTRA_DIM_JAC = 7  # (1, B, N_noise, L, 21, L, 21)


def _compute_and_write_jacobians_streaming(
  f: h5py.File,
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,
) -> None:
  """Compute Jacobians and stream them to an HDF5 file."""
  jac_ds = f.create_dataset(
    "categorical_jacobians",
    (0, 0, 0, 0, 0, 0),
    maxshape=(None, None, None, None, None, None),
    chunks=True,
  )
  seq_ds = f.create_dataset(
    "one_hot_sequences",
    (0, 0, 0),
    maxshape=(None, None, None),
    chunks=True,
  )
  apc_ds = (
    f.create_dataset(
      "apc_corrected_jacobians",
      (0, 0, 0, 0),
      maxshape=(None, None, None, None),
      chunks=True,
    )
    if spec.compute_apc
    else None
  )
  for jacobians_batch, one_hot_sequence_batch in _compute_jacobian_batches(
    spec,
    protein_iterator,
    conditional_logits_fn,
  ):
    current_size = jac_ds.shape[0]
    new_size = current_size + jacobians_batch.shape[0]

    jac_ds.resize((new_size, *jacobians_batch.shape[1:]))
    seq_ds.resize((new_size, *one_hot_sequence_batch.shape[1:]))

    jac_ds[current_size:new_size] = jacobians_batch
    seq_ds[current_size:new_size] = one_hot_sequence_batch

    if apc_ds:
      apc_jacobians = jax.vmap(jax.vmap(apc_corrected_frobenius_norm))(jacobians_batch)
      apc_ds.resize((new_size, *apc_jacobians.shape[1:]))
      apc_ds[current_size:new_size] = apc_jacobians


def _categorical_jacobian_streaming(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,
) -> dict[str, Any]:
  """Compute Jacobians and stream them to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  with h5py.File(spec.output_h5_path, "w") as f:
    _compute_and_write_jacobians_streaming(f, spec, protein_iterator, conditional_logits_fn)

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {
      "spec": spec,
    },
  }
