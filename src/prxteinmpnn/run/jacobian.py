"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import h5py
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Generator

  from grain.python import IterDataset

  from prxteinmpnn.utils.types import (
    AtomMask,
    ChainIndex,
    OneHotProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from prxteinmpnn.sampling.conditional_logits import ConditionalLogitsFn, make_conditional_logits_fn
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.catjac import (
  combine_jacobians_h5_stream,
  make_combine_jac,
)

from .prep import prep_protein_stream_and_model
from .specs import JacobianSpecification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def categorical_jacobian(
  spec: JacobianSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
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

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)
  conditional_logits_fn = make_conditional_logits_fn(model_parameters=model_parameters)

  if spec.output_h5_path:
    result = _categorical_jacobian_streaming(spec, protein_iterator, conditional_logits_fn)
    if spec.combine:
      if not spec.output_h5_path:
        msg = "output_h5_path must be provided for streaming."
        raise ValueError(msg)
      if not spec.combine_weights is not None:
        msg = "combine_weights must be provided for streaming."
        raise ValueError(msg)

      combine_fn = spec.combine_fn
      combine_jacobians_h5_stream(
        h5_path=spec.output_h5_path,
        combine_fn=combine_fn,  # pyright: ignore[reportArgumentType]
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
  conditional_logits_fn: Any,  # noqa: ANN401
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

  combined_jacs, mapping = (
    combine_jacs_fn(
      jacobians,
      jnp.concatenate(all_sequences, axis=0),
      jnp.asarray(spec.combine_weights, dtype=jnp.float32),
    )
    if spec.combine
    else (None, None)
  )

  return {
    "categorical_jacobians": jacobians,
    "apc_corrected_jacobians": apc_jacobians,
    "combined": combined_jacs,
    "mapping": mapping,
    "metadata": {
      "spec": spec,
    },
  }


def _compute_and_write_jacobians_streaming(
  f: h5py.File,
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn,
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
      apc_func = partial(
        apc_corrected_frobenius_norm,
        residue_batch_size=spec.apc_residue_batch_size,
      )
      apc_jacobians = jax.vmap(
        lambda noise_jac, apc_func=apc_func: jax.lax.map(
          apc_func,
          noise_jac,
          batch_size=spec.apc_batch_size,
        ),
      )(jacobians_batch)
      apc_ds.resize((new_size, *apc_jacobians.shape[1:]))
      apc_ds[current_size:new_size] = apc_jacobians
    f.flush()


def _categorical_jacobian_streaming(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn,
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
