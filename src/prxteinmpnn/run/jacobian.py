"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import itertools
import logging
import sys
from functools import partial
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

import h5py
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Callable, Generator

  from grain.python import IterDataset
  from jaxtyping import Float, Int

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    ChainIndex,
    EdgeFeatures,
    Logits,
    NeighborIndices,
    NodeFeatures,
    OneHotProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from prxteinmpnn.sampling.conditional_logits import (
  ConditionalLogitsFn,
  make_conditional_logits_fn,
  make_encoding_average_conditional_logits_fn,
)
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.catjac import (
  make_combine_jac,
)

from .prep import prep_protein_stream_and_model
from .specs import JacobianSpecification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def _compute_jacobian_from_logit_fn(
  logit_fn: Callable[[jax.Array], jax.Array],
  one_hot_sequence: OneHotProteinSequence,
  jacobian_batch_size: Int | None,
) -> jax.Array:
  """Compute the Jacobian of a logit function w.r.t. a one-hot sequence."""
  length = one_hot_sequence.shape[0]
  one_hot_flat = one_hot_sequence.flatten()
  input_dim = one_hot_flat.shape[0]

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
      kwargs, options are provided as keyword arguments. See JacobianSpecification for details.
      **kwargs: Additional keyword arguments for structure loading.

  Returns:
      A dictionary containing the Jacobian tensor and metadata.

  """
  if spec is None:
    spec = JacobianSpecification(**kwargs)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)
  if spec.average_encodings:
    encode_fn, conditional_logits_fn = make_encoding_average_conditional_logits_fn(
      model_parameters=model_parameters,
    )
  else:
    encode_fn = None
    conditional_logits_fn = make_conditional_logits_fn(model_parameters=model_parameters)

  if spec.output_h5_path:
    spec_hash = sha256(repr(spec).encode()).hexdigest()
    return _categorical_jacobian_streaming(
      spec,
      protein_iterator,
      conditional_logits_fn,
      encode_fn,
      spec_hash,
    )

  return _categorical_jacobian_in_memory(spec, protein_iterator, conditional_logits_fn, encode_fn)


def _compute_batch_outputs(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn,
  encode_fn: Callable | None,
) -> Generator[tuple[jax.Array, jax.Array], None, None]:
  """Generate and yield Jacobian batches or encoding batches."""
  for batched_ensemble in protein_iterator:
    if not spec.average_encodings:

      def compute_jacobian_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        one_hot_sequence: OneHotProteinSequence,
        noise: Float,
      ) -> Logits:
        def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
          one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
          logits, _, _ = conditional_logits_fn(
            jax.random.key(spec.random_seed),
            coords,
            one_hot_2d,
            mask,
            residue_ix,
            chain_ix,
            None,
            48,
            noise,
          )
          return logits.flatten()

        return _compute_jacobian_from_logit_fn(
          logit_fn,
          one_hot_sequence,
          spec.jacobian_batch_size,
        )

      def mapped_fn1(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        one_hot_sequence: OneHotProteinSequence,
      ) -> jax.Array:
        """Compute Jacobians for a single structure across multiple noise levels."""
        return jax.lax.map(
          partial(
            compute_jacobian_for_structure,
            coords,
            mask,
            residue_ix,
            chain_ix,
            one_hot_sequence,
          ),
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          batch_size=spec.noise_batch_size,
        )

      jacobians_batch = jax.vmap(mapped_fn1)(
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        batched_ensemble.one_hot_sequence,
      )
      yield jacobians_batch, batched_ensemble.one_hot_sequence
    if spec.average_encodings and encode_fn is not None:

      def compute_encodings_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        noise: Float,
      ) -> tuple[NodeFeatures, EdgeFeatures, NeighborIndices, AlphaCarbonMask, AutoRegressiveMask]:
        return encode_fn(
          jax.random.key(spec.random_seed),
          coords,
          mask,
          residue_ix,
          chain_ix,
          48,
          noise,
        )

      def mapped_fn(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
      ) -> jax.Array:
        """Compute encodings for a single structure across multiple noise levels."""
        return jax.lax.map(
          partial(compute_encodings_for_structure, coords, mask, residue_ix, chain_ix),
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          batch_size=spec.noise_batch_size,
        )

      encodings_batch = jax.vmap(mapped_fn)(
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
      )
      yield encodings_batch, batched_ensemble.one_hot_sequence


def _categorical_jacobian_in_memory(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,  # noqa: ANN401
  encode_fn: Callable | None,
) -> dict[str, jax.Array | dict[str, JacobianSpecification] | None]:
  """Compute Jacobians and store them in memory."""
  all_outputs_and_sequences = list(
    _compute_batch_outputs(spec, protein_iterator, conditional_logits_fn, encode_fn),
  )

  if not all_outputs_and_sequences:
    return {"categorical_jacobians": None, "metadata": None}

  all_outputs = [item[0] for item in all_outputs_and_sequences]
  all_sequences = [item[1] for item in all_outputs_and_sequences]

  if spec.average_encodings:
    # --- Path for Averaged Encodings ---
    # `all_outputs` is a list of pytrees. Concatenate them along the batch axis.
    concatenated_outputs = jax.tree_util.tree_map(
      lambda *xs: jnp.concatenate(xs, axis=0),
      *all_outputs,
    )
    # Average across the batch (axis=0) and noise (axis=1) dimensions.
    avg_encodings = jax.tree_util.tree_map(
      lambda x: jnp.mean(x, axis=(0, 1)),
      concatenated_outputs,
    )
    one_hot_sequence = all_sequences[0][0]  # Use the first sequence

    def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
      one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
      logits, _, _ = conditional_logits_fn(
        *avg_encodings,
        sequence=one_hot_2d,
      )
      return logits.flatten()

    jacobians = _compute_jacobian_from_logit_fn(
      logit_fn,
      one_hot_sequence,
      spec.jacobian_batch_size,
    )
    # Add batch and noise dimensions for consistency
    jacobians = jnp.expand_dims(jnp.expand_dims(jacobians, axis=0), axis=0)

  else:
    # --- Path for Standard Jacobian Computation ---
    jacobians = jnp.concatenate(all_outputs, axis=0)

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
  encode_fn: Callable | None,
  spec_hash: str,
) -> None:
  """Compute Jacobians and stream them to a group in an HDF5 file."""
  group = f.require_group(spec_hash)

  # --- Path for Averaged Encodings ---
  if spec.average_encodings:
    if "categorical_jacobians" in group:
      # If data exists, we assume it's complete for the averaging case, as it's not resumable.
      logger.info("Found existing data for averaged encoding, skipping computation.")
      return

    # Collect all encodings in memory before writing the single result.
    all_outputs_and_sequences = list(
      _compute_batch_outputs(spec, protein_iterator, conditional_logits_fn, encode_fn),
    )
    if not all_outputs_and_sequences:
      return

    all_outputs = [item[0] for item in all_outputs_and_sequences]
    all_sequences = [item[1] for item in all_outputs_and_sequences]

    concatenated_outputs = jax.tree_util.tree_map(
      lambda *xs: jnp.concatenate(xs, axis=0),
      *all_outputs,
    )
    avg_encodings = jax.tree_util.tree_map(
      lambda x: jnp.mean(x, axis=(0, 1)),
      concatenated_outputs,
    )
    one_hot_sequence = all_sequences[0][0]

    def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
      one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
      logits, _, _ = conditional_logits_fn(*avg_encodings, sequence=one_hot_2d)  # pyright: ignore[reportCallIssue]
      return logits.flatten()

    jacobians = _compute_jacobian_from_logit_fn(
      logit_fn,
      one_hot_sequence,
      spec.jacobian_batch_size,
    )
    jacobians = jnp.expand_dims(jnp.expand_dims(jacobians, axis=0), axis=0)

    group.create_dataset("categorical_jacobians", data=jacobians)
    group.create_dataset(
      "one_hot_sequences",
      data=jnp.expand_dims(one_hot_sequence, axis=0),
    )

    if spec.compute_apc:
      apc_jacobians = jax.vmap(jax.vmap(apc_corrected_frobenius_norm))(jacobians)
      group.create_dataset("apc_corrected_jacobians", data=apc_jacobians)
    return

  # --- Path for Standard (Non-Averaging) Streaming ---
  jac_ds = group.require_dataset(
    "categorical_jacobians",
    shape=(0, 0, 0, 0, 0, 0),
    maxshape=(None, None, None, None, None, None),
    chunks=True,
    dtype=jnp.float32,
  )
  start_index = jac_ds.shape[0]
  if start_index > 0:
    msg = f"Resuming computation from index {start_index} in {spec.output_h5_path}."
    logger.info(msg)

  seq_ds = group.require_dataset(
    "one_hot_sequences",
    shape=(0, 0, 0),
    maxshape=(None, None, None),
    chunks=True,
    dtype=jnp.float32,
  )

  apc_ds = (
    group.require_dataset(
      "apc_corrected_jacobians",
      (0, 0, 0, 0),
      maxshape=(None, None, None, None),
      chunks=True,
      dtype=jnp.float32,
    )
    if spec.compute_apc
    else None
  )

  resumable_iterator = cast(
    "IterDataset",
    itertools.islice(protein_iterator, start_index, None),
  )

  for batch_output, one_hot_sequence_batch in _compute_batch_outputs(
    spec,
    resumable_iterator,
    conditional_logits_fn,
    encode_fn,
  ):
    current_size = jac_ds.shape[0]
    new_size = current_size + batch_output.shape[0]

    jac_ds.resize((new_size, *batch_output.shape[1:]))
    seq_ds.resize((new_size, *one_hot_sequence_batch.shape[1:]))

    jac_ds[current_size:new_size] = batch_output
    seq_ds[current_size:new_size] = one_hot_sequence_batch

    if apc_ds is not None and spec.compute_apc:
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
      )(batch_output)
      apc_ds.resize((new_size, *apc_jacobians.shape[1:]))
      apc_ds[current_size:new_size] = apc_jacobians
    f.flush()


def _categorical_jacobian_streaming(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn,
  encode_fn: Callable | None,
  spec_hash: str,
) -> dict[str, Any]:
  """Compute Jacobians and stream them to a group in an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    raise ValueError(msg)

  with h5py.File(spec.output_h5_path, "a") as f:
    _compute_and_write_jacobians_streaming(
      f,
      spec,
      protein_iterator,
      conditional_logits_fn,
      encode_fn,
      spec_hash,
    )

  return {
    "output_h5_path": str(spec.output_h5_path),
    "spec_hash": spec_hash,
    "metadata": {
      "spec": spec,
    },
  }
