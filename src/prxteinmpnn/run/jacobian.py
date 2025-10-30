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
  from jaxtyping import Float, Int, PyTree

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
  BackboneDihedrals,
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
  make_encoding_conditional_logits_split_fn,
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
    encode_fn, conditional_logits_fn = make_encoding_conditional_logits_split_fn(
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
) -> Generator[tuple[Any, jax.Array], None, None]:
  """Generate and yield Jacobian batches or encoding batches."""
  for batched_ensemble in protein_iterator:
    if not spec.average_encodings:

      def compute_jacobian_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
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
            dihedrals=dihedrals,
            bias=None,
            k_neighbors=48,
            backbone_noise=noise,
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
        dihedrals: BackboneDihedrals,
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
            dihedrals,
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
        batched_ensemble.dihedrals,
        batched_ensemble.one_hot_sequence,
      )
      yield jacobians_batch, batched_ensemble.one_hot_sequence
    if spec.average_encodings and encode_fn is not None:

      def compute_encodings_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
        noise: Float,
      ) -> tuple[NodeFeatures, EdgeFeatures, NeighborIndices, AlphaCarbonMask, AutoRegressiveMask]:
        return encode_fn(
          jax.random.key(spec.random_seed),
          coords,
          mask,
          residue_ix,
          chain_ix,
          dihedrals,
          48,
          noise,
        )

      def mapped_fn(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        dihedrals: BackboneDihedrals,
      ) -> jax.Array:
        """Compute encodings for a single structure across multiple noise levels."""
        return jax.lax.map(
          partial(compute_encodings_for_structure, coords, mask, residue_ix, chain_ix, dihedrals),
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          batch_size=spec.noise_batch_size,
        )

      encodings_batch = jax.vmap(mapped_fn)(
        batched_ensemble.coordinates,
        batched_ensemble.mask,
        batched_ensemble.residue_index,
        batched_ensemble.chain_index,
        batched_ensemble.dihedrals,
      )
      yield encodings_batch, batched_ensemble.one_hot_sequence


def _get_initial_rolling_average_state(
  initial_encodings: PyTree,
) -> tuple[PyTree, int]:
  """Get the initial state for the rolling average of encodings."""
  initial_avg = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), initial_encodings)
  initial_count = initial_encodings[0].shape[0] * initial_encodings[0].shape[1]
  return initial_avg, initial_count


def _update_rolling_average(
  state: tuple[PyTree, int],
  new_encodings: PyTree,
) -> tuple[PyTree, int]:
  """Update the rolling average of encodings with a new batch."""
  avg_so_far, count_so_far = state
  new_avg = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), new_encodings)
  new_count = new_encodings[0].shape[0] * new_encodings[0].shape[1]
  total_count = count_so_far + new_count

  updated_avg = jax.tree_util.tree_map(
    lambda old, new: (old * count_so_far + new * new_count) / total_count,
    avg_so_far,
    new_avg,
  )
  return updated_avg, total_count


def _categorical_jacobian_in_memory(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,  # noqa: ANN401
  encode_fn: Callable | None,
) -> dict[str, jax.Array | dict[str, JacobianSpecification] | None]:
  """Compute Jacobians and store them in memory."""
  output_generator = _compute_batch_outputs(
    spec,
    protein_iterator,
    conditional_logits_fn,
    encode_fn,
  )

  if spec.average_encodings:
    try:
      first_batch, first_sequence_batch = next(output_generator)
    except StopIteration:
      return {"categorical_jacobians": None, "metadata": None}

    _, count = _get_initial_rolling_average_state(first_batch)
    one_hot_sequence = first_sequence_batch[0]
    # Separate the encodings that will be averaged from those that will be fixed
    (
      initial_node_features,
      initial_edge_features,
      initial_neighbor_indices,
      initial_mask,
      initial_ar_mask,
    ) = first_batch
    encodings_to_average = (initial_node_features, initial_edge_features)

    avg_features, count = _get_initial_rolling_average_state(encodings_to_average)

    neighbor_indices = initial_neighbor_indices[0, 0]
    mask = initial_mask[0, 0]
    ar_mask = initial_ar_mask[0, 0]

    for batch_outputs, _ in output_generator:
      avg_features, count = _update_rolling_average((avg_features, count), batch_outputs[:2])

    node_features, edge_features = avg_features

    def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
      one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
      logits, _, _ = conditional_logits_fn(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
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
    all_sequences = [first_sequence_batch]  # For combine_jacs_fn

  else:
    # --- Path for Standard Jacobian Computation ---
    all_outputs_and_sequences = list(output_generator)
    if not all_outputs_and_sequences:
      return {"categorical_jacobians": None, "metadata": None}

    all_outputs = [item[0] for item in all_outputs_and_sequences]
    all_sequences = [item[1] for item in all_outputs_and_sequences]
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

  if spec.average_encodings:
    if "categorical_jacobians" in group:
      logger.info("Found existing data for averaged encoding, skipping computation.")
      return

    output_generator = _compute_batch_outputs(
      spec,
      protein_iterator,
      conditional_logits_fn,
      encode_fn,
    )
    try:
      first_batch, first_sequence_batch = next(output_generator)
    except StopIteration:
      return

    avg_encodings, count = _get_initial_rolling_average_state(first_batch)
    one_hot_sequence = first_sequence_batch[0]
    # Separate the encodings that will be averaged from those that will be fixed
    (
      initial_node_features,
      initial_edge_features,
      initial_neighbor_indices,
      initial_mask,
      initial_ar_mask,
    ) = first_batch
    encodings_to_average = (initial_node_features, initial_edge_features)

    avg_features, count = _get_initial_rolling_average_state(encodings_to_average)

    neighbor_indices = initial_neighbor_indices[0, 0]
    mask = initial_mask[0, 0]
    ar_mask = initial_ar_mask[0, 0]

    for batch_outputs, _ in output_generator:
      avg_encodings, count = _update_rolling_average((avg_features, count), batch_outputs[:2])

    node_features, edge_features = avg_encodings

    def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
      one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
      logits, _, _ = conditional_logits_fn(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        sequence=one_hot_2d,  # pyright: ignore[reportCallIssue]
      )
      return logits.flatten()

    jacobians = _compute_jacobian_from_logit_fn(
      logit_fn,
      one_hot_sequence,
      spec.jacobian_batch_size,
    )
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
