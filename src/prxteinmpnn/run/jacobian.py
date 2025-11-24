"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import itertools
import logging
import sys
from functools import partial
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
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
  make_encoding_conditional_logits_split_fn,
)
from prxteinmpnn.utils.apc import apc_corrected_frobenius_norm
from prxteinmpnn.utils.catjac import (
  make_combine_jac,
)
from prxteinmpnn.utils.sharding import create_mesh, shard_pytree

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

  # Initialize mesh if requested
  mesh = None
  if spec.use_sharding:
    mesh = create_mesh()

  protein_iterator, model = prep_protein_stream_and_model(spec)
  model = eqx.tree_inference(model, value=True)

  if spec.average_encodings:
    encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(
      model=model,
    )
    conditional_logits_fn = None  # Not used in average_encodings path
  else:
    encode_fn = None
    decode_fn = None
    conditional_logits_fn = make_conditional_logits_fn(model=model)

  if spec.output_h5_path:
    spec_hash = sha256(repr(spec).encode()).hexdigest()
    return _categorical_jacobian_streaming(
      spec,
      protein_iterator,
      conditional_logits_fn,
      encode_fn,
      decode_fn,
      spec_hash,
      mesh=mesh,
    )

  return _categorical_jacobian_in_memory(
    spec,
    protein_iterator,
    conditional_logits_fn,
    encode_fn,
    decode_fn,
    mesh=mesh,
  )


def _compute_batch_outputs(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: ConditionalLogitsFn | None,
  encode_fn: Callable | None,
  mesh: jax.sharding.Mesh | None = None,
) -> Generator[tuple[Any, jax.Array], None, None]:
  """Generate and yield Jacobian batches or encoding batches."""
  for batched_ensemble in protein_iterator:
    if mesh is not None and spec.shard_batch:
      batched_ensemble = shard_pytree(batched_ensemble, mesh)  # noqa: PLW2901

    if not spec.average_encodings:
      if conditional_logits_fn is None:
        msg = "conditional_logits_fn must be provided when not using average_encodings"
        raise ValueError(msg)

      def compute_jacobian_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        one_hot_sequence: OneHotProteinSequence,
        noise: Float,
        struct_mapping: jax.Array | None,
      ) -> Logits:
        def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
          one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)
          logits = conditional_logits_fn(
            jax.random.key(spec.random_seed),
            coords,
            mask,
            residue_ix,
            chain_ix,
            one_hot_2d,
            None,  # ar_mask
            noise,  # backbone_noise
            struct_mapping,  # structure_mapping
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
        struct_mapping: jax.Array | None,
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
            struct_mapping=struct_mapping,
          ),
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          batch_size=spec.noise_batch_size,
        )

      def _compute_jacobians(ensemble):
        return jax.vmap(mapped_fn1)(
          ensemble.coordinates,
          ensemble.mask,
          ensemble.residue_index,
          ensemble.chain_index,
          ensemble.one_hot_sequence,
          ensemble.mapping,
        )

      if mesh is not None:
        with mesh:
          jacobians_batch = _compute_jacobians(batched_ensemble)
      else:
        jacobians_batch = _compute_jacobians(batched_ensemble)

      yield jacobians_batch, batched_ensemble.one_hot_sequence
    if spec.average_encodings and encode_fn is not None:

      def compute_encodings_for_structure(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        noise: Float,
        struct_mapping: jax.Array | None,
      ) -> tuple[NodeFeatures, EdgeFeatures, NeighborIndices, AlphaCarbonMask, AutoRegressiveMask]:
        return encode_fn(
          coords,
          mask,
          residue_ix,
          chain_ix,
          backbone_noise=noise,
          structure_mapping=struct_mapping,
        )

      def mapped_fn(
        coords: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_ix: ResidueIndex,
        chain_ix: ChainIndex,
        struct_mapping: jax.Array | None,
      ) -> jax.Array:
        """Compute encodings for a single structure across multiple noise levels."""
        return jax.lax.map(
          partial(
            compute_encodings_for_structure,
            coords,
            mask,
            residue_ix,
            chain_ix,
            struct_mapping=struct_mapping,
          ),
          jnp.asarray(spec.backbone_noise, dtype=jnp.float32),
          batch_size=spec.noise_batch_size,
        )

      def _compute_encodings(ensemble):
        return jax.vmap(mapped_fn)(
          ensemble.coordinates,
          ensemble.mask,
          ensemble.residue_index,
          ensemble.chain_index,
          ensemble.mapping,
        )

      if mesh is not None:
        with mesh:
          encodings_batch = _compute_encodings(batched_ensemble)
      else:
        encodings_batch = _compute_encodings(batched_ensemble)
      yield encodings_batch, batched_ensemble.one_hot_sequence


def _get_initial_rolling_average_state(
  initial_encodings: tuple,
) -> tuple[list, list, list, list, list]:
  """Get the initial state for the rolling average of encodings."""
  node_features, edge_features, neighbor_indices, mask, ar_mask = initial_encodings

  # Flatten batch and noise dimensions
  flat_node = node_features.reshape((-1, *node_features.shape[2:]))
  flat_edge = edge_features.reshape((-1, *edge_features.shape[2:]))
  flat_neighbors = neighbor_indices.reshape((-1, *neighbor_indices.shape[2:]))
  flat_mask = mask.reshape((-1, *mask.shape[2:]))
  flat_ar_mask = ar_mask.reshape((-1, *ar_mask.shape[2:]))

  return (
    [flat_node],
    [flat_edge],
    [flat_neighbors],
    [flat_mask],
    [flat_ar_mask],
  )


def _update_rolling_average(
  state: tuple[list, list, list, list, list],
  new_encodings: tuple,
) -> tuple[list, list, list, list, list]:
  """Update the rolling average of encodings with a new batch."""
  nodes, edges, neighbors_list, masks, ar_masks = state
  node_features, edge_features, neighbor_indices, mask, ar_mask = new_encodings

  # Flatten and append
  nodes.append(node_features.reshape((-1, *node_features.shape[2:])))
  edges.append(edge_features.reshape((-1, *edge_features.shape[2:])))
  neighbors_list.append(neighbor_indices.reshape((-1, *neighbor_indices.shape[2:])))
  masks.append(mask.reshape((-1, *mask.shape[2:])))
  ar_masks.append(ar_mask.reshape((-1, *ar_mask.shape[2:])))

  return nodes, edges, neighbors_list, masks, ar_masks


def _pad_and_concatenate_features(
  nodes_list: list[jax.Array],
  edges_list: list[jax.Array],
  neighbors_list: list[jax.Array],
  masks_list: list[jax.Array],
  ar_masks_list: list[jax.Array],
  sequences_list: list[jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Pad and concatenate feature lists along the batch dimension."""
  max_len = max(x.shape[1] for x in nodes_list)

  def _pad_list(arrays: list[jax.Array], pad_dims: tuple[int, ...]) -> jax.Array:
    padded = []
    for arr in arrays:
      curr_len = arr.shape[1]
      if curr_len < max_len:
        pad_width = [(0, 0)] * arr.ndim
        for dim in pad_dims:
          pad_width[dim] = (0, max_len - curr_len)
        padded.append(jnp.pad(arr, pad_width))
      else:
        padded.append(arr)
    return jnp.concatenate(padded, axis=0)

  return (
    _pad_list(nodes_list, (1,)),
    _pad_list(edges_list, (1,)),
    _pad_list(neighbors_list, (1,)),
    _pad_list(masks_list, (1,)),
    _pad_list(ar_masks_list, (1, 2)),
    _pad_list(sequences_list, (1,)),
  )


def _categorical_jacobian_in_memory(
  spec: JacobianSpecification,
  protein_iterator: IterDataset,
  conditional_logits_fn: Any,  # noqa: ANN401
  encode_fn: Callable | None,
  decode_fn: Callable | None,
  mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array | dict[str, JacobianSpecification] | None]:
  """Compute Jacobians and store them in memory."""
  output_generator = _compute_batch_outputs(
    spec,
    protein_iterator,
    conditional_logits_fn,
    encode_fn,
    mesh=mesh,
  )

  if spec.average_encodings:
    if encode_fn is None or decode_fn is None:
      msg = "encode_fn and decode_fn must be provided for average_encodings mode"
      raise ValueError(msg)

    all_encodings_and_sequences = list(output_generator)
    if not all_encodings_and_sequences:
      return {"categorical_jacobians": None, "metadata": None}

    all_encodings = [item[0] for item in all_encodings_and_sequences]
    all_sequences = [item[1] for item in all_encodings_and_sequences]

    nodes_list, edges_list, neighbors_list, masks_list, ar_masks_list = (
      _get_initial_rolling_average_state(all_encodings[0])
    )

    for encodings_batch in all_encodings[1:]:
      nodes_list, edges_list, neighbors_list, masks_list, ar_masks_list = _update_rolling_average(
        (nodes_list, edges_list, neighbors_list, masks_list, ar_masks_list),
        encodings_batch,
      )

    (
      all_nodes,
      all_edges,
      all_neighbors,
      all_mask,
      all_ar_mask,
      all_sequences_concat,
    ) = _pad_and_concatenate_features(
      nodes_list,
      edges_list,
      neighbors_list,
      masks_list,
      ar_masks_list,
      all_sequences,
    )

    # Compute averaged node and edge features using mask
    mask_expanded = all_mask[..., None]
    mask_sum = jnp.sum(mask_expanded, axis=0)
    mask_sum = jnp.maximum(mask_sum, 1e-8)

    averaged_node = jnp.sum(all_nodes * mask_expanded, axis=0) / mask_sum

    mask_expanded_edge = all_mask[..., None, None]
    averaged_edge = jnp.sum(all_edges * mask_expanded_edge, axis=0) / mask_sum[..., None]

    def compute_jacobian_for_sequence(
      one_hot_sequence: OneHotProteinSequence,
    ) -> jax.Array:
      """Compute jacobian for a single sequence using averaged encodings."""
      # one_hot_sequence is (L_max, 21) (from all_sequences_concat)

      def logit_fn(one_hot_flat: jax.Array) -> jax.Array:
        one_hot_2d = one_hot_flat.reshape(one_hot_sequence.shape)

        def decode_single(
          n_idx: jax.Array,
          m: jax.Array,
          ar_m: jax.Array,
        ) -> jax.Array:
          return decode_fn(
            (averaged_node, averaged_edge, n_idx, m, ar_m),
            one_hot_2d,
            None,
          )

        logits_batch = jax.vmap(decode_single)(all_neighbors, all_mask, all_ar_mask)
        logits = jnp.mean(logits_batch, axis=0)
        return logits.flatten()

      return _compute_jacobian_from_logit_fn(
        logit_fn,
        one_hot_sequence,
        spec.jacobian_batch_size,
      )

    jacobians = jax.vmap(compute_jacobian_for_sequence)(all_sequences_concat)
    jacobians = jacobians[:, None, :, :, :, :]  # Add noise dimension
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
        all_sequences_concat,
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
  conditional_logits_fn: ConditionalLogitsFn | None,
  encode_fn: Callable | None,
  decode_fn: Callable | None,
  spec_hash: str,
  mesh: jax.sharding.Mesh | None = None,
) -> None:
  """Compute Jacobians and stream them to a group in an HDF5 file."""
  group = f.require_group(spec_hash)

  if spec.average_encodings and (encode_fn is None or decode_fn is None):
    msg = "encode_fn and decode_fn must be provided for average_encodings mode"
    raise ValueError(msg)

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
    mesh=mesh,
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
  conditional_logits_fn: ConditionalLogitsFn | None,
  encode_fn: Callable | None,
  decode_fn: Callable | None,
  spec_hash: str,
  mesh: jax.sharding.Mesh | None = None,
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
      decode_fn,
      spec_hash,
      mesh=mesh,
    )

  return {
    "output_h5_path": str(spec.output_h5_path),
    "spec_hash": spec_hash,
    "metadata": {
      "spec": spec,
    },
  }
