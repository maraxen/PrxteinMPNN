"""Conformational-inference from ProteinMPNN logits or features."""

import logging
import sys
from collections.abc import Callable, Generator
from typing import Any, cast

import h5py
import jax
import jax.numpy as jnp
from grain.python import IterDataset

from prxteinmpnn.ensemble.ci import infer_states
from prxteinmpnn.ensemble.dbscan import (
  ConformationalStates,
)
from prxteinmpnn.ensemble.gmm import make_fit_gmm
from prxteinmpnn.run.prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import ConformationalInferenceSpecification
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils.types import (
  BackboneAtomCoordinates,
  EdgeFeatures,
  Logits,
  ModelParameters,
  NodeFeatures,
  StructureAtomicCoordinates,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def derive_states(
  spec: ConformationalInferenceSpecification | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Derive conformational states from a protein ensemble using a specified model.

  This function computes logits, node features, and edge features for each structure
  in the provided dataset. It supports both in-memory computation and streaming
  results to an HDF5 file for large datasets.

  Args:
    spec: A ConformationalInferenceSpecification object configuring the process.
    **kwargs: Keyword arguments to create a ConformationalInferenceSpecification if not provided.

  Returns:
    A dictionary containing the computed states or a path to the output file.

  """
  if spec is None:
    spec = ConformationalInferenceSpecification(**kwargs)

  protein_iterator, model_parameters = prep_protein_stream_and_model(spec)

  if spec.output_h5_path:
    return _derive_states_streaming(spec, protein_iterator, model_parameters)
  return _derive_states_in_memory(spec, protein_iterator, model_parameters)


def _get_logits_fn(
  spec: ConformationalInferenceSpecification,
  model_parameters: ModelParameters,
) -> tuple[Callable, bool]:
  """Select and partially configure the appropriate logits function."""
  match spec.inference_strategy:
    case "conditional":
      logits_fn = make_conditional_logits_fn(
        model_parameters=model_parameters,
        decoding_order_fn=spec.decoding_order_fn,
      )
      return logits_fn, True
    case "unconditional":
      logits_fn = make_unconditional_logits_fn(
        model_parameters=model_parameters,
        decoding_order_fn=spec.decoding_order_fn,
      )
      return logits_fn, False
    case "vmm":
      msg = "VMM inference strategy is not yet implemented."
      logger.error(msg)
      raise NotImplementedError(msg)
    case "coordinates":
      return lambda *args, **kwargs: (  # type: ignore[return]  # noqa: ARG005
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
      ), False
    case _:
      msg = f"Invalid inference strategy: {spec.inference_strategy}"
      logger.error(msg)
      raise ValueError(msg)


def _compute_states_batches(
  spec: ConformationalInferenceSpecification,
  protein_iterator: IterDataset,
  model_parameters: ModelParameters,
) -> Generator[
  tuple[
    Logits | None,
    NodeFeatures | None,
    EdgeFeatures | None,
    BackboneAtomCoordinates | None,
    StructureAtomicCoordinates | None,
  ],
  None,
  None,
]:
  """Generate and yield batches of computed residue states using in_axes for vmap.

  Args:
    spec: ConformationalInferenceSpecification, configuration for inference.
    protein_iterator: IterDataset, yields batches of protein ensembles.
    model_parameters: ModelParameters, parameters for the model.

  Returns:
    Generator yielding tuples of (logits, node_features, edge_features) for each batch.


  """
  get_logits, is_conditional = _get_logits_fn(spec, model_parameters)
  static_args = (None, 48, None)

  logger.info("Iterating through frames/proteins...")
  for batched_ensemble in protein_iterator:
    n_frames = batched_ensemble.coordinates.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(spec.random_seed), n_frames)

    if is_conditional:
      batch_states = jax.vmap(
        get_logits,
        in_axes=(
          0,
          0,
          None,
          None,
          None,
          None,
          None,
        ),
      )(
        keys,
        batched_ensemble.coordinates,
        batched_ensemble.one_hot_sequence[0],
        batched_ensemble.atom_mask[:, 0],
        batched_ensemble.residue_index[0],
        batched_ensemble.chain_index[0],
        *static_args,
      )
    else:
      batch_states = jax.vmap(
        get_logits,
        in_axes=(
          0,
          0,
          None,
          None,
          None,
          None,
          None,
          None,
        ),
      )(
        keys,
        batched_ensemble.coordinates,
        batched_ensemble.atom_mask[0, :, 0],
        batched_ensemble.residue_index[0],
        batched_ensemble.chain_index[0],
        *static_args,
      )

    logits, node_features, edge_features = batch_states
    yield (
      logits if "logits" in spec.inference_features else None,
      node_features if "node_features" in spec.inference_features else None,
      edge_features if "edge_features" in spec.inference_features else None,
      batched_ensemble.coordinates if "backbone_coordinates" in spec.inference_features else None,
      batched_ensemble.full_coordinates if "full_coordinates" in spec.inference_features else None,
    )


def _derive_states_in_memory(
  spec: ConformationalInferenceSpecification,
  protein_iterator: IterDataset,
  model_parameters: ModelParameters,
) -> dict[str, jax.Array | dict[str, ConformationalInferenceSpecification] | None]:
  """Compute global states and stores them in memory."""
  all_states = [
    (logits, node_features, edge_features, backbone_coordinates, full_coordinates)
    for (
      logits,
      node_features,
      edge_features,
      backbone_coordinates,
      full_coordinates,
    ) in _compute_states_batches(
      spec,
      protein_iterator,
      model_parameters,
    )
  ]
  (
    all_logits,
    all_node_features,
    all_edge_features,
    all_backbone_coordinates,
    all_full_coordinates,
  ) = zip(
    *all_states,
    strict=False,
  )

  return {
    "logits": jnp.concatenate(all_logits, axis=0) if all_logits[0] is not None else None,
    "node_features": jnp.concatenate(all_node_features, axis=0)
    if all_node_features[0] is not None
    else None,
    "edge_features": jnp.concatenate(all_edge_features, axis=0)
    if all_edge_features[0] is not None
    else None,
    "backbone_coordinates": jnp.concatenate(all_backbone_coordinates, axis=0)
    if all_backbone_coordinates[0] is not None
    else None,
    "full_coordinates": jnp.concatenate(all_full_coordinates, axis=0)
    if all_full_coordinates[0] is not None
    else None,
    "metadata": {"spec": spec},
  }


def _derive_states_streaming(
  spec: ConformationalInferenceSpecification,
  protein_iterator: IterDataset,
  model_parameters: ModelParameters,
) -> dict[str, Any]:
  """Compute global states and streams them to an HDF5 file."""
  if not spec.output_h5_path:
    msg = "output_h5_path must be provided for streaming."
    logger.error(msg)
    raise ValueError(msg)
  logger.info("Deriving states...")
  with h5py.File(spec.output_h5_path, "w") as f:
    dsets = {}
    for batch_idx, (
      logits,
      node_features,
      edge_features,
      backbone_coordinates,
      full_coordinates,
    ) in enumerate(
      _compute_states_batches(spec, protein_iterator, model_parameters),
    ):
      states = {
        "logits": logits,
        "node_features": node_features,
        "edge_features": edge_features,
        "backbone_coordinates": backbone_coordinates,
        "full_coordinates": full_coordinates,
      }
      if batch_idx == 0:
        # Create datasets on the first batch
        for key, arr in states.items():
          if arr is not None:
            dsets[key] = f.create_dataset(
              key,
              shape=(0, *arr.shape[1:]),
              maxshape=(None, *arr.shape[1:]),
              dtype=arr.dtype,
              chunks=True,
            )
          else:  # If no data for this key, create an empty dataset
            dsets[key] = f.create_dataset(
              key,
              shape=(0,),
              maxshape=(None,),
              dtype="float32",
              chunks=True,
            )

      # Append data to datasets
      for key, arr in states.items():
        if arr is None:
          continue
        dset = dsets[key]
        current_size = dset.shape[0]
        new_size = current_size + arr.shape[0]
        dset.resize((new_size, *arr.shape[1:]))
        dset[current_size:new_size] = arr
      f.flush()

  return {
    "output_h5_path": str(spec.output_h5_path),
    "metadata": {"spec": spec},
  }


def infer_conformations(
  spec: ConformationalInferenceSpecification,
) -> ConformationalStates:
  """Infer conformational states from a protein ensemble.

  This function orchestrates the process of:
  1. Deriving features (logits, node features, or edge features) from the ensemble.
     This can be done in-memory or streamed to an HDF5 file.
  2. Fitting a Gaussian Mixture Model (GMM) to the derived features.
  3. Clustering the GMM components using DBSCAN to identify coarse-grained
     conformational states.

  Args:
    spec: A GlobalStatesSpecification object containing all necessary configurations.

  Returns:
    A ConformationalStates object with the results of the analysis.

  Raises:
    ValueError: If an invalid inference strategy is provided or if no data is produced.

  """
  states_result = derive_states(spec)

  if spec.output_h5_path:
    with h5py.File(spec.output_h5_path, "r") as f:
      all_states = f[str(spec.inference_features[0])][:]  # type: ignore[index]
  else:
    all_states = states_result[spec.inference_features[0]]  # (N, L, F)
  if all_states is None:
    msg = "No data available for GMM fitting."
    raise ValueError(msg)

  all_states = cast("jnp.ndarray", all_states)
  key = jax.random.PRNGKey(spec.random_seed)

  if spec.mode == "per":
    all_states = jnp.transpose(all_states, (1, 0, *tuple(range(2, all_states.ndim))))  # (L, N, F)
    all_states = jnp.reshape(all_states, (all_states.shape[1], -1))  # (L, N*F)
  if spec.mode == "global":
    all_states = jnp.reshape(all_states, (all_states.shape[0], -1))  # (N, L*F)

  gmm_fitter = make_fit_gmm(
    n_components=spec.gmm_n_components,
    n_features=all_states.shape[-1] if spec.mode == "per" else all_states.shape[-2],
  )
  gmm = gmm_fitter(key, all_states[..., None] if spec.mode == "per" else all_states)

  return infer_states(
    gmm=gmm,
    features=all_states,
    eps_std_scale=spec.eps_std_scale,
    min_cluster_weight=spec.min_cluster_weight,
  )
