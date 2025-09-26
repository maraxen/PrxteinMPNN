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
from prxteinmpnn.ensemble.gmm import make_fit_gmm_in_memory, make_fit_gmm_streaming
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
  """Derive conformational states from a protein ensemble using a specified model."""
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
      raise NotImplementedError(msg)
    case "coordinates":
      return lambda *args, **kwargs: (  # type: ignore[return]  # noqa: ARG005
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
      ), False
    case _:
      msg = f"Invalid inference strategy: {spec.inference_strategy}"
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
  """Generate and yield batches of computed residue states using in_axes for vmap."""
  get_logits, is_conditional = _get_logits_fn(spec, model_parameters)
  static_args = (None, 48, None)

  logger.info("Iterating through frames/proteins...")
  for batched_ensemble in protein_iterator:
    n_frames = batched_ensemble.coordinates.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(spec.random_seed), n_frames)

    vmap_axes = (0, 0, None, None, None, None, None)
    if not is_conditional:
      vmap_axes += (None,)

    batch_states = jax.vmap(get_logits, in_axes=vmap_axes)(
      keys,
      batched_ensemble.coordinates,
      batched_ensemble.one_hot_sequence[0] if is_conditional else None,
      batched_ensemble.atom_mask[:, 0] if is_conditional else batched_ensemble.atom_mask[0, :, 0],
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
  all_batches = list(_compute_states_batches(spec, protein_iterator, model_parameters))
  all_logits, all_node_features, all_edge_features, all_backbone_coords, all_full_coords = zip(
    *all_batches,
    strict=False,
  )

  return {
    "logits": jnp.concatenate(all_logits) if all_logits[0] is not None else None,
    "node_features": jnp.concatenate(all_node_features)
    if all_node_features[0] is not None
    else None,
    "edge_features": jnp.concatenate(all_edge_features)
    if all_edge_features[0] is not None
    else None,
    "backbone_coordinates": jnp.concatenate(all_backbone_coords)
    if all_backbone_coords[0] is not None
    else None,
    "full_coordinates": jnp.concatenate(all_full_coords)
    if all_full_coords[0] is not None
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
    raise ValueError(msg)
  logger.info("Deriving states and streaming to %s...", spec.output_h5_path)
  with h5py.File(spec.output_h5_path, "w") as f:
    dsets = {}
    total_frames = 0
    for batch_idx, (
      logits,
      node_features,
      edge_features,
      backbone_coords,
      full_coords,
    ) in enumerate(
      _compute_states_batches(spec, protein_iterator, model_parameters),
    ):
      states = {
        "logits": logits,
        "node_features": node_features,
        "edge_features": edge_features,
        "backbone_coordinates": backbone_coords,
        "full_coordinates": full_coords,
      }
      if batch_idx == 0:
        for key, arr in states.items():
          if arr is not None:
            dsets[key] = f.create_dataset(
              key,
              shape=(0, *arr.shape[1:]),
              maxshape=(None, *arr.shape[1:]),
              dtype=arr.dtype,
              chunks=True,
            )
      for key, arr in states.items():
        if arr is not None and key in dsets:
          dset = dsets[key]
          new_size = dset.shape[0] + arr.shape[0]
          dset.resize((new_size, *arr.shape[1:]))
          dset[-arr.shape[0] :] = arr
          if key == "logits":
            total_frames += arr.shape[0]
      logger.info("...processed %d frames...", total_frames)
      f.flush()
  return {"output_h5_path": str(spec.output_h5_path), "metadata": {"spec": spec}}


def infer_conformations(
  spec: ConformationalInferenceSpecification,
) -> ConformationalStates:
  """Infer conformational states from a protein ensemble."""
  states_result = derive_states(spec)
  key = jax.random.PRNGKey(spec.random_seed)
  feature_key = str(spec.inference_features[0])

  if spec.output_h5_path:
    with h5py.File(spec.output_h5_path, "r") as f:
      all_states_h5 = f[feature_key]
      if all_states_h5.shape[0] == 0:  # type: ignore[attr-defined]
        msg = "No data in HDF5 file for GMM fitting."
        raise ValueError(msg)

      # The data for clustering must be reshaped and loaded into memory.
      features_for_ci = jnp.array(all_states_h5)
      n_samples = features_for_ci.shape[0]
      gmm_fitter_fn = make_fit_gmm_streaming(n_components=spec.gmm_n_components)
      gmm = gmm_fitter_fn(all_states_h5, key)  # type: ignore[arg-type]

  else:  # In-memory processing
    all_states = states_result[feature_key]
    if all_states is None or all_states.shape[0] == 0:
      msg = "No data available for GMM fitting."
      raise ValueError(msg)

    features_for_ci = cast("jnp.ndarray", all_states)
    n_samples = features_for_ci.shape[0]

    gmm_features = None
    if spec.mode == "per":
      gmm_features = jnp.transpose(
        all_states,
        (1, 0, *tuple(range(2, features_for_ci.ndim))),
      )  # (L, N, F)
      gmm_features = jnp.reshape(gmm_features, (n_samples, -1))  # (L, N*F)
    if spec.mode == "global":
      gmm_features = jnp.reshape(features_for_ci, (n_samples, -1))  # (N, L*F)

    gmm_fitter_fn = make_fit_gmm_in_memory(
      n_components=spec.gmm_n_components,
      gmm_max_iters=100,  # Example, should be in spec
    )
    if gmm_features is None:
      msg = "GMM features could not be determined."
      raise ValueError(msg)
    gmm = gmm_fitter_fn(gmm_features, key)

  return infer_states(
    gmm=gmm,
    features=jnp.reshape(features_for_ci, (n_samples, -1)),
    eps_std_scale=spec.eps_std_scale,
    min_cluster_weight=spec.min_cluster_weight,
  )
