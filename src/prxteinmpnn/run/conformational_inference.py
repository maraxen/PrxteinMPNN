"""Conformational-inference from ProteinMPNN logits or features."""

import logging
import pathlib
import sys
from collections.abc import Callable, Generator
from typing import Any, cast

import h5py
import jax
import jax.numpy as jnp
import pcax
from grain.python import IterDataset

from prxteinmpnn.ensemble.ci import infer_states
from prxteinmpnn.ensemble.dbscan import (
  ConformationalStates,
  GMMClusteringResult,
)
from prxteinmpnn.ensemble.gmm import make_fit_gmm
from prxteinmpnn.ensemble.pca import pca_transform
from prxteinmpnn.run.prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import ConformationalInferenceSpecification
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils.data_structures import GMM
from prxteinmpnn.utils.types import (
  EdgeFeatures,
  Logits,
  ModelParameters,
  NodeFeatures,
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
    if pathlib.Path(spec.output_h5_path).exists():
      if not spec.overwrite_cache:
        return {"output_h5_path": str(spec.output_h5_path), "metadata": {"spec": spec}}
      logger.info("Overwriting existing HDF5 file at %s", spec.output_h5_path)
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


def _compute_states_batches(
  spec: ConformationalInferenceSpecification,
  protein_iterator: IterDataset,
  model_parameters: ModelParameters,
) -> Generator[
  tuple[
    Logits | None,
    NodeFeatures | None,
    EdgeFeatures | None,
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
    if any(
      feature in spec.inference_features for feature in ["logits", "node_features", "edge_features"]
    ):
      keys = jax.random.split(jax.random.PRNGKey(spec.random_seed), n_frames)

      vmap_axes = (0, 0, None, None, None, None, None)
      if not is_conditional:
        vmap_axes += (None,)
        inference_args = (
          keys,
          batched_ensemble.coordinates,
          batched_ensemble.full_atom_mask[0, :, 0],
          batched_ensemble.residue_index[0],
          batched_ensemble.chain_index[0],
        )
      else:
        inference_args = (
          keys,
          batched_ensemble.coordinates,
          batched_ensemble.one_hot_sequence[0],
          batched_ensemble.full_atom_mask[0, :, 0],
          batched_ensemble.residue_index[0],
          batched_ensemble.chain_index[0],
          batched_ensemble.aatype[0],
        )

      batch_states = jax.vmap(get_logits, in_axes=vmap_axes)(
        *inference_args,
        *static_args,
      )

      logits, node_features, edge_features = batch_states
    else:
      logits, node_features, edge_features = None, None, None
    yield (
      logits if "logits" in spec.inference_features else None,
      node_features if "node_features" in spec.inference_features else None,
      edge_features if "edge_features" in spec.inference_features else None,
    )


def _derive_states_in_memory(
  spec: ConformationalInferenceSpecification,
  protein_iterator: IterDataset,
  model_parameters: ModelParameters,
) -> dict[str, jax.Array | dict[str, ConformationalInferenceSpecification] | None]:
  """Compute global states and stores them in memory."""
  all_batches = list(_compute_states_batches(spec, protein_iterator, model_parameters))
  all_logits, all_node_features, all_edge_features = zip(
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
    ) in enumerate(
      _compute_states_batches(spec, protein_iterator, model_parameters),
    ):
      states = {
        "logits": logits,
        "node_features": node_features,
        "edge_features": edge_features,
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
          if key == spec.inference_features[0]:
            total_frames += arr.shape[0]
      logger.info("...processed %d frames...", total_frames)
      f.flush()
  return {"output_h5_path": str(spec.output_h5_path), "metadata": {"spec": spec}}


def _derive_gmm_features(
  spec: ConformationalInferenceSpecification,
  all_states: jax.Array,
) -> tuple[pcax.pca.PCAState, jax.Array]:
  """Derive GMM features from all states based on the specified mode."""
  n_samples = all_states.shape[0]
  gmm_features = None
  if spec.mode == "per":
    gmm_features = jnp.transpose(
      all_states,
      (1, 0, *tuple(range(2, all_states.ndim))),
    )  # (L, N, F)
    gmm_features = jnp.reshape(gmm_features, (n_samples, -1))  # (L, N*F)
  elif spec.mode == "global":
    gmm_features = jnp.reshape(all_states, (n_samples, -1))  # (N, L*F)
  else:
    msg = f"Unknown mode: {spec.mode}"
    raise ValueError(msg)
  if gmm_features is None:
    msg = "GMM features could not be determined."
    raise ValueError(msg)
  return pca_transform(
    gmm_features,
    n_components=spec.pca_n_components,
  )


def save_pca_state(h5_file: h5py.File, pca_state: pcax.pca.PCAState, n_components: int) -> None:
  """Save PCA state to an HDF5 file."""
  grp = h5_file.create_group(f"pca_{n_components}_state")
  for key, value in pca_state._asdict().items():
    grp.create_dataset(key, data=jnp.array(value), compression="gzip")


def infer_conformations(  # noqa: PLR0915
  spec: ConformationalInferenceSpecification,
) -> tuple[ConformationalStates, GMMClusteringResult, GMM]:
  """Infer conformational states from a protein ensemble."""
  states_result = derive_states(spec)
  key = jax.random.PRNGKey(spec.random_seed)
  feature_key = str(spec.inference_features[0])
  result = None

  if spec.output_h5_path:
    with h5py.File(spec.output_h5_path, "r") as f:
      all_states_h5 = cast("h5py.Dataset", f[feature_key])
      if all_states_h5.shape[0] == 0:  # type: ignore[attr-defined]
        msg = "No data in HDF5 file for GMM fitting."
        raise ValueError(msg)

      gmm_fitter_fn = make_fit_gmm(
        n_components=jnp.array(spec.gmm_n_components, dtype=jnp.int32),
        covariance_type=spec.covariance_type,
        gmm_max_iters=spec.gmm_max_iters,
      )
      n_total_samples = all_states_h5.shape[0]
      key, subkey = jax.random.split(key)

      pca_found = f"pca_{spec.pca_n_components}_components" in f
      if pca_found:
        logger.info("Loading precomputed PCA components from HDF5...")
        pca_components = cast("h5py.Dataset", f[f"pca_{spec.pca_n_components}_components"])
        gmm_features: jnp.ndarray = jnp.array(pca_components[:])
        pca_state_group = cast("h5py.Group", f[f"pca_{spec.pca_n_components}_state"])
        pca_state = pcax.pca.PCAState(
          components=jnp.array(pca_state_group["components"][:]),  # type: ignore[index]
          means=jnp.array(pca_state_group["means"][:]),  # type: ignore[index]
          explained_variance=jnp.array(pca_state_group["explained_variance"][:]),  # type: ignore[index]
        )
        logger.info("PCA components shape: %s", gmm_features.shape)
      else:
        init_samples = min(spec.kmeans_init_samples, n_total_samples)
        sample_indices = jax.random.choice(
          subkey,
          n_total_samples,
          shape=(init_samples,),
          replace=False,
        )
        init_data = jnp.array(all_states_h5[jnp.sort(sample_indices)])
        pca_state, gmm_features = _derive_gmm_features(spec, init_data)
        logger.info("PCA components shape: %s", gmm_features.shape)
    if not pca_found:  # this is separate because we want to save after closing the file
      with h5py.File(spec.output_h5_path, "a") as f:
        f.create_dataset(
          f"pca_{spec.pca_n_components}_components",
          data=jnp.array(gmm_features),
          compression="gzip",
        )
        save_pca_state(f, pca_state, spec.pca_n_components)
        logger.info("Saved PCA components to HDF5.")
    em_result = gmm_fitter_fn(gmm_features, key)
    if not em_result.converged:
      logger.warning("GMM fitting did not converge.")
    result = em_result

  else:  # In-memory processing
    all_states = states_result[feature_key]
    if all_states is None or all_states.shape[0] == 0:
      msg = "No data available for GMM fitting."
      raise ValueError(msg)

    features_for_ci = cast("jnp.ndarray", all_states)

    pca_state, gmm_features = _derive_gmm_features(spec, features_for_ci)
    logger.info("PCA components shape: %s", gmm_features.shape)
    gmm_fitter_fn = make_fit_gmm(
      n_components=spec.gmm_n_components,
      covariance_type=spec.covariance_type,
      gmm_max_iters=spec.gmm_max_iters,
    )
    em_result = gmm_fitter_fn(gmm_features, key)

    if not em_result.converged:
      logger.warning("GMM fitting did not converge.")

    result = em_result
  if result is None:
    msg = "GMM fitting result is None."
    raise ValueError(msg)
  return infer_states(
    gmm=result.gmm,
    features=jnp.array(result.features),
    eps_std_scale=spec.eps_std_scale,
    min_cluster_weight=spec.min_cluster_weight,
  )
