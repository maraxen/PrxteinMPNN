"""Initializes and fits a Gaussian Mixture Model using K-Means++ for initial assignments.

This module provides a JAX-native implementation of the K-Means clustering
algorithm with K-Means++ initialization. The resulting cluster labels are then
used to create an initial responsibility matrix to seed the training of a
Gaussian Mixture Model (GMM), leading to potentially faster convergence and
more stable results.
"""

import logging
from collections.abc import Callable, Generator
from typing import Literal

import h5py
import jax
import jax.numpy as jnp
from gmmx.gmm import GaussianMixtureModelJax
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .em_fit import fit_gmm_generator, fit_gmm_in_memory

EnsembleData = Float[Array, "num_samples num_features"]
Centroids = Float[Array, "num_clusters num_features"]
Labels = Int[Array, "num_samples"]
GMMFitFnStreaming = Callable[[h5py.Dataset, PRNGKeyArray], GaussianMixtureModelJax]
GMMFitFnInMemory = Callable[[EnsembleData, PRNGKeyArray], GaussianMixtureModelJax]
logger = logging.getLogger(__name__)


def _kmeans_plusplus_init(
  key: PRNGKeyArray,
  data: EnsembleData,
  num_clusters: int,
) -> Centroids:
  """K-Means++ initialization for selecting well-separated initial centroids."""
  logger.info("Initializing centroids with K-Means++...")
  logger.info("Number of clusters: %d", num_clusters)
  logger.info("Data shape: %s", data.shape)
  n_samples, n_features = data.shape
  centroids = jnp.zeros((num_clusters, n_features))
  key, subkey = random.split(key)
  first_idx = random.choice(subkey, n_samples)
  centroids = centroids.at[0].set(data[first_idx])

  def select_next_centroid(
    i: int,
    state: tuple[Centroids, PRNGKeyArray],
  ) -> tuple[Centroids, PRNGKeyArray]:
    current_centroids, current_key = state
    is_valid_centroid = jnp.arange(num_clusters) < i
    distances_sq_all = jnp.sum(
      (data[:, None, :] - current_centroids[None, :, :]) ** 2,
      axis=-1,
    )
    distances_sq_masked = jnp.where(
      is_valid_centroid[None, :],
      distances_sq_all,
      jnp.inf,
    )
    distances_sq = jnp.min(distances_sq_masked, axis=-1)
    probabilities = distances_sq / jnp.sum(distances_sq)
    current_key, subkey = random.split(current_key)
    next_idx = random.choice(subkey, n_samples, p=probabilities)
    updated_centroids = current_centroids.at[i].set(data[next_idx])
    return updated_centroids, current_key

  final_centroids, _ = jax.lax.fori_loop(
    1,
    num_clusters,
    select_next_centroid,
    (centroids, key),
  )
  return final_centroids


def _kmeans(
  key: PRNGKeyArray,
  data: EnsembleData,
  num_clusters: int,
  max_iters: int = 100,
) -> Labels:
  """K-Means clustering algorithm with K-Means++ initialization."""
  initial_centroids = _kmeans_plusplus_init(key, data, num_clusters)

  def kmeans_iteration(_: int, centroids: Centroids) -> Centroids:
    distances_sq = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    labels = jnp.argmin(distances_sq, axis=-1)

    def update_centroid(j: int, centroids_state: Centroids) -> Centroids:
      mask = labels == j
      count = jnp.sum(mask)
      new_centroid = jnp.where(
        count > 0,
        jnp.sum(data * mask[:, None], axis=0) / count,
        centroids_state[j],
      )
      return centroids_state.at[j].set(new_centroid)

    return jax.lax.fori_loop(0, num_clusters, update_centroid, centroids)

  final_centroids = jax.lax.fori_loop(0, max_iters, kmeans_iteration, initial_centroids)
  final_distances_sq = jnp.sum(
    (data[:, None, :] - final_centroids[None, :, :]) ** 2,
    axis=-1,
  )
  return jnp.argmin(final_distances_sq, axis=-1)


def make_fit_gmm_streaming(
  n_components: int,
  reshaping_mode: Literal["global", "per"] = "global",
  batch_size: int = 4096,
  kmeans_init_samples: int = 1000,
  kmeans_max_iters: int = 100,
  gmm_max_iters: int = 100,
  reg_covar: float = 1e-6,
) -> GMMFitFnStreaming:
  """Create a GMM fitting function that streams data from an HDF5 file."""

  def _data_generator(dataset: h5py.Dataset) -> Generator[jax.Array, None, None]:
    n_total = dataset.shape[0]
    for i in range(0, n_total, batch_size):
      yield jnp.reshape(jnp.array(dataset[i : i + batch_size]), (min(n_total - i, batch_size), -1))

  def fit_gmm(dataset: h5py.Dataset, key: PRNGKeyArray) -> GaussianMixtureModelJax:
    n_total_samples = dataset.shape[0]
    init_samples = min(kmeans_init_samples, n_total_samples)
    logger.info("Running K-Means++ on a subset of %d samples...", init_samples)
    key, subkey = random.split(key)
    sample_indices = random.choice(
      subkey,
      n_total_samples,
      shape=(init_samples,),
      replace=False,
    )
    init_data = jnp.array(dataset[jnp.sort(sample_indices)])
    if reshaping_mode == "per":
      gmm_features = jnp.transpose(
        init_data,
        (1, 0, *tuple(range(2, init_data.ndim))),
      )  # (L, N, F)
      gmm_features = jnp.reshape(gmm_features, (init_data.shape[0], -1))  # (L, N*F)
    elif reshaping_mode == "global":
      gmm_features = jnp.reshape(init_data, (init_data.shape[0], -1))

    key, subkey = random.split(key)
    labels = _kmeans(subkey, gmm_features, n_components, max_iters=kmeans_max_iters)
    responsibilities = jax.nn.one_hot(labels, num_classes=n_components)

    logger.info("Initializing GMM from K-Means results...")
    initial_gmm = GaussianMixtureModelJax.from_responsibilities(
      x=jnp.expand_dims(gmm_features, axis=(1, 3)),
      resp=jnp.expand_dims(responsibilities, axis=(2, 3)),
      reg_covar=reg_covar,
    )

    logger.info("Fitting GMM using batch-based EM with batch size %d...", batch_size)
    data_gen = _data_generator(dataset)
    result = fit_gmm_generator(
      data_generator=data_gen,
      initial_gmm=initial_gmm,
      n_total_samples=n_total_samples,
      max_iter=gmm_max_iters,
      tol=1e-3,
      reg_covar=reg_covar,
    )
    logger.info(
      "GMM fitting finished in %d iterations. Converged: %s",
      result.n_iter,
      result.converged,
    )
    return result.gmm

  return fit_gmm


def make_fit_gmm_in_memory(
  n_components: int,
  kmeans_max_iters: int = 100,
  gmm_max_iters: int = 100,
  reg_covar: float = 1e-6,
) -> GMMFitFnInMemory:
  """Create a GMM fitting function for in-memory JAX arrays."""

  def fit_gmm(data: EnsembleData, key: PRNGKeyArray) -> GaussianMixtureModelJax:
    logger.info("Running K-Means++ on the full in-memory dataset...")
    key, subkey = random.split(key)
    labels = _kmeans(subkey, data, n_components, max_iters=kmeans_max_iters)
    responsibilities = jax.nn.one_hot(labels, num_classes=n_components)

    logger.info("Initializing GMM from K-Means results...")
    initial_gmm = GaussianMixtureModelJax.from_responsibilities(
      x=data[None, ...],
      resp=responsibilities[None, ...],
      reg_covar=reg_covar,
    )

    logger.info("Fitting GMM using in-memory EM...")
    result = fit_gmm_in_memory(
      x=data,
      initial_gmm=initial_gmm,
      max_iter=gmm_max_iters,
      tol=1e-3,
      reg_covar=reg_covar,
    )
    logger.info(
      "GMM fitting finished in %d iterations. Converged: %s",
      result.n_iter,
      result.converged,
    )
    return result.gmm

  return fit_gmm
