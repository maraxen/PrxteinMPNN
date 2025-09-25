"""Initializes and fits a Gaussian Mixture Model using K-Means++ for initial assignments.

This module provides a JAX-native implementation of the K-Means clustering
algorithm with K-Means++ initialization. The resulting cluster labels are then
used to create an initial responsibility matrix to seed the training of a
Gaussian Mixture Model (GMM), leading to potentially faster convergence and
more stable results.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from gmmx import EMFitter, GaussianMixtureModelJax
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

EnsembleData = Float[Array, "num_samples num_features"]
Centroids = Float[Array, "num_clusters num_features"]
Labels = Int[Array, "num_samples"]
GMMFitFn = Callable[[EnsembleData, PRNGKeyArray], GaussianMixtureModelJax]


def _kmeans_plusplus_init(
  key: PRNGKeyArray,
  data: EnsembleData,
  num_clusters: int,
) -> Centroids:
  """K-Means++ initialization for selecting well-separated initial centroids.

  Args:
      data: The data points with shape (n_samples, n_features).
      num_clusters: The number of clusters.
      key: A JAX PRNG key.

  Returns:
      The initial centroids with shape (num_clusters, n_features).

  """
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
    valid_centroids = jax.lax.dynamic_slice_in_dim(current_centroids, 0, i)
    distances_sq = jnp.min(
      jnp.sum((data[:, None, :] - valid_centroids[None, :, :]) ** 2, axis=-1),
      axis=-1,
    )
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
  """K-Means clustering algorithm with K-Means++ initialization.

  Args:
      data: The data points with shape (n_samples, n_features).
      num_clusters: The number of clusters.
      key: A JAX PRNG key.
      max_iters: The maximum number of iterations for the algorithm.

  Returns:
      The final cluster assignments (labels) for each data point.

  """
  initial_centroids = _kmeans_plusplus_init(key, data, num_clusters)

  def kmeans_iteration(
    _: int,
    centroids: Centroids,
  ) -> Centroids:
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

  final_centroids = jax.lax.fori_loop(
    0,
    max_iters,
    kmeans_iteration,
    initial_centroids,
  )
  final_distances_sq = jnp.sum(
    (data[:, None, :] - final_centroids[None, :, :]) ** 2,
    axis=-1,
  )
  return jnp.argmin(final_distances_sq, axis=-1)


def make_fit_gmm(
  n_components: int,
  n_features: int,
  kmeans_max_iters: int = 100,
  gmm_max_iters: int = 100,
  reg_covar: float = 1e-6,
) -> GMMFitFn:
  """Create a GMM fitting function that uses K-Means++ for initialization.

  Args:
      n_components: The number of mixture components (clusters).
      n_features: The number of features for each data point.
      kmeans_max_iters: Maximum iterations for the K-Means algorithm.
      gmm_max_iters: Maximum iterations for the EM algorithm for the GMM.
      reg_covar: A small value added to the diagonal of covariance matrices
                 to ensure they are non-singular.

  Returns:
      A function that takes data and a PRNG key and returns a fitted GMM.

  """
  em_fitter = EMFitter(tol=1e-3, max_iter=gmm_max_iters, reg_covar=reg_covar)

  @jax.jit
  def fit_gmm(key: PRNGKeyArray, data: EnsembleData) -> GaussianMixtureModelJax:
    labels = _kmeans(key, data, n_components, max_iters=kmeans_max_iters)
    responsibilities = jax.nn.one_hot(labels, num_classes=n_components)
    gmm = GaussianMixtureModelJax.from_responsibilities(
      responsibilities=responsibilities,
      data=data,
    )
    fitted_gmm, _ = em_fitter.fit(gmm=gmm, data=data)
    return fitted_gmm

  return fit_gmm
