"""K-Means clustering algorithm with K-Means++ initialization.

This module provides a JAX-native implementation of the K-Means clustering
algorithm with K-Means++ initialization. The resulting cluster labels are then
used to create an initial responsibility matrix to seed the training of a
Gaussian Mixture Model (GMM), leading to potentially faster convergence and
more stable results.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from prxteinmpnn.utils.types import Centroids, EnsembleData, Labels

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames="num_clusters")
def _kmeans_plusplus_init(
  key: PRNGKeyArray,
  data: EnsembleData,
  num_clusters: int,
) -> Centroids:
  """K-Means++ initialization for selecting well-separated initial centroids."""
  n_samples, n_features = data.shape
  centroids = jnp.zeros((num_clusters, n_features))
  key, subkey = random.split(key)
  first_idx = random.choice(subkey, n_samples)
  centroids = centroids.at[0].set(data[first_idx])

  def select_next_centroid(
    i: Int,
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


@partial(jax.jit, static_argnames=("max_iters", "num_clusters"))
def kmeans(
  key: PRNGKeyArray,
  data: EnsembleData,
  num_clusters: int,
  max_iters: int = 100,
) -> Labels:
  """K-Means clustering algorithm with K-Means++ initialization."""
  initial_centroids = _kmeans_plusplus_init(key, data, num_clusters)

  def kmeans_iteration(_: Int, centroids: Centroids) -> Centroids:
    distances_sq = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    labels = jnp.argmin(distances_sq, axis=-1)

    def update_centroid(j: Int, centroids_state: Centroids) -> Centroids:
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
