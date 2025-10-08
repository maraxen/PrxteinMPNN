"""Initializes and fits a Gaussian Mixture Model using K-Means++ for initial assignments.

This module provides a JAX-native implementation of the K-Means clustering
algorithm with K-Means++ initialization. The resulting cluster labels are then
used to create an initial responsibility matrix to seed the training of a
Gaussian Mixture Model (GMM), leading to potentially faster convergence and
more stable results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Int, PRNGKeyArray

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    ComponentCounts,
    EnsembleData,
    Means,
    Responsibilities,
  )

from prxteinmpnn.utils.types import EnsembleData

from .bic import compute_bic
from .em_fit import GMM, EMFitterResult, fit_gmm_states
from .kmeans import kmeans

GMMFitFn = Callable[[EnsembleData, PRNGKeyArray], EMFitterResult]
logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("min_weight", "max_weight"))
def prune_components(
  gmm: GMM,
  min_weight: float = 1e-3,
  max_weight: float = 0.99,
) -> tuple[GMM, Int]:
  """Remove mixture components with very small or very large weights.

  Components with weights below `min_weight` are considered degenerate and removed.
  Components with weights above `max_weight` dominate the mixture and may indicate
  overfitting, so they are also removed.

  Args:
    gmm: The Gaussian Mixture Model to prune.
    min_weight: Minimum weight threshold. Components below this are removed.
    max_weight: Maximum weight threshold. Components above this are removed.

  Returns:
    tuple: A tuple containing:
      - GMM: Pruned model with valid components only.
      - Array: Number of components removed.

  Example:
    >>> pruned_gmm, n_removed = prune_components(gmm, min_weight=0.01, max_weight=0.95)
    >>> print(f"Removed {n_removed} components")

  """
  valid_mask = (gmm.weights >= min_weight) & (gmm.weights <= max_weight)
  n_removed = jnp.asarray(gmm.n_components - jnp.sum(valid_mask), dtype=jnp.int32)

  if jnp.all(valid_mask):
    return gmm, jnp.asarray(0, dtype=jnp.int32)

  valid_indices = jnp.where(valid_mask, size=gmm.n_components, fill_value=-1)[0]
  valid_indices = valid_indices[valid_indices >= 0]

  new_n_components = len(valid_indices)

  new_weights = gmm.weights[valid_indices]
  new_weights = new_weights / jnp.sum(new_weights)

  new_means = gmm.means[valid_indices]
  new_covariances = gmm.covariances[valid_indices]

  if gmm.responsibilities is not None:
    new_responsibilities = gmm.responsibilities[:, valid_indices]
  else:
    new_responsibilities = jnp.zeros((0, new_n_components))

  return (
    GMM(
      weights=new_weights,
      means=new_means,
      covariances=new_covariances,
      responsibilities=new_responsibilities,
      n_components=new_n_components,
      n_features=gmm.n_features,
    ),
    n_removed,
  )


@partial(jax.jit, static_argnames=("covariance_type", "covariance_regularization", "min_variance"))
def gmm_from_responsibilities(
  data: EnsembleData,
  means: Means,
  responsibilities: Responsibilities,
  component_counts: ComponentCounts,
  covariance_type: Literal["full", "diag"] = "full",
  covariance_regularization: float = 1e-6,
  min_variance: float = 1e-3,
) -> GMM:
  """Create a GMM from data and responsibilities.

  Args:
    data: Input data array of shape (n_samples, n_features).
    means: Component means of shape (n_components, n_features).
    responsibilities: Responsibility matrix of shape (n_samples, n_components).
    component_counts: Sum of responsibilities for each component, shape (n_components,).
    covariance_type: Type of covariance matrix, either "full" or "diag".
    covariance_regularization: Regularization added to diagonal of covariance matrices.
    min_variance: Minimum variance threshold to prevent numerical instability.

  Returns:
    GMM: Gaussian Mixture Model with computed parameters.

  Raises:
    ValueError: If covariance_type is not "full" or "diag".

  """
  n_components, n_features = means.shape

  diff = data[:, None, :] - means[None, :, :]
  weighted_diff = responsibilities[:, :, None] * diff

  if covariance_type == "full":

    def component_covariance_fn(component_idx: Int) -> jax.Array:
      component_diff = diff[:, component_idx, :]
      weighted_component_diff = weighted_diff[:, component_idx, :]
      component_covariance = (
        jnp.dot(weighted_component_diff.T, component_diff) / component_counts[component_idx]
      )
      diag_values = jnp.diag(component_covariance)
      diag_values = jnp.maximum(diag_values, min_variance)
      diag_values = jax.nn.softplus(diag_values - min_variance) + min_variance
      return component_covariance.at[jnp.diag_indices(n_features)].set(
        diag_values + covariance_regularization,
      )

  elif covariance_type == "diag":

    def component_covariance_fn(component_idx: Int) -> jax.Array:
      component_diff = diff[:, component_idx, :]
      weighted_component_diff = weighted_diff[:, component_idx, :]
      component_covariance = (
        jnp.sum(weighted_component_diff * component_diff, axis=0) / component_counts[component_idx]
      )
      component_covariance = jnp.maximum(component_covariance, min_variance)
      component_covariance = jax.nn.softplus(component_covariance - min_variance) + min_variance
      return component_covariance + covariance_regularization

  covariances = jax.vmap(component_covariance_fn)(jnp.arange(n_components))

  weights = component_counts / jnp.sum(component_counts)
  weights = weights / jnp.sum(weights)

  return GMM(
    weights=weights,
    means=means,
    covariances=covariances,
    responsibilities=responsibilities,
    n_components=n_components,
    n_features=n_features,
  )


def make_fit_gmm(
  n_components: Int,
  covariance_type: Literal["full", "diag"] = "full",
  kmeans_max_iters: int = 200,
  gmm_max_iters: int = 100,
  covariance_regularization: float = 1e-6,
) -> GMMFitFn:
  """Create a GMM fitting function.

  Args:
    n_components: Number of mixture components.
    covariance_type: Type of covariance matrix, either "full" or "diag".
    kmeans_max_iters: Maximum iterations for K-Means initialization.
    gmm_max_iters: Maximum iterations for GMM fitting.
    covariance_regularization: Regularization added to diagonal of covariance matrices.

  Returns:
    Callable[[EnsembleData, PRNGKeyArray], EMFitterResult]: Function to fit GMM on data.

  """

  def fit_gmm(gmm_features: jax.Array, key: PRNGKeyArray) -> EMFitterResult:
    key, subkey = random.split(key)
    labels = kmeans(subkey, gmm_features, n_components, max_iters=kmeans_max_iters)
    responsibilities = jax.nn.one_hot(labels, num_classes=n_components)

    def cluster_means(gmm_features: jax.Array, labels: jax.Array, k: Int) -> jax.Array:
      return jnp.mean(gmm_features, axis=0, where=(labels == k)[:, None])

    initial_gmm = gmm_from_responsibilities(
      data=gmm_features,
      means=jax.vmap(cluster_means, in_axes=(None, None, 0))(
        gmm_features,
        labels,
        jnp.arange(n_components),
      ),
      responsibilities=responsibilities,
      component_counts=jnp.sum(responsibilities, axis=0),
      covariance_type=covariance_type,
      covariance_regularization=covariance_regularization,
    )

    result = fit_gmm_states(
      data=gmm_features,
      gmm=initial_gmm,
      max_iter=gmm_max_iters,
      tol=1e-3,
      covariance_regularization=covariance_regularization,
      covariance_type=covariance_type,
      min_variance=1e-3,
    )

    bic = compute_bic(
      log_likelihood=result.log_likelihood,
      n_samples=gmm_features.shape[0],
      n_components=result.gmm.n_components,
      n_features=result.gmm.n_features,
      covariance_type=covariance_type,
    )

    pruned_gmm, n_removed = prune_components(
      result.gmm,
      min_weight=1e-3,
      max_weight=0.99,
    )

    if n_removed > 0:
      result = EMFitterResult(
        gmm=pruned_gmm,
        n_iter=result.n_iter,
        log_likelihood=result.log_likelihood,
        log_likelihood_diff=result.log_likelihood_diff,
        converged=result.converged,
        features=result.features,
        bic=bic,
      )

    return result

  return fit_gmm
