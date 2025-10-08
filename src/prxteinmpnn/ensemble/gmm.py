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
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Int, PRNGKeyArray

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    EnsembleData,
  )

from prxteinmpnn.utils.types import EnsembleData

from .em_fit import GMM, EMFitterResult, _m_step_from_responsibilities, fit_gmm_states
from .kmeans import kmeans

GMMFitFn = Callable[[EnsembleData, PRNGKeyArray], EMFitterResult]
logger = logging.getLogger(__name__)


def make_fit_gmm(
  n_components: int,
  covariance_type: Literal["full", "diag"] = "full",
  kmeans_max_iters: int = 200,
  gmm_max_iters: int = 100,
  covariance_regularization: float = 1e-3,
  eps: float = 1e-6,
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

    weights, means, covariances = _m_step_from_responsibilities(
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
    initial_gmm = GMM(
      means=means,
      covariances=covariances,
      weights=weights,
      responsibilities=responsibilities,
      n_components=n_components,
      n_features=gmm_features.shape[1],
    )

    return fit_gmm_states(
      data=gmm_features,
      gmm=initial_gmm,
      max_iter=gmm_max_iters,
      tol=1e-3,
      covariance_regularization=covariance_regularization,
      covariance_type=covariance_type,
      min_variance=1e-3,
    )

  return fit_gmm
