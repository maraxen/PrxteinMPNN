"""Expectation-Maximization algorithm for Gaussian Mixture Models.

Adapted from gmmx (https://github.com/google/gmmx)
with modifications to support fitting from data generators, such as
HDF5 datasets.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
from jax import numpy as jnp

from prxteinmpnn.utils.data_structures import GMM, EMFitterResult, EMLoopState

from .bic import compute_bic

if TYPE_CHECKING:
  from jaxtyping import Float

  from prxteinmpnn.utils.types import (
    Converged,
    Covariances,
    EnsembleData,
    LogLikelihood,
    Means,
    Responsibilities,
    Weights,
  )

logger = logging.getLogger(__name__)


class Axis(int, Enum):
  """Internal axis order."""

  batch = 0
  components = 1
  features = 2
  features_covar = 3


@jax.jit
def _mahalanobis_distance_squared(
  data: EnsembleData,
  means: Means,
  precisions: Covariances,
) -> jax.Array:
  """Compute the squared Mahalanobis distance."""
  diff = data[:, None, :] - means[None, :, :]
  return jnp.sum((diff**2) * precisions[None, :, :], axis=2)


DIAG_NDIM, FULL_NDIM = 2, 3


@jax.jit
def log_likelihood_diag(
  data: EnsembleData,
  means: Means,
  covariances: Covariances,
) -> LogLikelihood:
  """Compute log likelihood for diagonal covariance GMM."""
  _, n_features = data.shape
  log_covariance_determinant = jnp.sum(jnp.log(covariances), axis=1)
  precision = 1.0 / covariances
  mahalanobis_distance = _mahalanobis_distance_squared(data, means, precision)
  return -0.5 * (
    n_features * jnp.log(2 * jnp.pi) + mahalanobis_distance + log_covariance_determinant[None, :]
  )


@jax.jit
def log_likelihood_full(
  data: EnsembleData,
  means: Means,
  covariances: Covariances,
) -> LogLikelihood:
  """Compute log likelihood for full covariance GMM.

  Args:
    data: Data array of shape (n_samples, n_features).
    means: Component means of shape (n_components, n_features).
    covariances: Full covariance matrices of shape (n_components, n_features, n_features).

  Returns:
    Log-likelihood array of shape (n_samples, n_components).

  """
  n_features = data.shape[1]
  n_components = covariances.shape[0]

  cholesky_lower = jax.scipy.linalg.cholesky(covariances, lower=True)

  log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky_lower, axis1=1, axis2=2)), axis=1)

  diff = data[:, None, :] - means[None, :, :]

  def mahalanobis_for_component(k: jax.Array) -> jax.Array:
    y = jax.scipy.linalg.solve_triangular(cholesky_lower[k], diff[:, k, :].T, lower=True)
    return jnp.sum(y**2, axis=0)

  mahalanobis = jax.vmap(mahalanobis_for_component)(jnp.arange(n_components)).T

  return -0.5 * (n_features * jnp.log(2 * jnp.pi) + log_det[None, :] + mahalanobis)


@jax.jit
def log_likelihood(data: EnsembleData, means: Means, covariances: Covariances) -> LogLikelihood:
  """Compute log likelihood from the covariance for a given feature vector.

  Dispatches to the correct implementation based on covariance matrix shape.
  """
  cov_values = jnp.squeeze(covariances)
  if cov_values.ndim == DIAG_NDIM:
    return log_likelihood_diag(data, means, cov_values)
  if cov_values.ndim == FULL_NDIM:
    return log_likelihood_full(data, means, cov_values)

  msg = f"Unsupported covariance shape: {covariances.shape}"
  raise ValueError(msg)


@jax.jit
def _e_step_diag(data: EnsembleData, gmm: GMM) -> tuple[Means, Responsibilities]:
  """E-step for diagonal covariance GMM."""
  log_prob = log_likelihood(data, gmm.means, gmm.covariances)
  weighted_log_prob = log_prob + jnp.log(gmm.weights)
  log_prob_norm = jax.scipy.special.logsumexp(weighted_log_prob, axis=1)
  log_resp = weighted_log_prob - log_prob_norm[:, None]
  return jnp.mean(log_prob_norm), log_resp


@jax.jit
def _e_step_full(data: EnsembleData, gmm: GMM) -> tuple[Means, Responsibilities]:
  """E-step for full covariance GMM."""
  log_prob = log_likelihood(data, gmm.means, gmm.covariances)
  weighted_log_prob = log_prob + jnp.log(gmm.weights)
  log_prob_norm = jax.scipy.special.logsumexp(
    weighted_log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  log_resp = weighted_log_prob - log_prob_norm
  return jnp.mean(log_prob_norm), log_resp


@partial(jax.jit, static_argnames=("covariance_type",))
def _e_step(
  data: Float,
  gmm: GMM,
  covariance_type: Literal["full", "diag"] = "diag",
) -> tuple[Means, Responsibilities]:
  """Dispatcher for the E-step based on covariance type."""
  if covariance_type == "full":
    return _e_step_full(data, gmm)
  return _e_step_diag(data, gmm)


@partial(
  jax.jit,
  static_argnames=("covariance_regularization", "covariance_type", "min_variance"),
)
def _m_step_from_responsibilities(
  data: EnsembleData,
  means: Means,
  covariances: Covariances,
  responsibilities: Responsibilities,
  covariance_regularization: float,
  covariance_type: Literal["full", "diag"] = "full",
  min_variance: float = 1e-3,
  eps: float = 1e-6,
) -> tuple[Weights, Means, Covariances]:
  """Maximization (M) step for in-memory data using responsibilities."""
  component_counts = jnp.sum(responsibilities, axis=Axis.batch)

  safe_component_counts = jnp.where(component_counts == 0, 1.0, component_counts)
  safe_component_counts = jnp.maximum(safe_component_counts, eps)

  updated_means = jnp.einsum("ij,ik->jk", responsibilities, data) / safe_component_counts[..., None]
  means = jnp.asarray(jnp.where(component_counts[..., None] > eps, updated_means, means))

  if covariance_type == "full":
    diff = data[:, None, :] - means[None, :, :]
    weighted_diff = responsibilities[:, :, None] * diff

    def component_covariance_fn(component_idx: jax.Array) -> jax.Array:
      component_diff = diff[:, component_idx, :]
      weighted_component_diff = weighted_diff[:, component_idx, :]
      component_covariance = (
        jnp.dot(weighted_component_diff.T, component_diff) / safe_component_counts[component_idx]
      )
      component_covariance = 0.5 * (component_covariance + component_covariance.T)
      diag_values = jnp.diag(component_covariance)
      diag_values = jnp.maximum(diag_values, min_variance)
      diag_values = jax.nn.softplus(diag_values - min_variance) + min_variance
      return component_covariance.at[jnp.diag_indices(component_diff.shape[1])].set(
        diag_values + covariance_regularization,
      )

    nk = component_counts[..., None, None]
  elif covariance_type == "diag":
    diff = data[:, None, :] - means[None, :, :]
    weighted_diff = responsibilities[:, :, None] * diff

    def component_covariance_fn(component_idx: jax.Array) -> jax.Array:
      component_diff = diff[:, component_idx, :]
      weighted_component_diff = weighted_diff[:, component_idx, :]
      component_covariance = (
        jnp.sum(weighted_component_diff * component_diff, axis=0)
        / safe_component_counts[component_idx]
      )
      component_covariance = jnp.maximum(component_covariance, min_variance)
      component_covariance = jax.nn.softplus(component_covariance - min_variance) + min_variance
      return component_covariance + covariance_regularization

    nk = component_counts[..., None]

  n_components = means.shape[0]
  updated_covariances = jax.vmap(component_covariance_fn)(jnp.arange(n_components))
  covariances_final = jnp.where(
    nk > eps,
    updated_covariances,
    covariances,
  )
  weights = safe_component_counts / data.shape[Axis.batch]
  weights = weights / jnp.maximum(jnp.sum(weights), eps)
  return weights, means, covariances_final


@partial(
  jax.jit,
  static_argnames=(
    "covariance_type",
    "max_iter",
    "min_iter",
    "tol",
    "covariance_regularization",
    "min_variance",
  ),
)
def fit_gmm_states(
  data: EnsembleData,
  gmm: GMM,
  covariance_type: Literal["full", "diag"] = "full",
  max_iter: int = 100,
  min_iter: int = 10,
  tol: float = 1e-3,
  covariance_regularization: float = 1e-3,
  min_variance: float = 1e-3,
  eps: float = 1e-6,
) -> EMFitterResult:
  """Fit a GMM to in-memory data using the EM algorithm.

  Args:
    data: Input data array of shape (n_samples, n_features).
    gmm: Initial GMM parameters.
    covariance_type: Type of covariance matrix, either "full" or "diag".
    max_iter: Maximum number of EM iterations.
    min_iter: Minimum number of iterations before checking convergence.
    tol: Convergence tolerance on log-likelihood difference.
    covariance_regularization: Regularization added to diagonal of covariance matrices.
    min_variance: Minimum variance threshold to prevent numerical instability.
    eps: Small value to avoid division by zero.

  Returns:
    EMFitterResult: Result containing the fitted GMM and convergence information.

  """

  @jax.jit
  def em_step_fn(state: EMLoopState) -> EMLoopState:
    """Run a single EM step."""
    log_likelihood, log_resp = _e_step(data, state.gmm, covariance_type)
    weights, means, covariances = _m_step_from_responsibilities(
      data,
      state.gmm.means,
      state.gmm.covariances,
      jnp.exp(log_resp),
      covariance_regularization,
      covariance_type,
      min_variance,
      eps,
    )
    return EMLoopState(
      gmm=GMM(
        means=means,
        covariances=covariances,
        weights=weights,
        responsibilities=log_resp,
        n_components=state.gmm.n_components,
        n_features=state.gmm.n_features,
      ),
      n_iter=state.n_iter + 1,
      log_likelihood=log_likelihood,
      log_likelihood_diff=jnp.abs(log_likelihood - state.log_likelihood),
    )

  def em_cond_fn(state: EMLoopState) -> Converged:
    """Check if EM should continue iterating."""
    result = state
    jax.debug.print(
      "EM iteration {}/{}: log-likelihood = {}, diff = {}, min_weight = {}, max_weight = {}",
      result.n_iter,
      max_iter,
      result.log_likelihood,
      result.log_likelihood_diff,
      jnp.min(result.gmm.weights),
      jnp.max(result.gmm.weights),
    )
    has_converged = jnp.abs(result.log_likelihood_diff) < tol
    keep_going = (~has_converged) | (result.n_iter < min_iter)
    return (result.n_iter < max_iter) & keep_going

  initial_state = EMLoopState(
    gmm=gmm,
    n_iter=0,
    log_likelihood=jnp.asarray(-jnp.inf),
    log_likelihood_diff=jnp.asarray(jnp.inf),
  )

  final_state = jax.lax.while_loop(em_cond_fn, em_step_fn, initial_state)
  final_state = jax.block_until_ready(final_state)
  bic = compute_bic(
    log_likelihood=final_state.log_likelihood,
    n_samples=data.shape[0],
    n_components=gmm.n_components,
    n_features=gmm.n_features,
    covariance_type=covariance_type,
  )

  return EMFitterResult(
    gmm=final_state.gmm,
    n_iter=final_state.n_iter,
    log_likelihood=final_state.log_likelihood,
    log_likelihood_diff=final_state.log_likelihood_diff,
    converged=final_state.log_likelihood_diff < tol,
    features=data,
    bic=bic,
  )
