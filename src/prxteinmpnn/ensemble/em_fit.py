"""Expectation-Maximization algorithm for Gaussian Mixture Models.

Adapted from gmmx (https://github.com/google/gmmx)
with modifications to support fitting from data generators, such as
HDF5 datasets.
"""

from collections.abc import Generator
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import jax
from flax.struct import dataclass
from jax import numpy as jnp


@dataclass
class GMM:
  """Dataclass to hold GMM parameters."""

  means: jax.Array
  covariances: jax.Array
  weights: jax.Array
  responsibilities: jax.Array
  n_components: int
  n_features: int


class Axis(int, Enum):
  """Internal axis order."""

  batch = 0
  components = 1
  features = 2
  features_covar = 3


def precisions_cholesky(covariances: jax.Array) -> jax.Array:
  """Compute precision matrices."""
  cov_chol = jax.scipy.linalg.cholesky(covariances, lower=True)

  identity = jnp.expand_dims(
    jnp.eye(covariances.shape[Axis.features]),
    axis=(Axis.batch, Axis.components),
  )
  b = jnp.repeat(identity, covariances.shape[Axis.components], axis=Axis.components)
  precisions_chol = jax.scipy.linalg.solve_triangular(cov_chol, b, lower=True)
  return precisions_chol.mT


def log_likelihood(data: jax.Array, means: jax.Array, covariances: jax.Array) -> jax.Array:
  """Compute log likelihood from the covariance for a given feature vector.

  Parameters
  ----------
  x : jax.array
      Feature vectors
  means : jax.array
      Means of the components

  Returns
  -------
  log_prob : jax.array
      Log likelihood

  """
  precisions = precisions_cholesky(covariances)

  y = jnp.matmul(data.mT, precisions) - jnp.matmul(
    means.mT,
    precisions,
  )
  return jnp.sum(
    jnp.square(y),
    axis=(Axis.features, Axis.features_covar),
    keepdims=True,
  )


class _EMLoopState(NamedTuple):
  """State for the in-memory EM loop."""

  gmm: GMM
  n_iter: int
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array


@dataclass
class EMFitterResult:
  """Result of the Expectation-Maximization fitting process.

  Attributes
  ----------
  gmm : GMM
      The final fitted Gaussian mixture model.
  n_iter : int
      The total number of iterations performed.
  log_likelihood : jax.Array
      The log-likelihood of the data under the final model.
  converged : bool
      A boolean indicating if the algorithm converged within the max iterations.

  """

  gmm: GMM
  n_iter: int
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array
  converged: bool


def _e_step(
  data: jax.Array,
  means: jax.Array,
  covariances: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Run Expectation (E) step of the EM algorithm.

  Args:
      data: Input data of shape (N, F).
      means: GMM means of shape (K, F).
      covariances: GMM covariances of shape (K, F, F) or (K, F, 1) for diagonal.

  Returns:
      A tuple containing the mean log-likelihood for the batch and the log responsibilities.

  """
  log_prob = log_likelihood(data, means, covariances)
  log_prob_norm = jax.scipy.special.logsumexp(
    log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  log_resp = log_prob - log_prob_norm
  return jnp.mean(log_prob_norm), log_resp


@partial(jax.jit, static_argnames=("reg_covar", "covariance_type"))
def _m_step_from_responsibilities(
  data: jax.Array,
  means: jax.Array,
  covariances: jax.Array,
  responsibilities: jax.Array,
  reg_covar: float,
  covariance_type: Literal["full", "diag"] = "full",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Maximization (M) step for in-memory data using responsibilities."""
  responsibilities = jnp.squeeze(responsibilities, axis=(-2, -1))
  nk = jnp.sum(responsibilities, axis=Axis.batch)

  safe_nk = jnp.where(nk == 0, 1.0, nk)

  updated_means = jnp.einsum("ij,ik->jk", responsibilities, data) / safe_nk[..., None]
  means = jnp.asarray(jnp.where(nk[..., None] > 0, updated_means, means))

  if covariance_type == "full":
    updated_covs = (
      jnp.einsum("ij,ik,il->jkl", responsibilities, data, data) / safe_nk[..., None, None]
    )
    updated_covs = updated_covs - jnp.einsum("...i,...j->...ij", means, means)
    updated_covs += reg_covar * jnp.eye(means.shape[-1])

    original_covs_squeezed = jnp.squeeze(covariances)
    covariances_3d = jnp.where(nk[..., None, None] > 0, updated_covs, original_covs_squeezed)
    covariances_final = covariances_3d[None, ...]
  elif covariance_type == "diag":
    updated_vars = (
      jnp.einsum("ij,ik->jk", responsibilities, data**2) / safe_nk[..., None] - means**2
    )
    updated_vars += reg_covar

    original_vars_squeezed = jnp.squeeze(covariances)
    variances_2d = jnp.where(nk[..., None] > 0, updated_vars, original_vars_squeezed)
    covariances_final = variances_2d[None, ..., None]

  weights = nk / data.shape[Axis.batch]

  return weights, means, covariances_final


@partial(jax.jit, static_argnames=("n_total_samples", "reg_covar", "covariance_type"))
def _m_step_from_stats(
  means: jax.Array,
  covariances: jax.Array,
  nk: jax.Array,
  xk: jax.Array,
  sk: jax.Array,
  n_total_samples: int,
  reg_covar: float,
  covariance_type: Literal["full", "diag"],
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Update GMM parameters from accumulated sufficient statistics for batch processing."""
  if n_total_samples == 0:
    return means, covariances, jnp.zeros_like(nk)

  weights = nk / n_total_samples

  # Add safeguard for components with no assigned data points
  safe_nk = jnp.where(nk == 0, 1.0, nk)

  updated_means = xk / safe_nk[..., None]
  means = jnp.array(
    jnp.where(nk[..., None] > 0, updated_means, means),
  )
  if covariance_type == "full":
    updated_covs = sk / safe_nk[..., None, None]
    updated_covs = updated_covs - jnp.einsum("...i,...j->...ij", means, means)
    updated_covs += reg_covar * jnp.eye(means.shape[-1])

    original_covs_squeezed = jnp.squeeze(covariances)
    covariances_3d = jnp.where(nk[..., None, None] > 0, updated_covs, original_covs_squeezed)
    covariances_final = covariances_3d[None, ...]
  elif covariance_type == "diag":
    updated_vars = sk / safe_nk[..., None] - means**2
    updated_vars += reg_covar

    original_vars_squeezed = jnp.squeeze(covariances)
    variances_2d = jnp.where(nk[..., None] > 0, updated_vars, original_vars_squeezed)
    covariances_final = variances_2d[None, ..., None]

  return weights, means, covariances_final


def fit_gmm_in_memory(
  data: jax.Array,
  gmm: GMM,
  covariance_type: Literal["full", "diag"] = "full",
  max_iter: int = 100,
  tol: float = 1e-3,
  reg_covar: float = 1e-6,
) -> EMFitterResult:
  """Fit a GMM to in-memory data using the EM algorithm."""

  @jax.jit
  def em_step_fn(state: _EMLoopState) -> _EMLoopState:
    """Run a single EM step."""
    log_likelihood, log_resp = _e_step(data, state.gmm.means, state.gmm.covariances)
    means, covariances, weights = _m_step_from_responsibilities(
      data,
      state.gmm.means,
      state.gmm.covariances,
      state.gmm.weights,
      jnp.exp(log_resp),
      reg_covar,
      covariance_type,
    )
    return _EMLoopState(
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

  def em_cond_fn(state: _EMLoopState) -> jax.Array:
    """Stop condition for the EM loop."""
    return (state.n_iter < max_iter) & (state.log_likelihood_diff >= tol)

  initial_state = _EMLoopState(
    gmm=gmm,
    n_iter=0,
    log_likelihood=jnp.asarray(-jnp.inf),
    log_likelihood_diff=jnp.asarray(jnp.inf),
  )

  final_state = jax.lax.while_loop(em_cond_fn, em_step_fn, initial_state)
  final_state = jax.block_until_ready(final_state)

  return EMFitterResult(
    gmm=final_state.gmm,
    n_iter=final_state.n_iter,
    log_likelihood=final_state.log_likelihood,
    log_likelihood_diff=final_state.log_likelihood_diff,
    converged=final_state.log_likelihood_diff < tol,
  )


def fit_gmm_generator(
  data_generator: Generator[jax.Array, None, None],
  gmm: GMM,
  n_total_samples: int,
  max_iter: int = 100,
  tol: float = 1e-3,
  reg_covar: float = 1e-6,
  covariance_type: str = "full",
) -> EMFitterResult:
  """Fit a GMM to data from a generator (for large, out-of-memory datasets)."""
  log_likelihood_prev = -jnp.inf
  converged = False
  n_iter = 0
  log_likelihood = jnp.asarray(-jnp.inf)
  log_likelihood_diff = jnp.asarray(jnp.inf)

  data_cache = list(data_generator)

  if not data_cache or n_total_samples == 0:
    return EMFitterResult(
      gmm=gmm,
      n_iter=0,
      log_likelihood=jnp.asarray(-jnp.inf),
      log_likelihood_diff=jnp.asarray(jnp.inf),
      converged=True,
    )

  for i in range(max_iter):
    n_iter = i + 1
    nk = jnp.zeros(gmm.n_components)
    xk = jnp.zeros((gmm.n_components, gmm.n_features))
    if covariance_type == "full":
      sk = jnp.zeros((gmm.n_components, gmm.n_features, gmm.n_features))
    elif covariance_type == "diag":
      sk = jnp.zeros((gmm.n_components, gmm.n_features))
    else:
      msg = f"Unsupported covariance type: {covariance_type}"
      raise ValueError(msg)
    log_likelihood_total = 0.0

    # E-step: Accumulate statistics over all batches from the cache
    for batch_data in data_cache:
      batch_size = batch_data.shape[0]
      batch_ll, log_resp = _e_step(batch_data, gmm.means, gmm.covariances)
      resp = jnp.exp(log_resp)
      resp = jnp.squeeze(resp, axis=(-2, -1))  # Squeeze to 2D

      log_likelihood_total += batch_ll * batch_size
      nk += jnp.sum(resp, axis=Axis.batch)
      xk += jnp.einsum("ij,ik->jk", resp, batch_data)
      if covariance_type == "full":
        sk += jnp.einsum("ij,ik,il->jkl", resp, batch_data, batch_data)
      elif covariance_type == "diag":
        sk += jnp.einsum("ij,ik->jk", resp, batch_data**2)

    # M-step: Update GMM parameters using accumulated stats
    gmm = _m_step_from_stats(
      gmm.means,
      gmm.covariances,
      nk,
      xk,
      sk,
      n_total_samples,
      reg_covar,
      covariance_type,
    )

    log_likelihood = jnp.asarray(log_likelihood_total / n_total_samples)
    log_likelihood_diff = jnp.abs(log_likelihood - log_likelihood_prev)

    if log_likelihood_diff < tol:
      converged = True
      break
    log_likelihood_prev = log_likelihood

  return EMFitterResult(
    gmm=gmm,
    n_iter=n_iter,
    log_likelihood=log_likelihood,
    log_likelihood_diff=log_likelihood_diff,
    converged=converged,
  )
