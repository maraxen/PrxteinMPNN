"""Expectation-Maximization algorithm for Gaussian Mixture Models.

Adapted from gmmx (https://github.com/google/gmmx)
with modifications to support fitting from data generators, such as
HDF5 datasets.
"""

from collections.abc import Generator
from enum import Enum
from typing import NamedTuple

import jax
from flax.struct import dataclass
from gmmx import GaussianMixtureModelJax
from jax import numpy as jnp


class Axis(int, Enum):
  """Internal axis order."""

  batch = 0
  components = 1
  features = 2
  features_covar = 3


class _EMLoopState(NamedTuple):
  """State for the in-memory EM loop."""

  gmm: GaussianMixtureModelJax
  n_iter: int
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array


@dataclass
class EMFitterResult:
  """Result of the Expectation-Maximization fitting process.

  Attributes
  ----------
  gmm : GaussianMixtureModelJax
      The final fitted Gaussian mixture model.
  n_iter : int
      The total number of iterations performed.
  log_likelihood : jax.Array
      The log-likelihood of the data under the final model.
  log_likelihood_diff : jax.Array
      The difference in log-likelihood from the previous iteration.
  converged : bool
      A boolean indicating if the algorithm converged within the max iterations.

  """

  gmm: GaussianMixtureModelJax
  n_iter: int
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array
  converged: bool


def _e_step(
  x: jax.Array,
  gmm: GaussianMixtureModelJax,
) -> tuple[jax.Array, jax.Array]:
  """Run Expectation (E) step of the EM algorithm.

  Args:
      x: Feature vectors batch.
      gmm: The current Gaussian mixture model.

  Returns:
      A tuple containing the mean log-likelihood for the batch and the log responsibilities.

  """
  log_prob = gmm.log_prob(x)
  log_prob_norm = jax.scipy.special.logsumexp(
    log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  log_resp = log_prob - log_prob_norm
  return jnp.mean(log_prob_norm), log_resp


def _m_step_from_responsibilities(
  x: jax.Array,
  resp: jax.Array,
  gmm: GaussianMixtureModelJax,
  reg_covar: float,
) -> GaussianMixtureModelJax:
  """Maximization (M) step for in-memory data using responsibilities."""
  return gmm.from_responsibilities(
    x,
    resp,
    reg_covar=reg_covar,
    covariance_type=gmm.covariances.type,
  )


def _m_step_from_stats(
  gmm: GaussianMixtureModelJax,
  nk: jax.Array,
  xk: jax.Array,
  sk: jax.Array,
  n_total_samples: int,
  reg_covar: float,
) -> GaussianMixtureModelJax:
  """Update GMM parameters from accumulated sufficient statistics for batch processing."""
  weights = nk / n_total_samples
  means = xk / nk[..., None]

  # Compute covariances and apply regularization
  covariances = sk / nk[..., None, None]
  covariances = covariances - jnp.einsum("...i,...j->...ij", means, means)
  covariances += reg_covar * jnp.eye(gmm.n_features)

  return gmm.replace(  # type: ignore[call-arg]
    weights=weights.flatten(),
    means=means,
    covariances=gmm.covariances.replace(values=covariances),  # type: ignore[call-arg]
  )


def fit_gmm_in_memory(
  x: jax.Array,
  initial_gmm: GaussianMixtureModelJax,
  max_iter: int = 100,
  tol: float = 1e-3,
  reg_covar: float = 1e-6,
) -> EMFitterResult:
  """Fit a GMM to in-memory data using the EM algorithm."""
  x_reshaped = x[None, ...]

  @jax.jit
  def em_step_fn(state: _EMLoopState) -> _EMLoopState:
    """Run a single EM step."""
    log_likelihood, log_resp = _e_step(x_reshaped, state.gmm)
    gmm = _m_step_from_responsibilities(x_reshaped, jnp.exp(log_resp), state.gmm, reg_covar)
    return _EMLoopState(
      gmm=gmm,
      n_iter=state.n_iter + 1,
      log_likelihood=log_likelihood,
      log_likelihood_diff=jnp.abs(log_likelihood - state.log_likelihood),
    )

  def em_cond_fn(state: _EMLoopState) -> jax.Array:
    """Stop condition for the EM loop."""
    return (state.n_iter < max_iter) & (state.log_likelihood_diff >= tol)

  initial_state = _EMLoopState(
    gmm=initial_gmm,
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
    converged=final_state.n_iter < max_iter,
  )


def fit_gmm_generator(
  data_generator: Generator[jax.Array, None, None],
  initial_gmm: GaussianMixtureModelJax,
  n_total_samples: int,
  max_iter: int = 100,
  tol: float = 1e-3,
  reg_covar: float = 1e-6,
) -> EMFitterResult:
  """Fit a GMM to data from a generator (for large, out-of-memory datasets)."""
  gmm = initial_gmm
  log_likelihood_prev = -jnp.inf
  converged = False
  n_iter = 0
  log_likelihood = jnp.asarray(-jnp.inf)
  log_likelihood_diff = jnp.asarray(jnp.inf)

  for _ in range(max_iter):
    nk = jnp.zeros(gmm.n_components)
    xk = jnp.zeros((gmm.n_components, gmm.n_features))
    sk = jnp.zeros((gmm.n_components, gmm.n_features, gmm.n_features))
    log_likelihood_total = 0.0

    # E-step: Accumulate statistics over all batches
    for batch_data in data_generator:
      batch_size = batch_data.shape[0]
      batch_ll, log_resp = _e_step(batch_data, gmm)
      resp = jnp.exp(log_resp)

      log_likelihood_total += batch_ll * batch_size
      nk += jnp.sum(resp, axis=Axis.batch)
      xk += jnp.einsum("ij,ik->jk", resp, batch_data)
      sk += jnp.einsum("ij,ik,il->jkl", resp, batch_data, batch_data)

    # M-step: Update GMM parameters using accumulated stats
    gmm = _m_step_from_stats(gmm, nk, xk, sk, n_total_samples, reg_covar)

    log_likelihood = jnp.asarray(log_likelihood_total / n_total_samples)
    log_likelihood_diff = jnp.abs(log_likelihood - log_likelihood_prev)

    if log_likelihood_diff < tol:
      converged = True
      break
    log_likelihood_prev = log_likelihood

  return EMFitterResult(
    gmm=gmm,
    n_iter=n_iter + 1,
    log_likelihood=log_likelihood,
    log_likelihood_diff=log_likelihood_diff,
    converged=converged,
  )