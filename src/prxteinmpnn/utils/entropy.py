"""Utilities for calculating entropy from a collection of logits.

Adapted from github.com/justktln2/cimist

"""

import jax.numpy as jnp
from jax import Array, jit
from jax.lax import digamma, polygamma
from jax.lax.linalg import eigh
from jax.scipy.special import entr
from jax.typing import ArrayLike
from jaxtyping import Float

from prxteinmpnn.utils.types import Logits


@jit
def von_neumman(rho: Float) -> ArrayLike:
  r"""Compute the von Neumman entropy of the density overlap matrix rho.

  With a square matrix $\rho$ defined by $\rho_{ij} = \\sqrt{p(i)}\\sqrt{p(j)}$,
  the von Neumann entropy is given by $S_{vn} = -\\sum_{j} \\lambda_j \\log \\lambda_j$.

  Attributes:
    rho(Float):

  """
  _, lambda_ = eigh(rho)

  return jnp.sum(jnp.where(lambda_ > 0, entr(lambda_), 0.0))


def mle_entropy(states: Logits) -> ArrayLike:
  r"""Compute the maximum likelihood or "plugin" estimator of the entropy.

  The maximum likelihood estimator is given by
  $$
  \\hat{S}_{MLE} = -\\sum_{i} \frac{n_i}\frac{N} \\log \frac{n_i}\frac{N}.
  $$


  """
  n = jnp.array(states).flatten()
  p = n / states.sum()
  return entr(p).sum()


### Core functions on which Bayesian estimators all depend.
def posterior_entropy_mean(alpha: Float) -> ArrayLike:
  r"""Calculate expected entropy of a categorical distribution $p \\sim Dirichlet(\alpha)$.

  References
  ----------
  [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research.
    15(81):2833-2868, 2014. doi:10.5555/2627435.2697056

  Notes
  -----
  Equation 18 of appendix A.1 of ref [1].

  """
  alpha_sum = jnp.sum(alpha)
  return digamma(alpha_sum + 1) - jnp.sum(alpha * digamma(alpha + 1)) / alpha_sum


def posterior_entropy_squared_mean(alpha: Array) -> ArrayLike:
  r"""Calculate expected squared entropy of a categorical distribution $p \\sim Dirichlet(\alpha)$.

  References
  ----------
  [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research.
    15(81):2833-2868, 2014. doi:10.5555/2627435.2697056

  Notes
  -----
  See equation 19 in appendix A.2 of ref [1].

  """
  alpha_sum = jnp.sum(alpha)

  diagonal = (
    (digamma(alpha + 2) - digamma(alpha_sum + 2)) ** 2
    + polygamma(1.0, jnp.asarray(alpha + 2, dtype=jnp.float32))
    - polygamma(1.0, jnp.asarray(alpha_sum + 2, dtype=jnp.float32))
  )
  diag_sum = jnp.dot(alpha * (alpha + 1), diagonal)
  off_diagonal = digamma(alpha + 1) - digamma(alpha_sum + 2)
  full = jnp.outer(off_diagonal, off_diagonal) - polygamma(1.0, alpha_sum + 2)
  alpha_outer = jnp.outer(alpha, alpha)
  off_diag_sum = jnp.sum(full * alpha_outer) - jnp.sum(jnp.diag(full * alpha_outer))
  return (diag_sum + off_diag_sum) / (alpha_sum * (alpha_sum + 1))


def posterior_entropy_moments(alpha: Array) -> Array:
  """Calculate Bayesian posterior entropy moments.

  References
  ----------
  [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research.
    15(81):2833-2868, 2014. doi:10.5555/2627435.2697056

  Notes
  -----
  See equations 18-19 in appendix A of reference [1].

  """
  alpha = alpha.flatten()
  return jnp.array([posterior_entropy_mean(alpha), posterior_entropy_squared_mean(alpha)])


@jit
def posterior_mean_std(alpha: Array) -> Array:
  r"""Calculate the mean and standard deviation of the distribution of the entropy.

  For a categorical distribution $p \\sim Dirichlet(\alpha)$.

  Returns
  -------
  A 2x1 array with the posterior mean as the first entropy and the posterior standard deviation as
    the second.

  References
  ----------
  [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research.
    15(81):2833-2868, 2014. doi:10.5555/2627435.2697056

  Notes
  -----
  See equations 18-19 in appendix A of reference [1].

  """
  mean_entropy, mean_squared_entropy = posterior_entropy_moments(alpha)
  return jnp.array([mean_entropy, jnp.sqrt(mean_squared_entropy - mean_entropy**2)])
