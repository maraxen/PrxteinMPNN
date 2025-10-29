"""Bayesian Information Criterion (BIC) computation for model selection.

This module provides functionality to compute the Bayesian Information Criterion (BIC)
for Gaussian Mixture Models and other statistical models. BIC is used for model selection
by balancing model fit (log-likelihood) against model complexity (number of parameters).

The BIC formula is: BIC = -2 * log_likelihood + k * log(n)
where k is the number of parameters and n is the number of samples.

For Gaussian Mixture Models, the number of parameters depends on:
- Mean parameters: n_components * n_features
- Covariance parameters: depends on covariance_type ("full" or "diag")
- Weight parameters: n_components - 1 (due to sum-to-one constraint)

Lower BIC values indicate better models, making it useful for selecting the optimal
number of components in mixture models.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import BIC, LogLikelihood


@partial(jax.jit, static_argnames=("covariance_type",))
def compute_bic(
  log_likelihood: LogLikelihood,
  n_samples: int,
  n_components: int,
  n_features: int,
  covariance_type: Literal["full", "diag"] = "full",
) -> BIC:
  """Compute the Bayesian Information Criterion (BIC) for model selection.

  BIC penalizes model complexity to prevent overfitting. Lower BIC values indicate
  better models. The penalty term accounts for the number of free parameters in the model.

  Args:
    log_likelihood: Log-likelihood of the data under the model.
    n_samples: Number of data samples.
    n_components: Number of mixture components.
    n_features: Number of features in the data.
    covariance_type: Type of covariance matrix, either "full" or "diag".

  Returns:
    Array: The BIC score (lower is better).

  Example:
    >>> bic = compute_bic(-1000.0, 500, 3, 10, "diag")
    >>> print(bic)

  """
  n_mean_params = n_components * n_features

  if covariance_type == "full":
    n_cov_params = n_components * n_features * (n_features + 1) // 2
  else:
    n_cov_params = n_components * n_features

  n_weight_params = n_components - 1

  n_params = n_mean_params + n_cov_params + n_weight_params

  return jnp.asarray(-2 * log_likelihood + n_params * jnp.log(n_samples))
