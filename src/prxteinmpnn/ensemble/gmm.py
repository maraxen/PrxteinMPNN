"""Fit GMM from logits.

Adapted from https://github.com/justktln2/ciMIST.
"""

from collections.abc import Callable
from functools import partial

from gmmx import EMFitter, GaussianMixtureModelJax

from prxteinmpnn.utils.types import Logits

gmm = GaussianMixtureModelJax.create(n_components=100, n_features=21)

GMMFitFn = Callable[[Logits], GaussianMixtureModelJax]


def make_fit_gmm(n_components: int = 100, n_features: int = 21) -> GMMFitFn:
  """Make the Expectation Maximization Gaussian Model Mixture Fit function.

  Args:
    n_components: number of components GMM
    n_features: number of features for GMM

  """
  gmm = GaussianMixtureModelJax.create(n_components=n_components, n_features=n_features)
  em_fitter = EMFitter(tol=1e-3, max_iter=100)
  return partial(em_fitter.fit, gmm=gmm)
