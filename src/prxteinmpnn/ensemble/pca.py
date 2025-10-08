"""Principal Component Analysis (PCA) utilities."""

from functools import partial
from typing import Literal

import jax
import pcax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.utils.types import PCAInputData


@partial(jax.jit, static_argnames="n_components")
def pca_transform(
  data: PCAInputData,
  n_components: int,
  solver: Literal["full", "randomized"] = "full",
  rng: PRNGKeyArray | None = None,
) -> tuple[jax.Array, pcax.pca.PCAState]:
  """Create a PCA transformer.

  Args:
    data: Input data, shape (num_samples, num_features).
    n_components: Number of principal components to keep.
    solver: PCA solver to use, either "full" or "randomized".
    rng: Optional JAX random key for randomized solver.

  Returns:
    A PCA transformer object.

  """
  pca_state = pcax.fit(data, n_components=n_components, solver=solver, rng=rng)
  return pcax.transform(pca_state, data), pca_state
