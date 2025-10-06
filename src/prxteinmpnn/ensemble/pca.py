"""Principal Component Analysis (PCA) utilities."""

import jax
import pcax


def pca_transform(
  data: jax.Array,
  n_components: int,
) -> jax.Array:
  """Create a PCA transformer.

  Args:
    data: Input data, shape (num_samples, num_features).
    n_components: Number of principal components to keep.

  Returns:
    A PCA transformer object.

  """
  return pcax.transform(pcax.fit(data, n_components=n_components), data)
