"""Utilities for autoregression.

prxteinmpnn.utils.autoregression
"""

from .types import AtomMask, DecodingOrder


def generate_ar_mask(decoding_order: DecodingOrder) -> AtomMask:
  """Get the autoregressive mask for the given decoding order.

  Args:
    decoding_order: The order in which atoms are decoded.

  Returns:
    An atom mask where each atom can only attend to itself and previous atoms.

  """
  row_indices = decoding_order[:, None]
  col_indices = decoding_order[None, :]
  return (row_indices >= col_indices).astype(int)
