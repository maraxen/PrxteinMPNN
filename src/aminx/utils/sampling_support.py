"""The support rule every residue draw in aminx shares: X is never chosen.

Upstream LigandMPNN softmaxes all 21 columns and then samples only the first 20
(``model_utils.py``, "hard omit X"), so X is never drawn. PottsMPNN is not uniform:
its energy-based design sites slice to the first 20 letters ("Gap and X should
never be chosen", ``run_utils.py``), but its autoregressive decoder
(``potts_mpnn_utils.py`` ``decoder``) excludes X only when the caller lists it in
``omit_AAs``, and the shipped example config leaves that list empty. A draw that
does not apply this mask can emit the unknown token at a designed position, which
is what debt #2660 found.

Use ``exclude_x_logits`` on the logits right before each draw. The AST registry
in ``tests/sampling/test_x_draw_registry.py`` fails when a new draw site appears
without one.
"""

from __future__ import annotations

import jax.numpy as jnp

from aminx.utils.aa_convert import MPNN_ALPHABET

#: Index of the unknown token in the 21-letter MPNN alphabet (``ACDEFGHIKLMNPQRSTVWYX``).
X_INDEX = MPNN_ALPHABET.index("X")


def exclude_x_logits(logits: jnp.ndarray, x_index: int = X_INDEX) -> jnp.ndarray:
  """Return ``logits`` with the X column set to ``-inf`` so no draw can select it.

  Works on any leading shape; only the last axis is the 21-letter vocabulary.
  ``-inf`` (not a large negative constant) makes the probability exactly zero, and
  it leaves the argmax of ``logits + gumbel`` unchanged for every other column.
  Use this on logits before ``jax.random.categorical`` or a Gumbel argmax.
  """
  return logits.at[..., x_index].set(-jnp.inf)


def exclude_x_probs(probs: jnp.ndarray, x_index: int = X_INDEX) -> jnp.ndarray:
  """Return the unnormalised weights ``probs`` with the X column set to exactly zero.

  Use this where weights are built by exponentiating something that has been clipped
  (a clipped ``-inf`` becomes a small finite weight, so a logit mask would leak X). The
  caller normalises afterwards.
  """
  return probs.at[..., x_index].set(0.0)
