"""Fixed-site conditioning for Potts models: folding fixed neighbours into the fields gives the exact conditional.

Ground truth is brute-force enumeration of every free configuration. The negative control shows that plain masking
(which deletes a site) is NOT conditioning, so the test identifies the fold rather than passing vacuously.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.potts.sampling import (
    _conditional_logits_site,
    _potts_log_unnormalized,
    condition_on_fixed_sites,
)


def _system(rng: np.random.Generator, n: int, q: int):
  h = rng.normal(size=(n, q))
  j = rng.normal(size=(n, n, q, q))
  j = 0.5 * (j + np.transpose(j, (1, 0, 3, 2)))
  w = np.triu((rng.random((n, n)) < 0.7).astype(float), 1)
  return h, j, w + w.T


def _logpsi(h, j, w, seq, mask) -> float:
  args = (h, j, w, seq, mask)
  dtypes = (jnp.float32, jnp.float32, jnp.float32, jnp.int32, jnp.float32)
  return float(_potts_log_unnormalized(*(jnp.asarray(a, d) for a, d in zip(args, dtypes, strict=True))))


@pytest.mark.parametrize(("n", "q", "n_fixed", "seed"), [(7, 3, 3, 0), (6, 4, 2, 1), (7, 3, 1, 2)])
def test_fold_matches_exact_conditional(n: int, q: int, n_fixed: int, seed: int) -> None:
  """Folded log-psi equals the unmasked joint up to a constant, for every free configuration."""
  rng = np.random.default_rng(seed)
  h, j, w = _system(rng, n, q)
  fixed = np.zeros(n)
  fixed[rng.choice(n, size=n_fixed, replace=False)] = 1.0
  seq0 = rng.integers(0, q, size=n)
  hp, free = condition_on_fixed_sites(
      jnp.asarray(h, jnp.float32), jnp.asarray(j, jnp.float32), jnp.asarray(w, jnp.float32),
      jnp.asarray(seq0, jnp.int32), jnp.asarray(fixed, jnp.float32),
  )
  free_idx = np.nonzero(fixed == 0)[0]
  assert np.array_equal(np.asarray(free), 1.0 - fixed)
  full, folded, unfolded = [], [], []
  for cfg in itertools.product(range(q), repeat=len(free_idx)):
    s = seq0.copy()
    s[free_idx] = cfg
    full.append(_logpsi(h, j, w, s, np.ones(n)))
    folded.append(_logpsi(np.asarray(hp), j, w, s, np.asarray(free)))
    unfolded.append(_logpsi(h, j, w, s, np.asarray(free)))

  def deviation(approx: list[float]) -> float:
    d = np.asarray(full) - np.asarray(approx)
    return float(np.max(np.abs(d - d.mean())))

  assert deviation(folded) < 1e-4
  assert deviation(unfolded) > 1e-2  # negative control: masking alone deletes the fixed sites, it does not condition


def test_gibbs_conditional_uses_fold() -> None:
  """The single-site Gibbs conditional on folded fields equals the exact conditional; unfolded masking does not."""
  rng = np.random.default_rng(3)
  n, q = 7, 3
  h, j, w = _system(rng, n, q)
  fixed = np.array([1, 0, 1, 0, 0, 1, 0], dtype=float)
  w[0, 1] = w[1, 0] = w[2, 1] = w[1, 2] = 1.0  # guarantee the probed site 1 has fixed neighbours
  seq = rng.integers(0, q, size=n)
  hp, free = condition_on_fixed_sites(
      jnp.asarray(h, jnp.float32), jnp.asarray(j, jnp.float32), jnp.asarray(w, jnp.float32),
      jnp.asarray(seq, jnp.int32), jnp.asarray(fixed, jnp.float32),
  )
  i = 1
  truth = np.array([_logpsi(h, j, w, np.where(np.arange(n) == i, a, seq), np.ones(n)) for a in range(q)])
  truth = np.exp(truth - truth.max())
  truth /= truth.sum()

  def probs(field) -> np.ndarray:
    lg = np.asarray(_conditional_logits_site(
        jnp.asarray(field, jnp.float32), jnp.asarray(j, jnp.float32), jnp.asarray(w, jnp.float32),
        jnp.asarray(seq, jnp.int32), free, jnp.int32(i)))
    p = np.exp(lg - lg.max())
    return p / p.sum()

  assert np.max(np.abs(probs(hp) - truth)) < 1e-5
  assert np.max(np.abs(probs(h) - truth)) > 1e-2


def test_deleted_sites_do_not_condition() -> None:
  """A site with mask == 0 stays deleted even if flagged fixed: it contributes no folded field."""
  rng = np.random.default_rng(4)
  n, q = 5, 3
  h, j, w = _system(rng, n, q)
  seq = rng.integers(0, q, size=n)
  fixed = jnp.asarray([1, 0, 0, 0, 0], jnp.float32)
  mask = jnp.asarray([0, 1, 1, 1, 1], jnp.float32)
  hp, free = condition_on_fixed_sites(
      jnp.asarray(h, jnp.float32), jnp.asarray(j, jnp.float32), jnp.asarray(w, jnp.float32),
      jnp.asarray(seq, jnp.int32), fixed, mask,
  )
  np.testing.assert_allclose(np.asarray(hp), h, atol=1e-6)
  assert np.array_equal(np.asarray(free), np.asarray(mask))
