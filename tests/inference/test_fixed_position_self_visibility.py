"""A teacher-forced position must not read its own identity through the ar_mask diagonal.

WHY THIS EXISTS
===============
Every ar_mask constructor in this library sets the diagonal. `generate_ar_mask`'s untied
branch is `row_indices >= col_indices` (non-strict); `generate_wave_ar_mask` returns
`earlier_wave | (same_wave & same_group)`, which is trivially true when i == j. Only
`full_context_ar_mask` is diagonal-free.

That is CORRECT for a designable position, whose own slot holds a token that has not been
drawn yet -- reading the placeholder is the intended sampling behaviour.

It is WRONG for a FIXED position. `sample_autoregressive` bakes `fixed_tokens` into
`init_sequence` before the wave scan begins, so a fixed position's slot holds its true
identity from the start, and `decoder.py` gates the SEQUENCE edge features by
`take_along_axis(ar_mask, neighbor_indices)` -- a residue being always among its own KNN
neighbours. The fixed position therefore reads its own answer through the diagonal alone,
independent of decoding order, and the contamination reaches designable neighbours through
the decoder layers. `scoring/score.py` carried this same defect class until it moved to
`full_context_ar_mask`; there it measured +0.036243 nats on 1LVB chain A (paired over 8
seeds, t = 41.3), always over-confident.

A downstream project hit this on both its probe path and its production wave path and had
to hand-patch `* (1 - eye)` at every call site, which over-corrects: it also strips the
legitimate placeholder self-view at designable positions.

WHAT IS ASSERTED
================
The correction is surgical. It must remove self-visibility exactly at fixed positions,
preserve it everywhere else, leave the off-diagonal structure bit-identical, and be an exact
no-op when nothing is fixed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.utils.autoregression import full_context_ar_mask, generate_ar_mask

L = 12


def _minimal_inputs(seq_len: int = L):
  """Smallest inputs build_inference_bundle accepts, with distinguishable geometry."""
  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(seq_len, 4, 3)) * 3.0, dtype=jnp.float32)
  mask = jnp.ones((seq_len,), dtype=jnp.float32)
  residue_index = jnp.arange(seq_len, dtype=jnp.int32)
  chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
  return coords, mask, residue_index, chain_index


def _ar_mask_of(built) -> np.ndarray:
  """build_inference_bundle returns (InferenceBundle, InferenceConfig).

  The resolved mask lives at `bundle.conditioning.ar_mask`, shaped (S, L, L).
  """
  bundle = built[0] if isinstance(built, tuple) else built
  return np.asarray(bundle.conditioning.ar_mask)


def _build(fixed_mask, *, correct: bool = True):
  coords, mask, residue_index, chain_index = _minimal_inputs()
  out = build_inference_bundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    ar_mask=generate_ar_mask(jnp.arange(L, dtype=jnp.int32)),
    fixed_mask=fixed_mask,
    fixed_tokens=jnp.zeros((L,), dtype=jnp.int32),
    correct_fixed_self_visibility=correct,
  )
  return _ar_mask_of(out)


def test_generate_ar_mask_still_sets_the_diagonal():
  """Pin the upstream behaviour the correction exists to compensate for.

  If this ever fails, `generate_ar_mask` changed and the correction below may have become
  redundant -- which should be re-examined, not left silently double-applied.
  """
  m = np.asarray(generate_ar_mask(jnp.arange(L, dtype=jnp.int32)))
  assert int(np.diag(m).sum()) == L


def test_full_context_ar_mask_is_diagonal_free():
  """The scoring remedy is a DIFFERENT object: non-causal, full context minus self."""
  m = np.asarray(full_context_ar_mask(L))
  assert int(np.diag(m).sum()) == 0
  # Non-causal: it is not lower-triangular, which is exactly why it must not be substituted
  # for a causal decode.
  assert not np.array_equal(m, np.tril(np.ones((L, L), dtype=m.dtype), k=-1))


def test_fixed_positions_lose_self_visibility():
  fixed_idx = [2, 5, 7]
  fm = np.zeros(L, dtype=np.float32)
  fm[fixed_idx] = 1.0
  got = _build(jnp.asarray(fm))
  diag = np.diag(got[0]) if got.ndim == 3 else np.diag(got)
  assert diag[fixed_idx].sum() == 0, (
    f"fixed positions still read their own slot: diagonal={diag.tolist()}"
  )


def test_designable_positions_keep_their_placeholder_self_view():
  """Over-correction guard: a whole-diagonal zero would break intended sampling semantics."""
  fixed_idx = [2, 5, 7]
  fm = np.zeros(L, dtype=np.float32)
  fm[fixed_idx] = 1.0
  got = _build(jnp.asarray(fm))
  m = got[0] if got.ndim == 3 else got
  designable = [i for i in range(L) if i not in fixed_idx]
  assert np.diag(m)[designable].sum() == len(designable), (
    "designable positions lost their own-slot placeholder view; that is the intended "
    "sampling behaviour, not part of the defect"
  )


def test_off_diagonal_structure_is_untouched():
  """The correction is surgical -- it must not re-derive or perturb the causal structure."""
  fm = np.zeros(L, dtype=np.float32)
  fm[[2, 5, 7]] = 1.0
  corrected = _build(jnp.asarray(fm))
  uncorrected = _build(jnp.asarray(fm), correct=False)
  a = corrected[0] if corrected.ndim == 3 else corrected
  b = uncorrected[0] if uncorrected.ndim == 3 else uncorrected
  off = ~np.eye(L, dtype=bool)
  assert np.array_equal(a[off], b[off]), "off-diagonal entries changed"
  # And it must be non-degenerate: an all-zero mask would also pass a diagonal check while
  # removing ALL context, a strictly worse defect.
  assert a.sum() > 0


def test_no_op_when_nothing_is_fixed():
  """Backward compatibility: callers that fix no positions see identical behaviour."""
  fm = jnp.zeros((L,), dtype=jnp.float32)
  corrected = _build(fm)
  uncorrected = _build(fm, correct=False)
  assert np.array_equal(corrected, uncorrected)


def test_opt_out_reproduces_pre_fix_behaviour():
  """The escape hatch exists so banked pre-fix numbers remain reproducible ON PURPOSE.

  It must be explicit, never the default -- the default is correctness.
  """
  fm = np.zeros(L, dtype=np.float32)
  fm[[1, 4]] = 1.0
  uncorrected = _build(jnp.asarray(fm), correct=False)
  m = uncorrected[0] if uncorrected.ndim == 3 else uncorrected
  assert int(np.diag(m).sum()) == L, "opt-out did not restore the original set diagonal"


@pytest.mark.parametrize("n_states", [1, 3])
def test_correction_applies_across_the_state_axis(n_states):
  """Multistate bundles broadcast ar_mask to (S, L, L); the fix must reach every state."""
  fm = np.zeros(L, dtype=np.float32)
  fm[[0, 3]] = 1.0
  coords, mask, residue_index, chain_index = _minimal_inputs()
  if n_states > 1:
    coords = jnp.broadcast_to(coords[None], (n_states, *coords.shape))
    mask = jnp.broadcast_to(mask[None], (n_states, *mask.shape))
    residue_index = jnp.broadcast_to(residue_index[None], (n_states, *residue_index.shape))
    chain_index = jnp.broadcast_to(chain_index[None], (n_states, *chain_index.shape))
  out = build_inference_bundle(
    coords=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    ar_mask=generate_ar_mask(jnp.arange(L, dtype=jnp.int32)),
    fixed_mask=jnp.asarray(fm),
    fixed_tokens=jnp.zeros((L,), dtype=jnp.int32),
  )
  m = _ar_mask_of(out)
  m3 = m if m.ndim == 3 else m[None]
  for s in range(m3.shape[0]):
    assert np.diag(m3[s])[[0, 3]].sum() == 0, f"state {s} still self-visible at fixed positions"
