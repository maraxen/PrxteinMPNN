"""The Alanine guard: freezing a position without choosing its identity must raise.

`fixed_mask` selects WHICH positions are frozen; `fixed_tokens` says TO WHAT. Setting the first
without the second used to silently lock every frozen position to token 0 -- Alanine -- because
decode does `final_token = where(is_group_fixed, fixed_tokens, sampled)`
(decode/autoregressive.py:340-346) and `fixed_tokens` defaulted to zeros. Range validation never
caught it: token 0 is a legal amino acid, not a sentinel, and the check sat inside
`if fixed_tokens is not None`, so it could not fire for the default that needed catching.

Consequence had it shipped: a caller "fixing the catalytic triad" gets Ala/Ala/Ala -- a dead
enzyme -- with valid-shaped output and no error.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _prepare_fixed_controls, resolve_native_tokens
from aminx.run.specs import SamplingSpecification
from aminx.utils.aa_convert import AF_ALPHABET, MPNN_ALPHABET
from aminx.utils.data_structures import Protein

_SEQ_LEN = 8
_TRIAD = (2, 4, 6)

# Fixtures are **AF**, because that is what `aatype` actually is. This is not a detail: the
# previous fixtures were written in MPNN -- `[0, 5, 6, 5, 2, 5, 1, 5]` with a comment reading
# "position 6 -> C", which is true in MPNN (index 1 = Cys) and false in AF (index 1 = Arg).
# The author assumed aatype was MPNN, encoded that assumption into the fixture, and the test
# then agreed with them. A synthetic fixture can only ever confirm its author's belief about
# the alphabet, which is why the alphabet-sensitive assertions now live in
# tests/utils/test_alphabet_boundary.py against a real parse. These fixtures stay synthetic
# because what they test -- whether the guard raises -- is alphabet-agnostic; they just have
# to stop lying about what they contain.
def _af(*residues: str) -> np.ndarray:
  """Build an AF-encoded aatype row from residue letters, so the fixture reads as biology."""
  return np.array([AF_ALPHABET.index(r) for r in residues], dtype=np.int32)


# TEV's catalytic triad at _TRIAD, Gln elsewhere as filler.
_AF_HDC = _af("A", "Q", "H", "Q", "D", "Q", "C", "Q")[None, :]
# The C151A mutant: identical except the triad Cys is Ala -- the real 1LVB case.
_AF_HDA = _af("A", "Q", "H", "Q", "D", "Q", "A", "Q")[None, :]


def _protein(batch_size: int = 1, aatype: np.ndarray | None = None) -> Protein:
  if aatype is None:
    aatype = np.zeros((batch_size, _SEQ_LEN), dtype=np.int32)
  return Protein(
    coordinates=jnp.zeros((batch_size, _SEQ_LEN, 4, 3), dtype=jnp.float32),
    aatype=jnp.asarray(aatype),
    atom_mask=jnp.ones((batch_size, _SEQ_LEN, 37), dtype=jnp.float32),
    residue_index=jnp.arange(_SEQ_LEN, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
    chain_index=jnp.zeros((batch_size, _SEQ_LEN), dtype=jnp.int32),
    mask=jnp.ones((batch_size, _SEQ_LEN), dtype=jnp.float32),
    mapping=None,
  )


def _triad_mask() -> np.ndarray:
  mask = np.zeros(_SEQ_LEN, dtype=np.float32)
  mask[list(_TRIAD)] = 1.0
  return mask


def _spec(**kw: object) -> SamplingSpecification:
  return SamplingSpecification(inputs=["x.pdb"], **kw)


class TestGuardFires:
  """The guard must reject the dangerous input. Verified by constructing it deliberately."""

  def test_fixed_mask_without_fixed_tokens_raises(self) -> None:
    spec = _spec(fixed_mask=_triad_mask())
    with pytest.raises(ValueError, match="fixed_tokens is None") as exc:
      _prepare_fixed_controls(spec, batched_ensemble=_protein())
    # The message must be actionable, not merely correct: it has to name the destructive
    # default AND both escape routes, or the next caller re-derives this from scratch.
    text = str(exc.value)
    assert "Alanine" in text or "'A'" in text
    assert "fixed_tokens explicitly" in text
    assert "resolve_native_tokens" in text

  def test_fixed_positions_without_fixed_tokens_raises(self) -> None:
    """fixed_positions is a mask alias, so it must arm the same guard."""
    spec = _spec(fixed_positions=_triad_mask())
    with pytest.raises(ValueError, match="fixed_tokens is None"):
      _prepare_fixed_controls(spec, batched_ensemble=_protein())

  def test_guard_does_not_fire_when_nothing_is_frozen(self) -> None:
    """An all-zero mask freezes nothing; there is no identity to choose."""
    spec = _spec(fixed_mask=np.zeros(_SEQ_LEN, dtype=np.float32))
    fm, ft = _prepare_fixed_controls(spec, batched_ensemble=_protein())
    assert not bool(jnp.any(fm))
    assert not bool(jnp.any(ft))


class TestUniformOverride:
  """The common intent: freeze the triad at the SAME identity in every state."""

  def test_1d_fixed_tokens_locks_one_identity_across_all_states(self) -> None:
    tokens = np.zeros(_SEQ_LEN, dtype=np.int32)
    tokens[2], tokens[4], tokens[6] = 6, 2, 1  # H, D, C in MPNN_ALPHABET

    spec = _spec(fixed_mask=_triad_mask(), fixed_tokens=tokens)
    fm, ft = _prepare_fixed_controls(spec, batched_ensemble=_protein(batch_size=4))

    assert fm.shape == (4, _SEQ_LEN)
    for state in range(4):
      assert [int(ft[state, p]) for p in _TRIAD] == [6, 2, 1], (
        "a 1-D fixed_tokens must broadcast the same identity to every state"
      )
      assert [float(fm[state, p]) for p in _TRIAD] == [1.0, 1.0, 1.0]

  def test_the_bug_this_guard_prevents(self) -> None:
    """Regression: the frozen positions must NOT come back as Alanine."""
    tokens = np.zeros(_SEQ_LEN, dtype=np.int32)
    tokens[2], tokens[4], tokens[6] = 6, 2, 1

    spec = _spec(fixed_mask=_triad_mask(), fixed_tokens=tokens)
    _fm, ft = _prepare_fixed_controls(spec, batched_ensemble=_protein())

    assert [int(ft[0, p]) for p in _TRIAD] != [0, 0, 0], "frozen triad decayed to Ala/Ala/Ala"


class TestNativeOverride:
  """resolve_native_tokens: freeze each position at whatever is already there."""

  def test_resolves_native_and_satisfies_the_guard(self) -> None:
    ensemble = _protein(aatype=_AF_HDC)

    tokens = resolve_native_tokens(ensemble, _triad_mask())
    spec = _spec(fixed_mask=_triad_mask(), fixed_tokens=tokens)
    _fm, ft = _prepare_fixed_controls(spec, batched_ensemble=ensemble)

    # AF in, MPNN out -- the RESIDUES are preserved, the encoding changes. Asserting the
    # residues rather than the raw ints is the whole point: `tokens == aatype` was the old
    # assertion, and it can only hold in one alphabet while naming neither.
    assert [MPNN_ALPHABET[int(ft[0, p])] for p in _TRIAD] == ["H", "D", "C"]
    assert [int(ft[0, p]) for p in _TRIAD] == [6, 2, 1]

  def test_divergent_natives_raise_naming_the_positions(self) -> None:
    """The C151A case: states disagree at a frozen position, so 'native' is undefined.

    The design is ONE sequence. Silently picking a state is how a dead enzyme ships.
    """
    aatype = np.array(
      [
        *_AF_HDC,  # position 6 -> C
        *_AF_HDA,  # position 6 -> A  (the C151A mutant)
      ],
      dtype=np.int32,
    )
    with pytest.raises(ValueError, match="structures disagree") as exc:
      resolve_native_tokens(_protein(batch_size=2, aatype=aatype), _triad_mask())

    text = str(exc.value)
    assert "position 6" in text, "the divergent position must be named, not just counted"
    assert "fixed_tokens explicitly" in text
    assert "allow_heterogeneous" in text

  def test_divergence_accepted_only_when_asked_for(self) -> None:
    aatype = np.concatenate([_AF_HDC, _AF_HDA], axis=0)
    tokens = resolve_native_tokens(
      _protein(batch_size=2, aatype=aatype), _triad_mask(), allow_heterogeneous=True,
    )
    assert MPNN_ALPHABET[int(tokens[0, 6])] == "C", (
      "structure 0's residue is taken, deliberately -- and it is Cys, not the mutant's Ala"
    )

  def test_agreeing_natives_need_no_override(self) -> None:
    aatype = np.repeat(_AF_HDC, 3, axis=0)
    tokens = resolve_native_tokens(_protein(batch_size=3, aatype=aatype), _triad_mask())
    assert [MPNN_ALPHABET[int(tokens[0, p])] for p in _TRIAD] == ["H", "D", "C"]
