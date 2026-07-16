"""Pin the two alphabet facts every other module silently assumes.

aminx carries two amino-acid alphabets and marks neither at any boundary:

- **AF** ``ARNDCQEGHILKMFPSTWYVX`` -- what proxide's parsers put in ``aatype``.
- **MPNN** ``ACDEFGHIKLMNPQRSTVWYX`` -- the model's token space (logits, probs, sampled
  tokens, ``fixed_tokens``).

They agree only at index 0 (``A``) and 20 (``X``); ~15 of 21 differ. **Every value is legal
in both**, so no range check, dtype check, or shape check can ever detect a confusion -- it
produces plausible garbage, never an error. Two guards in this repo prove the point by
failing to catch anything: ``align.py:563``'s ``jnp.clip(seq, 0, 20)`` and tev_design's
``if native_aa < 0 or native_aa >= N_AA``. Both pass for every value in both alphabets.

``tests/utils/test_aa_convert.py`` already tests the permutation, and the permutation has
never been the bug. It tests ``af_to_mpnn`` against itself and never asserts that anything
*produces* AF or *consumes* MPNN -- so the crossings had zero coverage while five separate
crossing bugs shipped. **These tests cover the crossings instead.**

Why a distributional check and not a value check: you cannot look at an array and tell which
alphabet it is in. But you *can* compare against ground truth. A permuted sequence cannot
fake native recovery -- it lands at chance. That signal needs no type system, survives a
rename, and reaches across the repo boundary into tev_design, which has none of aminx's
types. It makes a bad crossing impossible to *ship*, which is weaker than impossible to
*write*, but five recurrences have already shipped, so shipping is the binding constraint.

**Real parses only, never synthetic fixtures.** A fixture hand-written in the alphabet its
author assumed can only ever confirm that assumption. That is precisely how the
``resolve_native_tokens`` bug hid inside the commit whose purpose was preventing it.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.utils.aa_convert import AF_ALPHABET, MPNN_ALPHABET, af_to_mpnn

_DATA = Path(__file__).resolve().parents[1] / "data"
_UBQ = _DATA / "1ubq.pdb"

# 1UBQ's sequence, from the deposition -- external ground truth, not derived from this
# codebase. The whole point is to check aminx against something aminx did not produce.
_UBQ_SEQUENCE = (
  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

# Native recovery for ProteinMPNN on a monomer sits near 0.5-0.6; chance over 21 tokens is
# ~0.05, and a permuted sequence measures ~0.09-0.12 (the permutation is not random -- A and
# X are fixed points -- so it beats pure chance slightly). These thresholds sit in the wide
# empty gap between the two regimes, not near either.
_RECOVERY_FLOOR = 0.40
_CHANCE_CEILING = 0.25


def _decode(aatype: np.ndarray, alphabet: str) -> str:
  return "".join(alphabet[int(i)] if 0 <= int(i) < len(alphabet) else "?" for i in aatype)


def _first_protein(parsed: object) -> object:
  return parsed if hasattr(parsed, "aatype") else next(iter(parsed))  # type: ignore[call-overload]


# ---------------------------------------------------------------------------------------
# Fact 1: the parsers emit AF. Both ingresses -- there are two, and they are easy to miss.
# ---------------------------------------------------------------------------------------


@pytest.mark.alphabet_boundary
def test_parse_structure_emits_af_not_mpnn() -> None:
  """``aminx.io.parsing.parse_structure`` yields AF-ordered ``aatype``.

  Asserts BOTH directions. "AF decodes correctly" alone would still pass if the two
  alphabets happened to agree on this sequence; requiring MPNN to *fail* proves the test
  discriminates rather than passing for free.
  """
  from aminx.io.parsing import parse_structure  # noqa: PLC0415

  aatype = np.asarray(_first_protein(parse_structure(str(_UBQ))).aatype)  # type: ignore[attr-defined]

  assert _decode(aatype, AF_ALPHABET) == _UBQ_SEQUENCE, (
    "parse_structure's aatype no longer decodes to 1UBQ's real sequence under the AF "
    "alphabet. Either the parser's convention changed (in which case every af_to_mpnn call "
    "in aminx and tev_design is now wrong and must be revisited), or parsing broke."
  )
  assert _decode(aatype, MPNN_ALPHABET) != _UBQ_SEQUENCE, (
    "aatype decodes to the correct sequence under BOTH alphabets, so this test cannot tell "
    "them apart and proves nothing. Pick a sequence where they differ."
  )


@pytest.mark.alphabet_boundary
def test_training_ingress_emits_af_not_mpnn() -> None:
  """``create_protein_dataset`` -- the *other* ingress -- also yields AF.

  This is not redundant with the test above. ``host/prep.py:102`` and
  ``training/trainer.py:17`` call ``proxide.ops.dataset.create_protein_dataset`` directly and
  **never touch** ``aminx/io/parsing/dispatch.py``. An audit that checked only the dispatch
  wrapper concluded aminx had a single parse chokepoint; it has two, and the one it missed is
  the production inference and training path.
  """
  from proxide.ops.dataset import create_protein_dataset  # noqa: PLC0415

  batch = next(iter(create_protein_dataset([str(_UBQ)], batch_size=1)))
  aatype = np.asarray(batch.aatype).reshape(-1)[: len(_UBQ_SEQUENCE)]

  assert _decode(aatype, AF_ALPHABET) == _UBQ_SEQUENCE, (
    "The training/inference ingress no longer emits AF. trainer.py feeds this array "
    "straight into the cross-entropy loss against MPNN-ordered logits, so a change here "
    "silently changes what every model is trained against."
  )
  assert _decode(aatype, MPNN_ALPHABET) != _UBQ_SEQUENCE


# ---------------------------------------------------------------------------------------
# Fact 2: the model consumes MPNN. This is the gate.
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ubq_structure() -> dict:
  from aminx.io.parsing import parse_structure  # noqa: PLC0415

  protein = _first_protein(parse_structure(str(_UBQ)))
  return {
    "aatype": jnp.asarray(np.asarray(protein.aatype), dtype=jnp.int32),  # type: ignore[attr-defined]
    "coordinates": jnp.asarray(np.asarray(protein.coordinates)),  # type: ignore[attr-defined]
    "mask": jnp.asarray(np.asarray(protein.mask)),  # type: ignore[attr-defined]
    "residue_index": jnp.asarray(np.asarray(protein.residue_index)),  # type: ignore[attr-defined]
    "chain_index": jnp.asarray(np.asarray(protein.chain_index)),  # type: ignore[attr-defined]
  }


def _conditional_logits(structure: dict, sequence: jnp.ndarray) -> jnp.ndarray:
  """Conditional logits for ``sequence`` against ``structure``, via the real factory.

  Uses ``make_conditional_logits_fn`` deliberately: it is the exact factory
  ``host/runner.py:779`` uses for the ``conditional_logits`` feature, so tests built on this
  measure the shipped path rather than a reconstruction of it.
  """
  from aminx.io.weights import load_model  # noqa: PLC0415
  from aminx.sampling.conditional_logits import make_conditional_logits_fn  # noqa: PLC0415

  model = load_model("proteinmpnn_v_48_020")
  logits_fn = make_conditional_logits_fn(model)
  logits = jnp.asarray(
    logits_fn(
      jax.random.PRNGKey(0),
      structure["coordinates"],
      structure["mask"],
      structure["residue_index"],
      structure["chain_index"],
      sequence,
    ),
  )
  if logits.ndim == 3:  # noqa: PLR2004 -- (1, L, 21) vs (L, 21)
    logits = logits[0]
  return logits


def _native_recovery(structure: dict, sequence: jnp.ndarray) -> float:
  """Fraction of positions where the model's argmax equals the sequence it was given.

  Self-consistency, deliberately: "did the model agree with what I told it?" A model asked
  to score a sequence it was handed in the wrong alphabet cannot agree with it, and no
  amount of structural signal rescues that. Compare
  ``test_inspect_conditions_on_the_converted_sequence``, which documents why the *other*
  reading of recovery -- against the true native -- is insensitive here.
  """
  logits = _conditional_logits(structure, sequence)
  return float((jnp.argmax(logits, axis=-1) == sequence).mean())


@pytest.mark.alphabet_boundary
def test_resolve_native_tokens_returns_mpnn() -> None:
  """``resolve_native_tokens`` must return MPNN, because decode substitutes it as MPNN.

  Real parse, deliberately. ``tests/host/test_fixed_tokens_guard.py`` covers this function
  well for *raising* behaviour, which is alphabet-agnostic, using synthetic fixtures -- and
  that is fine. What no test covered was the **alphabet of the return value**, and a
  synthetic fixture could never have covered it: the fixture author would have written the
  aatype in whichever alphabet they already believed, and the test would have agreed with
  them. That is exactly how this bug shipped in 026403d.

  The check is against 1UBQ's real residues: freeze position 0 and the function must report
  the token for Met, in MPNN. Under the old raw-AF return it reported AF's Met (12), which
  MPNN reads as Asn.
  """
  from aminx.host._sampling_helper import resolve_native_tokens  # noqa: PLC0415
  from aminx.io.parsing import parse_structure  # noqa: PLC0415

  protein = _first_protein(parse_structure(str(_UBQ)))
  aatype = np.asarray(protein.aatype)[None, :]  # type: ignore[attr-defined]

  class _Ensemble:
    pass

  ensemble = _Ensemble()
  ensemble.aatype = jnp.asarray(aatype)  # type: ignore[attr-defined]

  fixed_mask = np.zeros(aatype.shape[1], dtype=np.float32)
  fixed_mask[0] = 1.0  # 1UBQ position 0 is Met

  tokens = np.asarray(resolve_native_tokens(ensemble, fixed_mask))  # type: ignore[arg-type]

  assert MPNN_ALPHABET[int(tokens[0, 0])] == "M", (
    f"resolve_native_tokens returned token {int(tokens[0, 0])}, which is "
    f"'{MPNN_ALPHABET[int(tokens[0, 0])]}' in MPNN -- but 1UBQ position 0 is Met. The "
    f"return value feeds fixed_tokens, which decode substitutes directly against "
    f"MPNN-sampled tokens, so it must be MPNN."
  )
  # The whole sequence, not just the frozen position: the function returns a full-length
  # array and every entry has to be in the same alphabet.
  assert _decode(tokens[0], MPNN_ALPHABET) == _UBQ_SEQUENCE


@pytest.mark.alphabet_boundary
@pytest.mark.requires_weights
@pytest.mark.slow
def test_model_consumes_mpnn_not_af(_ubq_structure: dict) -> None:
  """The model's token space is MPNN: converting raises recovery from chance to ~0.6.

  **This single test would have caught all five known recurrences** -- resolve_native_tokens,
  runner.inspect/jacobian, tev_design's native_rank, _token_name, and the trainer. It is the
  only mechanism available that keys on something real rather than on a name or a type: a
  permuted sequence cannot fake native recovery.

  Both assertions are load-bearing. The floor catches a broken conversion. The ceiling proves
  the measurement has power -- without it, a metric that returned a constant 0.6 would pass
  forever and silently certify nothing. (See ~/.claude/rules/BATHOS.md: verify the
  measurement against synthetic invariants before trusting it. That rule was written after a
  temperature-inversion bug burned two hours of hypothesis testing.)
  """
  af_native = _ubq_structure["aatype"]
  mpnn_native = af_to_mpnn(af_native).astype(jnp.int32)

  converted = _native_recovery(_ubq_structure, mpnn_native)
  raw = _native_recovery(_ubq_structure, af_native)

  assert converted > _RECOVERY_FLOOR, (
    f"Native recovery through af_to_mpnn is {converted:.3f}, at or near chance. Either the "
    f"model no longer consumes MPNN, af_to_mpnn broke, or the parser's convention changed. "
    f"Whichever it is, every conversion call site in aminx and tev_design is now suspect."
  )
  assert raw < _CHANCE_CEILING, (
    f"Feeding the model raw AF aatype scores {raw:.3f}, which is NOT at chance. That should "
    f"be impossible while the two alphabets differ, so this test has lost its power to "
    f"discriminate and would pass even with the conversion removed. Do not relax this "
    f"threshold to make it green -- find out why the permutation stopped mattering."
  )


@pytest.mark.alphabet_boundary
@pytest.mark.requires_weights
@pytest.mark.slow
def test_inspect_conditions_on_the_converted_sequence(_ubq_structure: dict) -> None:
  """``inspect()``'s ``conditional_logits`` must condition on MPNN, matching the factory.

  **This is an exact-equality wiring check, deliberately, and the first version of it was a
  false green.** The obvious test -- "assert inspect()'s recovery beats chance" -- does not
  work here, and the reason is worth writing down.

  ``conditional_logits`` sees the structure *and* the sequence, and the structure dominates.
  Measured on 1UBQ with real proteinmpnn_v_48_020 weights:

  ==========================  ===========================  =========================
  conditioning fed            recovery vs *true* native    recovery vs *fed* sequence
  ==========================  ===========================  =========================
  MPNN (correct)              0.592                        0.592
  raw AF (the bug)            0.487                        0.118
  ==========================  ===========================  =========================

  So feeding a permuted sequence degrades recovery 0.592 -> 0.487. It does **not** collapse
  to chance, and a threshold test at any defensible floor sails straight past it. The widely
  quoted "0.09, i.e. chance" figure is the *right-hand* column -- self-consistency against
  whatever was fed -- not evidence that the feature returns noise.

  That does not make the bug benign: the logits answer a different question than the caller
  asked, so any downstream NLL or per-residue analysis built on them is wrong. It makes the
  bug **undetectable by the metric everyone reaches for first**, which is exactly why it sat
  in a shipped CLI command for five weeks (since 2119537, 2026-06-08).

  Hence exact equality against the factory called with the converted sequence. It pins the
  one thing at issue -- *which sequence runner hands over* -- and it fails the moment the
  conversion is removed. ``jacobian()`` (``runner.py:1022``) and ``decoded_node_features``
  (``:840``) share the identical defect; this covers the cheapest to exercise.
  """
  from aminx.host.runner import inspect  # noqa: PLC0415

  native_mpnn = af_to_mpnn(_ubq_structure["aatype"]).astype(jnp.int32)
  length = int(native_mpnn.shape[0])

  reference = _conditional_logits(_ubq_structure, native_mpnn)
  wrong = _conditional_logits(_ubq_structure, _ubq_structure["aatype"])

  result = inspect(
    inputs=[str(_UBQ)],
    inspection_features=["conditional_logits"],
    checkpoint_id="proteinmpnn_v_48_020",
  )
  actual = np.asarray(result["conditional_logits"][0])[:length]  # inspect pads to max_length

  assert np.allclose(actual, np.asarray(reference), atol=1e-4), (
    "inspect()'s conditional_logits do not match the logits produced by conditioning on "
    "af_to_mpnn(aatype). runner is handing the model a sequence in the wrong alphabet, so "
    "the feature answers a different question than the caller asked. This is a shipped CLI "
    "command (`aminx run inspect`)."
  )
  # Proves the check above can fail: the two conditionings really do produce different
  # logits. Without this, a bug that made both paths identically wrong would pass silently.
  assert not np.allclose(np.asarray(reference), np.asarray(wrong), atol=1e-4), (
    "Conditioning on AF and on MPNN produced identical logits, so this test cannot "
    "distinguish them and proves nothing. Do not delete it -- find out why the sequence "
    "input stopped mattering."
  )
