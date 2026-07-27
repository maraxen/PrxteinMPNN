"""Alphabet-boundary guardrail for the ProteinEBM ``sequence_embedding`` path.

**Why this file exists.** The PR #130 accuracy regression was an AA-alphabet
permutation: the eval harness fed ProteinMPNN-ordered residue indices into the
AlphaFold2-ordered ``sequence_embedding`` table (``model.py:187``), selecting
the wrong embedding row for ~15/21 residues. It survived every existing gate
because none of them pin the fact that the EBM consumes **AF-order**:

* ``scripts/ebm/checkpoint_parity_check.py`` feeds identical *random* ints to
  both the JAX port and the PyTorch reference -- both agree on any integer->row
  mapping, so it is alphabet-agnostic by construction.
* ``tests/ebm/test_ddg_stability.py::TestRealCheckpointDestabilizingMutationSanity``
  asserts a disruptive-vs-conservative ``|ddG diff| > 2.0``. That is a
  *degeneracy* check (two distinct embedding rows score differently) -- it is
  satisfied under the WRONG alphabet too, so it cannot discriminate ordering.
* Every other EBM test uses random ints, all-zeros, or asserts no accuracy.

**The discriminator used here.** A signal that discriminates alphabet
correctness must be *sequence-embedding-meaning-dependent at fixed structure*
-- not a structure-quality signal (energy-vs-RMSD) or a two-row degeneracy
signal, both of which partly survive a scrambled alphabet. The clean probe is
**native-sequence preference**: on a fixed native backbone, a trained EBM
assigns lower energy to the correct native sequence (AF-order) than to the
same residues fed in the WRONG order (``af_to_mpnn(native)`` -- exactly the
PR #130 bug), and lower than random shuffles of the native sequence. When the
alphabet is wrong, that preference collapses -- the mis-ordered sequence looks
like just another random sequence. This is the "assert-correct AND assert-
wrong-fails" discrimination pattern of ``tests/utils/test_alphabet_boundary.py``
(the ProteinMPNN-side analogue), applied to the EBM side.

Skipped unless the real E3.5-ported orbax checkpoint is present (not committed;
see ``scripts/ebm/checkpoint_parity_check.py``). Marked ``slow`` +
``requires_weights`` -- deselected by default (``addopts`` in pyproject), run
explicitly (``-m "slow and requires_weights"``) where the checkpoint lives.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.ebm.model import ProteinEBMModel
from aminx.utils.aa_convert import AF_ALPHABET, af_to_mpnn

# --- Checkpoint plumbing (mirrors tests/ebm/test_decoy_ranking.py) ------------
_TEST_DATA = Path(__file__).resolve().parent.parent / "data"
_UBIQUITIN_PDB = _TEST_DATA / "1ubq.pdb"
_ORBAX_CHECKPOINT_DIR = Path("/tmp/proteinebm_weights/ported_jax_model")
_REAL_CHECKPOINT_AVAILABLE = (_ORBAX_CHECKPOINT_DIR / "0" / "model").exists()

# The shipped ProteinEBM-x checkpoint's own dimensions (aminx.ebm.checkpoint).
_CKPT_TOKEN_S = 256
_CKPT_TOKEN_Z = 128
_CKPT_DIM_FOURIER = 256
_CKPT_TRANSITION_LAYERS = 2
_CKPT_DEPTH = 16
_CKPT_HEADS = 8
_CKPT_NUM_CONTACT = 3

_PINNED_T = 0.05  # MVP noise-time target (design spec §9), same as the harnesses.
_N_RANDOM_SHUFFLES = 24


def _restore_ported_model() -> ProteinEBMModel:
  import orbax.checkpoint as ocp  # noqa: PLC0415 -- dev-only import, mirrors checkpoint scripts

  template = ProteinEBMModel(
    token_s=_CKPT_TOKEN_S,
    token_z=_CKPT_TOKEN_Z,
    dim_fourier=_CKPT_DIM_FOURIER,
    conditioning_transition_layers=_CKPT_TRANSITION_LAYERS,
    transformer_depth=_CKPT_DEPTH,
    transformer_heads=_CKPT_HEADS,
    num_contact_embeddings=_CKPT_NUM_CONTACT,
    key=jax.random.PRNGKey(0),
  )
  options = ocp.CheckpointManagerOptions(max_to_keep=1)
  manager = ocp.CheckpointManager(
    str(_ORBAX_CHECKPOINT_DIR),
    options=options,
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  restored = manager.restore(0, items={"model": template})
  return restored["model"]


def _load_native_af(pdb_name: str) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Parse a local PDB into (scaled CA coords, AF-order aatype, mask).

  ``parse_structure`` emits ``Protein.aatype`` in AF order -- the exact order
  ``sequence_embedding`` expects -- so NO alphabet remap is applied here (the
  correct path). See ``tests/ebm/test_decoy_ranking.py::_load_ca_structure``.
  """
  from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING  # noqa: PLC0415
  from aminx.io.parsing import parse_structure  # noqa: PLC0415 -- heavy import, keep lazy
  from proxide.chem.residues import atom_order  # noqa: PLC0415

  protein = parse_structure(str(_TEST_DATA / pdb_name))
  ca_angstrom = np.asarray(protein.coordinates[:, atom_order["CA"], :], dtype=np.float32)
  coords_scaled = jnp.asarray(ca_angstrom * DEFAULT_COORDINATE_SCALING)
  aatype_af = jnp.asarray(np.asarray(protein.aatype, dtype=np.int32))
  mask = jnp.asarray(np.asarray(protein.mask, dtype=bool))
  return coords_scaled, aatype_af, mask


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.alphabet_boundary
@pytest.mark.skipif(
  not _REAL_CHECKPOINT_AVAILABLE,
  reason=(
    f"Real E3.5-ported orbax checkpoint not found at {_ORBAX_CHECKPOINT_DIR} "
    "(not committed; see scripts/ebm/checkpoint_parity_check.py)."
  ),
)
class TestEbmConsumesAfAlphabet:
  """On real weights, the EBM must prefer the AF-order native sequence.

  Reproduces the PR #130 bug (``af_to_mpnn(native)`` fed to the AF-ordered
  embedding) and asserts the trained model's native-sequence preference is
  present under AF-order and destroyed under MPNN-order.
  """

  @pytest.fixture(scope="class")
  def model(self) -> ProteinEBMModel:
    return _restore_ported_model()

  @pytest.fixture(scope="class")
  def native(self) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _load_native_af("1ubq.pdb")

  def _energy(
    self,
    model: ProteinEBMModel,
    coords: jax.Array,
    aatype: jax.Array,
    mask: jax.Array,
  ) -> float:
    # contacts=None now resolves to the reference default (ones for num=3) via
    # trunk_features -- the Track-B change; the discrimination is unaffected by
    # the constant contact row, since it is identical across all sequences here.
    return float(model.energy(coords, aatype, jnp.asarray(_PINNED_T), mask))

  def test_native_af_scores_lower_than_wrong_alphabet_and_random(
    self,
    model: ProteinEBMModel,
    native: tuple[jax.Array, jax.Array, jax.Array],
    capsys: pytest.CaptureFixture[str],
  ) -> None:
    coords, aatype_af, mask = native

    # Correct convention: AF-order native (what parse_structure emits).
    e_af = self._energy(model, coords, aatype_af, mask)

    # The exact PR #130 bug: MPNN-ordered indices fed to the AF-ordered table.
    # af_to_mpnn re-labels each residue, i.e. a fixed permutation of the native
    # sequence -- a different (non-native) sequence on the same backbone.
    aatype_wrong = jnp.asarray(af_to_mpnn(aatype_af), dtype=jnp.int32)
    e_wrong = self._energy(model, coords, aatype_wrong, mask)

    # Random-shuffle null distribution: permute the native residue labels.
    rng = np.random.default_rng(0)
    native_np = np.asarray(aatype_af)
    e_random = np.array(
      [
        self._energy(model, coords, jnp.asarray(rng.permutation(native_np), dtype=jnp.int32), mask)
        for _ in range(_N_RANDOM_SHUFFLES)
      ],
    )

    # Diagnostic (printed, not a hidden gate) -- mirrors the neighbouring
    # real-checkpoint tests that document observed numbers for transparency.
    rand_mean = float(e_random.mean())
    rand_std = float(e_random.std())
    z_native = (e_af - rand_mean) / rand_std if rand_std > 0 else float("nan")
    z_wrong = (e_wrong - rand_mean) / rand_std if rand_std > 0 else float("nan")
    with capsys.disabled():
      print(
        f"\n[alphabet_boundary/ebm] E(native AF)={e_af:.4f}  "
        f"E(wrong MPNN)={e_wrong:.4f}  random μ={rand_mean:.4f} σ={rand_std:.4f}  "
        f"z_native={z_native:+.2f}  z_wrong={z_wrong:+.2f}  "
        f"rand_frac_below_native={float((e_random < e_af).mean()):.3f}",
      )

    # (1) Reproduce-the-bug discrimination: the correct AF-order native must
    #     score strictly lower than the mis-ordered (PR #130) sequence.
    assert e_af < e_wrong, (
      f"AF-order native energy {e_af:.4f} is not below the wrong-alphabet "
      f"energy {e_wrong:.4f} -- the EBM is not consuming AF-order as required."
    )

    # (2) Native preference: the AF-order native must sit in the low tail of the
    #     random-shuffle null (a trained EBM prefers the native sequence). A
    #     conservative gate -- native below the 25th percentile of randoms --
    #     rules out a degenerate/alphabet-blind pipeline without being fragile.
    assert e_af < float(np.percentile(e_random, 25.0)), (
      f"AF-order native energy {e_af:.4f} is not in the low quartile of the "
      f"random-shuffle null (25th pct = {np.percentile(e_random, 25.0):.4f}); "
      "the model shows no native-sequence preference under AF-order."
    )

    # (3) Assert-wrong-fails: the mis-ordered sequence must NOT be preferred --
    #     it should look like a random shuffle, not a native-low outlier.
    assert e_wrong >= float(np.percentile(e_random, 25.0)), (
      f"wrong-alphabet energy {e_wrong:.4f} landed in the native-low quartile "
      f"(25th pct = {np.percentile(e_random, 25.0):.4f}); the discriminator is "
      "not separating AF-order from MPNN-order -- investigate before trusting."
    )

  def test_af_alphabet_decodes_native_ubiquitin_prefix(
    self,
    native: tuple[jax.Array, jax.Array, jax.Array],
  ) -> None:
    """Cross-check that the parsed AF-order aatype really is the native sequence.

    Guards the premise of the discrimination test above: decoding the AF-order
    aatype via ``AF_ALPHABET`` must reproduce ubiquitin's known N-terminal
    sequence (``MQIFVK...``). If ``parse_structure`` ever silently changed
    convention, this fails loudly instead of quietly weakening test (1)-(3).
    """
    _coords, aatype_af, _mask = native
    decoded = "".join(AF_ALPHABET[int(i)] for i in np.asarray(aatype_af)[:6])
    assert decoded == "MQIFVK", f"expected ubiquitin prefix MQIFVK, decoded {decoded!r}"
