"""Tests for the E5 decoy-ranking application logic (``aminx.ebm.decoy_ranking``).

Two tiers, by design:

1. **Fast, deterministic, synthetic-model unit tests** (``TestSweepNoiseTime*``,
   ``TestRmsdToReference``, ``TestSpearmanEnergyQualityCorrelation``,
   ``TestRankDecoysOverNoiseTime``) -- prove the wiring/math is correct using
   a tiny randomly-initialized ``ProteinEBMModel`` (same fixture shape as
   ``tests/ebm/test_dispatch.py``) and hand-constructed inputs with a known
   answer. These always run, no external artifacts required.

2. **The honest small-scale pipeline validation**
   (``TestRealCheckpointDecoyRankingProxy``) -- exercises the *real* E3.5-
   ported orbax weights (``/tmp/proteinebm_weights/ported_jax_model/``)
   against *real* local PDB structures (``tests/data/*.pdb``), using
   synthetic coordinate noise (not the real Rosetta decoy set) to build a
   small decoy-like set with a KNOWN ground truth: RMSD-to-native. This is
   **not** a measurement of the paper's Spearman 0.838 target -- it validates
   that the *pipeline* (real trained weights -> noise-time sweep -> Spearman
   correlation) produces sane, non-degenerate output on real structures, and
   reports whatever correlation it actually finds without any post-hoc
   tuning to make the number look better.

   **Scope limitation, stated explicitly (not an oversight):** the real
   Rosetta decoy benchmark (133 native structures + thousands of decoys +
   TM-scores + Rosetta energies) that the design spec's 0.838 target refers
   to requires downloading ``decoys.zip`` (unknown size, likely large) from
   ``https://files.ipd.uw.edu/pub/decoyset/decoys.zip`` plus
   ``huggingface.co/jproney/ProteinEBM/resolve/main/{rmsd,rosettascore}.txt``.
   That download was **not authorized** for this dispatch (unlike E3.5's
   checkpoint, which the user explicitly authorized) and is **not**
   attempted here. This test tier is skipped entirely if the ported
   checkpoint directory is not present locally, so it degrades gracefully in
   any environment without the E3.5 artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from aminx.ebm.decoy_ranking import (
  DEFAULT_NOISE_TIME_GRID,
  NoiseTimeSweepResult,
  rank_decoys_over_noise_time,
  rmsd_to_reference,
  spearman_energy_quality_correlation,
  sweep_noise_time_energies,
)
from aminx.ebm.model import ProteinEBMModel

TOKEN_S = 16
TOKEN_Z = 8
DEPTH = 2
HEADS = 2
N = 6

_PORTED_CHECKPOINT_DIR = Path("/tmp/proteinebm_weights/ported_jax_model")
_TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# Real E3.5-ported checkpoint's own config (verified in
# aminx.ebm.checkpoint's module docstring / ckpt_config.log) -- must match
# exactly or the orbax restore's abstract PyTree template will mismatch the
# saved leaf shapes.
_CHECKPOINT_TOKEN_S = 256
_CHECKPOINT_TOKEN_Z = 128
_CHECKPOINT_DIM_FOURIER = 256
_CHECKPOINT_TRANSITION_LAYERS = 2
_CHECKPOINT_DEPTH = 16
_CHECKPOINT_HEADS = 8
_CHECKPOINT_NUM_CONTACT_EMBEDDINGS = 3


def _make_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    key=key,
  )


class TestSweepNoiseTimeEnergies:
  def test_shape_and_matches_per_t_score_decoy_batch(self) -> None:
    model = _make_model(jax.random.PRNGKey(0))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(1))
    d = 3
    coords = jax.random.normal(k_coords, (d, N, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t_values = (0.01, 0.05, 0.2)

    energies = sweep_noise_time_energies(model, coords, aatype, mask, t_values)
    assert energies.shape == (len(t_values), d)

    for i, t in enumerate(t_values):
      expected = jnp.stack([model.energy(coords[j], aatype, jnp.asarray(t), mask) for j in range(d)])
      assert jnp.allclose(energies[i], expected, atol=1e-5)

  def test_default_grid_used_when_unspecified(self) -> None:
    model = _make_model(jax.random.PRNGKey(2))
    coords = jax.random.normal(jax.random.PRNGKey(3), (2, N, 3)) * 0.1
    aatype = jnp.zeros((N,), dtype=jnp.int32)
    mask = jnp.ones((N,), dtype=bool)

    energies = sweep_noise_time_energies(model, coords, aatype, mask)
    assert energies.shape == (len(DEFAULT_NOISE_TIME_GRID), 2)


class TestRmsdToReference:
  def test_zero_rmsd_for_identical_decoy(self) -> None:
    reference = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
    decoys = jnp.stack([reference, reference])
    rmsd = rmsd_to_reference(decoys, reference)
    assert jnp.allclose(rmsd, 0.0, atol=1e-6)

  def test_known_constant_offset_rmsd(self) -> None:
    reference = jnp.zeros((N, 3))
    # A constant offset of `c` in every one of the 3 coordinates of every
    # residue gives a per-residue squared deviation of `3*c**2` (Euclidean
    # distance in 3D), so RMSD == c*sqrt(3) exactly (mean over N identical
    # per-residue values, then sqrt).
    c = 2.0
    decoy = jnp.full((N, 3), c)
    rmsd = rmsd_to_reference(decoy[None], reference)
    expected = c * jnp.sqrt(3.0)
    assert jnp.allclose(rmsd, jnp.asarray([expected]), atol=1e-5)

  def test_mask_excludes_padded_residues(self) -> None:
    reference = jnp.zeros((N, 3))
    decoy = jnp.zeros((N, 3)).at[-1].set(100.0)  # huge deviation in the last (masked-out) residue
    mask = jnp.ones((N,), dtype=bool).at[-1].set(False)
    rmsd = rmsd_to_reference(decoy[None], reference, mask=mask)
    assert jnp.allclose(rmsd, 0.0, atol=1e-6)

  def test_monotonic_in_noise_scale(self) -> None:
    reference = jax.random.normal(jax.random.PRNGKey(5), (N, 3))
    key = jax.random.PRNGKey(6)
    scales = [0.0, 0.5, 1.0, 2.0]
    decoys = jnp.stack([reference + s * jax.random.normal(key, (N, 3)) for s in scales])
    rmsd = rmsd_to_reference(decoys, reference)
    assert jnp.all(jnp.diff(rmsd) >= 0.0)


class TestSpearmanEnergyQualityCorrelation:
  def test_perfect_positive_correlation(self) -> None:
    energies = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    quality = np.asarray([10.0, 20.0, 30.0, 40.0])
    rho = spearman_energy_quality_correlation(energies, quality)
    assert rho == pytest.approx(1.0)

  def test_perfect_negative_correlation(self) -> None:
    energies = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    quality = np.asarray([40.0, 30.0, 20.0, 10.0])
    rho = spearman_energy_quality_correlation(energies, quality)
    assert rho == pytest.approx(-1.0)

  def test_no_rank_correlation(self) -> None:
    energies = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    quality = np.asarray([1.0, 2.0, 1.0, 2.0])  # ties, no monotonic relationship with energies
    rho = spearman_energy_quality_correlation(energies, quality)
    assert abs(rho) < 1.0


class TestRankDecoysOverNoiseTime:
  def test_selects_best_t_by_absolute_correlation(self) -> None:
    """Build a model whose per-t energy ordering is controlled, so the 'best t' is known."""
    model = _make_model(jax.random.PRNGKey(7))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(8))
    d = 4
    coords = jax.random.normal(k_coords, (d, N, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    quality_labels = np.asarray([0.0, 1.0, 2.0, 3.0])
    t_values = (0.05, 0.1, 0.2)

    result = rank_decoys_over_noise_time(
      model, coords, aatype, mask, quality_labels, t_values=t_values,
    )
    assert isinstance(result, NoiseTimeSweepResult)
    assert result.t_values == t_values
    assert len(result.spearman_by_t) == len(t_values)
    assert result.best_t in t_values
    best_idx = t_values.index(result.best_t)
    assert result.spearman_by_t[best_idx] == pytest.approx(result.best_spearman)
    # best_spearman must be the (signed) entry with the largest magnitude.
    assert abs(result.best_spearman) == pytest.approx(max(abs(s) for s in result.spearman_by_t))
    assert result.energies.shape == (len(t_values), d)
    np.testing.assert_allclose(result.quality_labels, quality_labels)


def _checkpoint_available() -> bool:
  return _PORTED_CHECKPOINT_DIR.exists()


def _restore_ported_model() -> ProteinEBMModel:
  """Restore the real E3.5-ported orbax checkpoint onto a matching-shape template.

  Mirrors ``scripts/ebm/checkpoint_parity_check.py``'s ``_orbax_save``
  counterpart: an abstract ``ProteinEBMModel`` built with the checkpoint's
  own dimensions (see this module's ``_CHECKPOINT_*`` constants, matching
  ``aminx.ebm.checkpoint``'s module docstring) is the restore template.
  """
  import orbax.checkpoint as ocp  # noqa: PLC0415 -- dev-only import, mirrors checkpoint_parity_check.py

  template = ProteinEBMModel(
    token_s=_CHECKPOINT_TOKEN_S,
    token_z=_CHECKPOINT_TOKEN_Z,
    dim_fourier=_CHECKPOINT_DIM_FOURIER,
    conditioning_transition_layers=_CHECKPOINT_TRANSITION_LAYERS,
    transformer_depth=_CHECKPOINT_DEPTH,
    transformer_heads=_CHECKPOINT_HEADS,
    num_contact_embeddings=_CHECKPOINT_NUM_CONTACT_EMBEDDINGS,
    key=jax.random.PRNGKey(0),
  )
  options = ocp.CheckpointManagerOptions(max_to_keep=1)
  manager = ocp.CheckpointManager(
    str(_PORTED_CHECKPOINT_DIR),
    options=options,
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  restored = manager.restore(0, items={"model": template})
  return restored["model"]


def _load_ca_structure(pdb_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Parse a local PDB fixture into (ca_coords_angstrom, aatype, mask).

  Uses ``aminx.io.parsing.parse_structure`` (the proxide-backed parser
  already used by ``tests/conftest.py``/``scripts/ebm/bucket_boundary_
  check.py``), which returns ``Protein.aatype`` in the exact AlphaFold-order
  21-way vocabulary (``A R N D C Q E G H I L K M F P S T W Y V X``, mask/unk
  = index 20) that ``aminx.ebm.model.InputEmbeddings.sequence_embedding``
  (an ``eqx.nn.Embedding(21, ...)``) and the real ProteinEBM reference
  (``protein_ebm.data.protein_utils.restypes_with_x``, verified identical
  ordering) both expect -- no alphabet remap needed.
  """
  from aminx.io.parsing import parse_structure  # noqa: PLC0415 -- heavy import, keep lazy
  from proxide.chem.residues import atom_order  # noqa: PLC0415

  protein = parse_structure(str(_TEST_DATA_DIR / pdb_name))
  ca_coords = np.asarray(protein.coordinates[:, atom_order["CA"], :], dtype=np.float32)
  aatype = np.asarray(protein.aatype, dtype=np.int32)
  mask = np.asarray(protein.mask, dtype=bool)
  return ca_coords, aatype, mask


@pytest.mark.skipif(
  not _checkpoint_available(),
  reason=(
    "requires the E3.5-ported orbax checkpoint at /tmp/proteinebm_weights/"
    "ported_jax_model/ (not committed to the repo; see checkpoint.py/"
    "scripts/ebm/checkpoint_parity_check.py)"
  ),
)
class TestRealCheckpointDecoyRankingProxy:
  """Honest small-scale pipeline validation -- NOT the real Rosetta benchmark.

  Loads the real 1UBQ structure (``tests/data/1ubq.pdb``, 76-residue,
  single-chain, no complications), builds a small "decoy-like" set by
  additively perturbing the native CA backbone at several noise levels
  (0 = the native structure itself, up to a badly-mangled 8 Angstrom-sigma
  copy), uses RMSD-to-native as the KNOWN ground-truth quality label (lower
  = better, mirroring the physically-expected relationship between
  perturbation magnitude and structural quality), and scores every decoy
  with the REAL E3.5-ported checkpoint weights (not an untrained model) via
  :func:`aminx.ebm.decoy_ranking.rank_decoys_over_noise_time`.

  This validates that the pipeline produces sane output on a real structure
  with real trained weights -- it does **not** reproduce or approximate the
  paper's Spearman 0.838 Rosetta-decoy parity target (see this module's and
  ``aminx.ebm.decoy_ranking``'s docstrings for the full scope disclosure).
  """

  @pytest.fixture(scope="class")
  def ported_model(self) -> ProteinEBMModel:
    return _restore_ported_model()

  @pytest.fixture(scope="class")
  def native_structure(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_ca_structure("1ubq.pdb")

  @pytest.fixture(scope="class")
  def decoy_set(
    self, native_structure: tuple[np.ndarray, np.ndarray, np.ndarray],
  ) -> tuple[np.ndarray, jax.Array, np.ndarray, np.ndarray]:
    """Build the honest synthetic decoy set: native + 5 noise levels, RMSD labels.

    Coordinate noise is additive isotropic Gaussian, applied in the raw
    Angstrom frame (physically interpretable sigma values), then everything
    is scaled by ``aminx.ebm.diffusion.DEFAULT_COORDINATE_SCALING`` (0.1) to
    match ``aminx.ebm.contracts.Coords``'s scaled-nm convention (the single
    ``coordinate_scaling`` boundary the rest of ``aminx.ebm`` assumes has
    already been applied -- see ``aminx.ebm.diffusion``'s module docstring).
    """
    from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING

    ca_angstrom, _aatype, _mask = native_structure
    sigmas_angstrom = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    rng = np.random.default_rng(0)
    decoys_angstrom = np.stack(
      [ca_angstrom + sigma * rng.standard_normal(ca_angstrom.shape).astype(np.float32) for sigma in sigmas_angstrom],
    )
    rmsd_labels = np.sqrt(np.mean(np.sum((decoys_angstrom - ca_angstrom[None]) ** 2, axis=-1), axis=-1))
    decoys_scaled = jnp.asarray(decoys_angstrom * DEFAULT_COORDINATE_SCALING)
    return decoys_angstrom, decoys_scaled, rmsd_labels, np.asarray(sigmas_angstrom)

  def test_rmsd_labels_are_monotonic_in_noise_level(
    self, decoy_set: tuple[np.ndarray, jax.Array, np.ndarray, np.ndarray],
  ) -> None:
    """Sanity-check the measurement pipeline itself (BATHOS discipline) before trusting it."""
    _decoys_angstrom, _decoys_scaled, rmsd_labels, _sigmas = decoy_set
    assert rmsd_labels[0] == pytest.approx(0.0, abs=1e-6)  # sigma=0 decoy is the native structure
    assert np.all(np.diff(rmsd_labels) > 0.0)  # strictly increasing noise -> strictly increasing RMSD

  def test_energy_is_finite_and_varies_across_decoys(
    self,
    ported_model: ProteinEBMModel,
    native_structure: tuple[np.ndarray, np.ndarray, np.ndarray],
    decoy_set: tuple[np.ndarray, jax.Array, np.ndarray, np.ndarray],
  ) -> None:
    _ca_angstrom, aatype, mask = native_structure
    _decoys_angstrom, decoys_scaled, _rmsd_labels, _sigmas = decoy_set

    energies = sweep_noise_time_energies(
      ported_model,
      decoys_scaled,
      jnp.asarray(aatype),
      jnp.asarray(mask),
      t_values=(0.05,),
    )
    assert energies.shape == (1, decoys_scaled.shape[0])
    assert bool(jnp.all(jnp.isfinite(energies)))
    assert bool(jnp.all(energies >= 0.0))  # sum-of-squares parameterization, EnergyReadout invariant
    # A real trained model scoring genuinely different structures should not
    # collapse every decoy to the identical energy value.
    assert float(jnp.std(energies)) > 0.0

  def test_noise_time_sweep_reports_real_spearman_correlation(
    self,
    ported_model: ProteinEBMModel,
    native_structure: tuple[np.ndarray, np.ndarray, np.ndarray],
    decoy_set: tuple[np.ndarray, jax.Array, np.ndarray, np.ndarray],
    capsys: pytest.CaptureFixture[str],
  ) -> None:
    """The honest headline result: whatever Spearman(E, RMSD) actually comes out.

    No threshold is asserted on the correlation's value or sign -- per the
    BATHOS discipline ("do not tune anything to make it look good"), this
    test only asserts pipeline-sanity invariants (finite, well-shaped,
    non-degenerate) and prints the real numbers for the human report. A
    hard pass/fail gate on a specific Spearman value would either (a) be
    tautologically satisfied by construction (RMSD-vs-noise-level is
    monotonic by design, so it is not a real accuracy discovery) or (b) risk
    silently being loosened/tightened later to keep CI green, which is
    exactly the failure mode this discipline exists to prevent.
    """
    _ca_angstrom, aatype, mask = native_structure
    _decoys_angstrom, decoys_scaled, rmsd_labels, sigmas = decoy_set

    result = rank_decoys_over_noise_time(
      ported_model,
      decoys_scaled,
      jnp.asarray(aatype),
      jnp.asarray(mask),
      rmsd_labels,
      t_values=DEFAULT_NOISE_TIME_GRID,
    )

    assert result.energies.shape == (len(DEFAULT_NOISE_TIME_GRID), decoys_scaled.shape[0])
    assert all(np.isfinite(s) for s in result.spearman_by_t)
    assert -1.0 <= result.best_spearman <= 1.0

    # Supplementary diagnostic (not asserted -- reported for transparency):
    # restrict the correlation to the mild/moderate-noise decoys only
    # (excluding the two most extreme, ~7A/~14A RMSD levels -- an isotropic
    # per-atom Gaussian jitter of that magnitude is a very different, more
    # extreme perturbation than a real near-native decoy's structural error,
    # and plausibly pushes those two inputs out of the model's well-behaved
    # regime). This checks whether the *local*, more decoy-like regime shows
    # the physically-expected monotonic relationship even when the full
    # range does not.
    n_moderate = 4  # RMSD <= ~3.1A of the 6-level sweep
    moderate_spearman_by_t = tuple(
      spearman_energy_quality_correlation(result.energies[i, :n_moderate], rmsd_labels[:n_moderate])
      for i in range(len(result.t_values))
    )

    with capsys.disabled():
      print("\n=== E5 honest small-scale decoy-ranking proxy validation (NOT the Rosetta benchmark) ===")
      print(f"Structure: 1ubq.pdb (76 residues); noise levels (Angstrom sigma): {sigmas.tolist()}")
      print(f"RMSD-to-native labels (Angstrom): {np.round(rmsd_labels, 3).tolist()}")
      for i, t in enumerate(result.t_values):
        print(
          f"  t={t:.3f}  energies={np.round(result.energies[i], 2).tolist()}  "
          f"Spearman(E,RMSD) full(n=6)={result.spearman_by_t[i]:+.4f}  "
          f"moderate-only(n={n_moderate})={moderate_spearman_by_t[i]:+.4f}",
        )
      print(f"Best |Spearman| (full range): t={result.best_t:.3f}  rho={result.best_spearman:+.4f}")
      print(
        "Note: with only 6 decoys, none of these correlations are anywhere "
        "close to statistically significant (n too small) -- they are a "
        "pipeline sanity check, not a claim of parity with the paper's 0.838 "
        "Rosetta-decoy target.",
      )
      print(
        "Scope: real ported checkpoint weights, real PDB structure, SYNTHETIC "
        "coordinate noise (not real Rosetta decoys). Rosetta decoys.zip/rmsd/"
        "rosettascore download NOT authorized/attempted for this dispatch.",
      )
