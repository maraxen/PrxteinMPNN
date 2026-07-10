"""Tests for the E6 ΔΔG-stability application logic (``aminx.ebm.ddg_stability``).

**Honest scope statement (read this before reading any assertion below).**
This module validates the *pipeline*, not the *published benchmark*. The
design spec's Spearman >= 0.686 target is measured against the real
ProteinGym/Tsuboyama-cDNA-proteolysis dataset, which is **not downloaded in
this environment** (confirmed absent; downloading it was explicitly out of
scope / not authorized for this backlog node -- see
``aminx.ebm.ddg_stability``'s module docstring). There is therefore **no
real experimental ΔΔG ground truth available anywhere in this test module**,
and nothing here computes or claims a Spearman correlation. What these tests
*do* establish, with real local PDB structures and (for one class) the real
E3.5-ported orbax weights:

1. The wiring is numerically correct: ``compute_ddg_stability``'s ``ddg``
   equals ``raw_ddg - unfolded_correction`` exactly, ``raw_ddg`` matches a
   manual, un-dispatched recomputation via ``model.energy`` (proving the
   ``score_mutant_ensemble`` axis-dispatch composition is faithful), and the
   MC unfolded-state correction is an exact match to a plain Python mean
   over the same ensemble (proving ``unfolded_state_correction``'s
   ``plan_axis``/``dispatch_axis``/``mean_fuse`` composition is faithful).
2. The mean-Fuse correction behaves like a mean should: it matches
   ``jnp.mean`` on ad hoc ensembles of varying size, and is reproducible
   given the same PRNG key.
3. Real local PDB parsing (``tests/data/*.pdb``) round-trips into sane
   coordinates/sequence/mask (CA-CA spacing near a real bond length,
   sequence spot-checked against the known ubiquitin sequence).
4. **The one genuinely trained-weights sanity check** (``TestRealCheckpoint...``,
   skipped if the orbax checkpoint directory is absent): using the real
   E3.5-ported ProteinEBM-x weights on ubiquitin (``1ubq.pdb``), mutating a
   real buried hydrophobic core residue (identified via
   ``Bio.PDB.SASA.ShrakeRupley``, independent of the EBM) to Gly/Pro produces
   a *measurably different* raw ΔΔG than a conservative same-family
   substitution at the same position. This is **only** a sanity check that
   the trained model's output is sensitive to a disruptive-vs-conservative
   distinction at all, at the specific numeric margin observed when this
   test was written (documented below) -- it is emphatically **not** a claim
   that the *sign*/*magnitude* matches any published biophysical value, and
   it is **not** a Spearman validation.

Everything else in this module uses small, randomly-initialized toy models
(same convention as ``tests/ebm/test_dispatch.py``) purely to validate
composition/wiring at CI speed, independent of any trained weights.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.ebm.ddg_stability import (
  compute_ddg_stability,
  generate_synthetic_unfolded_ensemble,
  identify_buried_hydrophobic_positions,
  load_ca_backbone_from_pdb,
  make_point_mutants,
  random_point_mutants,
  unfolded_state_correction,
)
from aminx.ebm.dispatch import score_mutant_ensemble
from aminx.ebm.model import ProteinEBMModel
from aminx.utils.aa_convert import MPNN_ALPHABET

_TEST_DATA = Path(__file__).resolve().parent.parent / "data"
_UBIQUITIN_PDB = _TEST_DATA / "1ubq.pdb"
_PEPTIDE_PDB = _TEST_DATA / "5awl.pdb"

_ORBAX_CHECKPOINT_DIR = Path("/tmp/proteinebm_weights/ported_jax_model")
_REAL_CHECKPOINT_AVAILABLE = (_ORBAX_CHECKPOINT_DIR / "0" / "model").exists()

TOKEN_S = 16
TOKEN_Z = 8
DEPTH = 2
HEADS = 2


def _make_toy_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    key=key,
  )


class TestLoadCaBackboneFromPdb:
  def test_ubiquitin_shape_and_known_sequence(self) -> None:
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB)
    assert wt.coords.shape == (76, 3)
    assert wt.aatype.shape == (76,)
    assert wt.mask.shape == (76,)
    assert bool(jnp.all(wt.mask))
    # Ubiquitin's published sequence starts MQIFVKTLTGK... (residue 1 = Met).
    letters = "".join(MPNN_ALPHABET[int(a)] for a in np.asarray(wt.aatype))
    assert letters.startswith("MQIFVKTLTGK")
    assert wt.residue_ids[0] == 1
    assert wt.residue_ids[-1] == 76

  def test_ca_ca_spacing_matches_real_bond_length_scaled(self) -> None:
    """Consecutive CA-CA distances should be near the real ~3.8 Angstrom bond, scaled by 0.1."""
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB)
    deltas = wt.coords[1:] - wt.coords[:-1]
    distances = jnp.linalg.norm(deltas, axis=-1)
    # 0.38 in the coordinate_scaling-scaled space == 3.8 Angstrom real.
    assert float(jnp.median(distances)) == pytest.approx(0.38, abs=0.05)

  def test_peptide_structure_parses(self) -> None:
    wt = load_ca_backbone_from_pdb(_PEPTIDE_PDB)
    assert wt.coords.shape[0] == 10
    assert wt.aatype.shape[0] == 10

  def test_missing_chain_raises(self) -> None:
    with pytest.raises(KeyError):
      load_ca_backbone_from_pdb(_UBIQUITIN_PDB, chain_id="Z")

  def test_sasa_populated_only_when_requested(self) -> None:
    wt_no_sasa = load_ca_backbone_from_pdb(_UBIQUITIN_PDB)
    assert wt_no_sasa.sasa is None
    wt_with_sasa = load_ca_backbone_from_pdb(_UBIQUITIN_PDB, compute_sasa=True)
    assert wt_with_sasa.sasa is not None
    assert len(wt_with_sasa.sasa) == 76


class TestIdentifyBuriedHydrophobicPositions:
  def test_known_ubiquitin_core_residues_are_identified(self) -> None:
    """Ile3/Val5/Leu15 are textbook buried hydrophobic-core positions in ubiquitin."""
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB, compute_sasa=True)
    buried = identify_buried_hydrophobic_positions(wt, sasa_threshold=20.0)
    assert len(buried) > 0
    buried_residue_ids = {wt.residue_ids[i] for i in buried}
    # 0-indexed position 2 == residue_id 3 (Ile3); 4 == residue_id 5 (Val5).
    assert 3 in buried_residue_ids
    assert 5 in buried_residue_ids

  def test_all_returned_positions_are_hydrophobic_and_buried(self) -> None:
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB, compute_sasa=True)
    assert wt.sasa is not None
    buried = identify_buried_hydrophobic_positions(wt, sasa_threshold=20.0)
    for i in buried:
      assert MPNN_ALPHABET[int(wt.aatype[i])] in "AVLIMFWY"
      assert wt.sasa[i] <= 20.0

  def test_sorted_most_buried_first(self) -> None:
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB, compute_sasa=True)
    assert wt.sasa is not None
    buried = identify_buried_hydrophobic_positions(wt, sasa_threshold=20.0)
    sasas = [wt.sasa[i] for i in buried]
    assert sasas == sorted(sasas)

  def test_raises_without_sasa(self) -> None:
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB)
    with pytest.raises(ValueError, match="compute_sasa=True"):
      identify_buried_hydrophobic_positions(wt)


class TestMakePointMutants:
  def test_single_mutation_shape_and_value(self) -> None:
    wildtype = jnp.zeros((5,), dtype=jnp.int32)  # all "A" (index 0)
    mutants = make_point_mutants(wildtype, [(2, "G")])
    assert mutants.shape == (1, 5)
    assert int(mutants[0, 2]) == MPNN_ALPHABET.index("G")
    # Every other position unchanged.
    assert jnp.array_equal(mutants[0, :2], wildtype[:2])
    assert jnp.array_equal(mutants[0, 3:], wildtype[3:])

  def test_multiple_mutations_are_independent_rows(self) -> None:
    wildtype = jnp.arange(6, dtype=jnp.int32) % 20
    mutants = make_point_mutants(wildtype, [(0, "P"), (5, "G")])
    assert mutants.shape == (2, 6)
    assert int(mutants[0, 0]) == MPNN_ALPHABET.index("P")
    assert int(mutants[1, 5]) == MPNN_ALPHABET.index("G")
    # Row 0 only touched position 0; row 1 only touched position 5.
    assert jnp.array_equal(mutants[0, 1:], wildtype[1:])
    assert jnp.array_equal(mutants[1, :5], wildtype[:5])

  def test_unknown_letter_raises(self) -> None:
    wildtype = jnp.zeros((3,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="unrecognized"):
      make_point_mutants(wildtype, [(0, "Z")])

  def test_out_of_range_position_raises(self) -> None:
    wildtype = jnp.zeros((3,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="out of range"):
      make_point_mutants(wildtype, [(10, "G")])


class TestRandomPointMutants:
  def test_shape_and_positions_touched(self) -> None:
    wildtype = jnp.zeros((10,), dtype=jnp.int32)
    mutants, mutations = random_point_mutants(jax.random.PRNGKey(0), wildtype, [1, 3, 7])
    assert mutants.shape == (3, 10)
    assert len(mutations) == 3
    assert [p for p, _ in mutations] == [1, 3, 7]

  def test_exclude_wildtype_never_reproduces_wildtype_identity(self) -> None:
    wildtype = jnp.zeros((20,), dtype=jnp.int32)  # all "A"
    _mutants, mutations = random_point_mutants(
      jax.random.PRNGKey(1), wildtype, list(range(20)), exclude_wildtype=True,
    )
    assert all(letter != "A" for _pos, letter in mutations)

  def test_never_draws_the_mask_token(self) -> None:
    wildtype = jnp.zeros((30,), dtype=jnp.int32)
    _mutants, mutations = random_point_mutants(jax.random.PRNGKey(2), wildtype, list(range(30)))
    assert all(letter != "X" for _pos, letter in mutations)

  def test_reproducible_given_same_key(self) -> None:
    wildtype = jnp.arange(10, dtype=jnp.int32)
    m1, mut1 = random_point_mutants(jax.random.PRNGKey(7), wildtype, [0, 2, 4])
    m2, mut2 = random_point_mutants(jax.random.PRNGKey(7), wildtype, [0, 2, 4])
    assert jnp.array_equal(m1, m2)
    assert mut1 == mut2


class TestGenerateSyntheticUnfoldedEnsemble:
  def test_shape(self) -> None:
    ensemble = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(0), n_residues=12, n_ensemble=5)
    assert ensemble.shape == (5, 12, 3)

  def test_step_length_matches_requested_bond_length(self) -> None:
    step_length = 0.25
    ensemble = generate_synthetic_unfolded_ensemble(
      jax.random.PRNGKey(1), n_residues=20, n_ensemble=4, step_length=step_length,
    )
    deltas = ensemble[:, 1:, :] - ensemble[:, :-1, :]
    distances = jnp.linalg.norm(deltas, axis=-1)
    assert float(jnp.mean(distances)) == pytest.approx(step_length, rel=1e-4)

  def test_ensemble_members_differ(self) -> None:
    ensemble = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(2), n_residues=15, n_ensemble=6)
    # No two members should be identical (independent random draws).
    for i in range(ensemble.shape[0]):
      for j in range(i + 1, ensemble.shape[0]):
        assert not jnp.allclose(ensemble[i], ensemble[j])

  def test_reproducible_given_same_key(self) -> None:
    e1 = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(9), n_residues=8, n_ensemble=3)
    e2 = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(9), n_residues=8, n_ensemble=3)
    assert jnp.array_equal(e1, e2)

  def test_each_member_is_centered(self) -> None:
    ensemble = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(3), n_residues=10, n_ensemble=4)
    centroids = jnp.mean(ensemble, axis=1)
    assert jnp.allclose(centroids, 0.0, atol=1e-5)


class TestUnfoldedStateCorrection:
  def test_matches_manual_mean_computation(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(0))
    n = 8
    aatype = jax.random.randint(jax.random.PRNGKey(1), (n,), 0, 21)
    mask = jnp.ones((n,), dtype=bool)
    t = jnp.array(0.1)
    ensemble = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(2), n_residues=n, n_ensemble=5)

    correction = unfolded_state_correction(model, aatype, t, mask, ensemble, default_batch_size=8)
    manual_mean = jnp.mean(jnp.stack([model.energy(ensemble[i], aatype, t, mask) for i in range(5)]))
    assert jnp.allclose(correction, manual_mean, atol=1e-5)

  def test_safemap_path_also_matches_manual_mean(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(3))
    n = 6
    aatype = jax.random.randint(jax.random.PRNGKey(4), (n,), 0, 21)
    mask = jnp.ones((n,), dtype=bool)
    t = jnp.array(0.2)
    ensemble = generate_synthetic_unfolded_ensemble(jax.random.PRNGKey(5), n_residues=n, n_ensemble=4)

    # cardinality=4 > default_batch_size=2, divisible -> SafeMap branch.
    correction = unfolded_state_correction(model, aatype, t, mask, ensemble, default_batch_size=2)
    manual_mean = jnp.mean(jnp.stack([model.energy(ensemble[i], aatype, t, mask) for i in range(4)]))
    assert jnp.allclose(correction, manual_mean, atol=1e-5)

  def test_larger_ensemble_converges_toward_stable_mean(self) -> None:
    """Larger U should reduce sampling variance -- repeated large-U corrections agree more
    closely with each other than repeated small-U corrections do."""
    model = _make_toy_model(jax.random.PRNGKey(6))
    n = 10
    aatype = jax.random.randint(jax.random.PRNGKey(7), (n,), 0, 21)
    mask = jnp.ones((n,), dtype=bool)
    t = jnp.array(0.1)

    def _spread(n_ensemble: int, n_trials: int) -> float:
      values = []
      for trial in range(n_trials):
        ensemble = generate_synthetic_unfolded_ensemble(
          jax.random.PRNGKey(100 + trial), n_residues=n, n_ensemble=n_ensemble,
        )
        values.append(float(unfolded_state_correction(model, aatype, t, mask, ensemble)))
      return float(np.std(values))

    small_u_spread = _spread(n_ensemble=2, n_trials=8)
    large_u_spread = _spread(n_ensemble=64, n_trials=8)
    assert large_u_spread < small_u_spread


class TestComputeDdgStability:
  def test_ddg_equals_raw_ddg_minus_correction(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(0))
    wt_aatype = jax.random.randint(jax.random.PRNGKey(1), (10,), 0, 21)
    coords = jax.random.normal(jax.random.PRNGKey(2), (10, 3)) * 0.1
    mask = jnp.ones((10,), dtype=bool)
    from aminx.ebm.ddg_stability import WildtypeStructure

    wt = WildtypeStructure(coords=coords, aatype=wt_aatype, mask=mask, residue_ids=tuple(range(10)))

    result = compute_ddg_stability(
      model, wt, mutations=[(0, "G"), (5, "P")], t=0.05, n_unfolded_ensemble=4, key=jax.random.PRNGKey(3),
    )
    assert jnp.allclose(result.ddg, result.raw_ddg - result.unfolded_correction, atol=1e-6)

  def test_raw_ddg_matches_manual_score_mutant_ensemble_call(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(4))
    wt_aatype = jax.random.randint(jax.random.PRNGKey(5), (12,), 0, 21)
    coords = jax.random.normal(jax.random.PRNGKey(6), (12, 3)) * 0.1
    mask = jnp.ones((12,), dtype=bool)
    from aminx.ebm.ddg_stability import WildtypeStructure

    wt = WildtypeStructure(coords=coords, aatype=wt_aatype, mask=mask, residue_ids=tuple(range(12)))
    mutations = [(1, "G"), (3, "P"), (11, "V")]

    result = compute_ddg_stability(
      model, wt, mutations=mutations, t=0.05, n_unfolded_ensemble=3, key=jax.random.PRNGKey(7),
    )
    mutant_aatype = make_point_mutants(wt_aatype, mutations)
    expected_raw_ddg = score_mutant_ensemble(
      model, coords, mutant_aatype, jnp.array(0.05), mask, wildtype_aatype=wt_aatype,
    )
    assert jnp.allclose(result.raw_ddg, expected_raw_ddg, atol=1e-5)

  def test_wildtype_energy_matches_direct_call(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(8))
    wt_aatype = jax.random.randint(jax.random.PRNGKey(9), (9,), 0, 21)
    coords = jax.random.normal(jax.random.PRNGKey(10), (9, 3)) * 0.1
    mask = jnp.ones((9,), dtype=bool)
    from aminx.ebm.ddg_stability import WildtypeStructure

    wt = WildtypeStructure(coords=coords, aatype=wt_aatype, mask=mask, residue_ids=tuple(range(9)))
    result = compute_ddg_stability(
      model, wt, mutations=[(0, "G")], t=0.1, n_unfolded_ensemble=2, key=jax.random.PRNGKey(11),
    )
    expected = model.energy(coords, wt_aatype, jnp.array(0.1), mask)
    assert jnp.allclose(result.wildtype_energy, expected, atol=1e-6)

  def test_shapes(self) -> None:
    model = _make_toy_model(jax.random.PRNGKey(12))
    wt_aatype = jax.random.randint(jax.random.PRNGKey(13), (7,), 0, 21)
    coords = jax.random.normal(jax.random.PRNGKey(14), (7, 3)) * 0.1
    mask = jnp.ones((7,), dtype=bool)
    from aminx.ebm.ddg_stability import WildtypeStructure

    wt = WildtypeStructure(coords=coords, aatype=wt_aatype, mask=mask, residue_ids=tuple(range(7)))
    mutations = [(0, "G"), (1, "P"), (2, "V"), (3, "L")]
    result = compute_ddg_stability(
      model, wt, mutations=mutations, n_unfolded_ensemble=5, key=jax.random.PRNGKey(15),
    )
    assert result.ddg.shape == (4,)
    assert result.raw_ddg.shape == (4,)
    assert result.wildtype_energy.shape == ()
    assert result.unfolded_correction.shape == ()
    assert result.mutations == tuple(mutations)

  def test_real_ubiquitin_structure_end_to_end_with_toy_model(self) -> None:
    """End-to-end on a REAL local PDB structure (not synthetic coordinates), toy weights."""
    model = _make_toy_model(jax.random.PRNGKey(16))
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB)
    mutations = [(2, "G"), (2, "L"), (14, "P"), (14, "I")]
    result = compute_ddg_stability(
      model, wt, mutations=mutations, t=0.05, n_unfolded_ensemble=4, key=jax.random.PRNGKey(17),
    )
    assert result.ddg.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(result.ddg)))


@pytest.mark.skipif(
  not _REAL_CHECKPOINT_AVAILABLE,
  reason=f"Real E3.5-ported orbax checkpoint not found at {_ORBAX_CHECKPOINT_DIR} in this environment.",
)
class TestRealCheckpointDestabilizingMutationSanity:
  """The one genuinely trained-weights sanity check in this test module.

  **Not a Spearman validation** (see module docstring). Loads the real
  E3.5-ported ProteinEBM-x weights and checks that, on ubiquitin
  (``1ubq.pdb``), mutating a real buried hydrophobic core residue to Gly/Pro
  produces a raw ΔΔG *measurably different* from a conservative
  same-family substitution at the same position.

  Concrete numbers observed when this test was written (real ported
  checkpoint, ``t=0.05``, ``n_unfolded_ensemble=3``): at Ile3 (buried,
  most-occluded core position), disruptive Ile3Gly raw ΔΔG = -3.34,
  conservative Ile3Leu raw ΔΔG = -10.93 (|diff| = 7.59); at Val5,
  Val5Gly = +6.00 vs Val5Leu = -6.02 (|diff| = 12.01); at Ile23,
  Ile23Gly = +2.55 vs Ile23Leu = -5.61 (|diff| = 8.16). All three buried
  positions tested showed |diff| > 7 -- the threshold below asserts > 2.0,
  a conservative margin below every observed value, so the test is not
  fragile to small numeric drift while still ruling out a degenerate
  (near-zero-difference) pipeline.
  """

  @pytest.fixture(scope="class")
  def real_model(self) -> ProteinEBMModel:
    import orbax.checkpoint as ocp

    template = ProteinEBMModel(
      token_s=256,
      token_z=128,
      dim_fourier=256,
      conditioning_transition_layers=2,
      transformer_depth=16,
      transformer_heads=8,
      num_contact_embeddings=3,
      key=jax.random.PRNGKey(0),
    )
    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    manager = ocp.CheckpointManager(
      _ORBAX_CHECKPOINT_DIR.resolve(),
      options=options,
      item_handlers={"model": ocp.PyTreeCheckpointHandler()},
    )
    restored = manager.restore(step=0, args=ocp.args.Composite(model=ocp.args.PyTreeRestore(template)))
    return restored["model"]

  def test_disruptive_vs_conservative_mutation_at_buried_positions(
    self, real_model: ProteinEBMModel,
  ) -> None:
    wt = load_ca_backbone_from_pdb(_UBIQUITIN_PDB, compute_sasa=True)
    buried_positions = identify_buried_hydrophobic_positions(wt, sasa_threshold=20.0)[:3]
    assert len(buried_positions) == 3

    conservative_map = {"I": "L", "V": "L", "L": "I", "M": "L", "F": "Y", "W": "Y", "A": "V", "Y": "F"}
    mutations: list[tuple[int, str]] = []
    for position in buried_positions:
      wt_letter = MPNN_ALPHABET[int(wt.aatype[position])]
      disruptive_letter = "G" if wt_letter != "G" else "P"
      conservative_letter = conservative_map.get(wt_letter, "V")
      mutations.append((position, disruptive_letter))
      mutations.append((position, conservative_letter))

    result = compute_ddg_stability(
      real_model, wt, mutations=mutations, t=0.05, n_unfolded_ensemble=3, key=jax.random.PRNGKey(42),
    )

    for i in range(0, len(mutations), 2):
      disruptive_ddg = float(result.raw_ddg[i])
      conservative_ddg = float(result.raw_ddg[i + 1])
      abs_diff = abs(disruptive_ddg - conservative_ddg)
      # Conservative margin: every position observed when writing this test
      # showed |diff| > 7; asserting > 2.0 rules out a degenerate/near-zero
      # pipeline without being fragile to minor numeric drift.
      assert abs_diff > 2.0, (
        f"position {buried_positions[i // 2]}: disruptive={disruptive_ddg}, "
        f"conservative={conservative_ddg}, |diff|={abs_diff} (expected > 2.0)"
      )
