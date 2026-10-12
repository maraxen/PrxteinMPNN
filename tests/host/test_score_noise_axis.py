"""``runner.score`` honours a backbone-noise level axis (spec 261011_noise-level-axis, N-D1..N-D4).

Replaces the #214 refusal test (debt #2509): the plain and fused multi-state score paths now
apply every ``spec.backbone_noise`` level, drawing the noise from the key ``runner.sample`` uses
for the same spec/seed.

Acceptance covered: (1) noise 0 is unchanged, (2) D >= 2 gives ``(S, C, D)`` whose ``[..., d]``
slice equals a single-level run, (3) score and sample noise the backbone identically, (4) nonzero
noise changes the score (a control the old silent-drop behaviour would fail).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aminx.host._sampling_grid_lineage import _base_sampling_key
from aminx.host.runner import _score_noise_key, score
from aminx.run.specs import SamplingSpecification, ScoringSpecification

_STRUCTURE = "tests/data/1ubq.pdb"
_PROTEIN_CHECKPOINT = "proteinmpnn_v_48_020"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_CANDIDATES = [_UBIQUITIN, "A" * len(_UBIQUITIN)]
_LEVELS = (0.0, 0.1, 0.3)
_ATOL = 2e-5  # float32 reduction-order noise between a vmapped and an un-vmapped level (cf. encode-once)


def _spec(**kwargs: object) -> ScoringSpecification:
  kwargs.setdefault("random_seed", 7)
  kwargs.setdefault("max_length", 128)
  return ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=_PROTEIN_CHECKPOINT,
    sequences_to_score=list(_CANDIDATES),
    return_logits=True,
    **kwargs,  # type: ignore[arg-type]
  )


def _sample_spec() -> SamplingSpecification:
  return SamplingSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=_PROTEIN_CHECKPOINT,
    max_length=128,
    random_seed=7,
    backbone_noise=_LEVELS,
  )


def test_noise_key_is_designs_key_without_weights() -> None:
  """N-D2 (weight-free): score's per-structure key equals design's ``encode_key`` for the same seed."""
  score_spec = _spec(backbone_noise=_LEVELS)
  sample_spec = _sample_spec()
  for structure_idx in (0, 1, 5):
    design_key = jax.random.fold_in(
      _base_sampling_key(sample_spec, grid_lineage=None),
      structure_idx,
    )
    np.testing.assert_array_equal(
      jax.random.key_data(_score_noise_key(score_spec, structure_idx)),
      jax.random.key_data(design_key),
    )
  # Negative control: a different seed or a different structure index gives a different key.
  other_seed = _spec(backbone_noise=_LEVELS, random_seed=8)
  assert not np.array_equal(
    jax.random.key_data(_score_noise_key(other_seed, 0)),
    jax.random.key_data(_score_noise_key(score_spec, 0)),
  )
  assert not np.array_equal(
    jax.random.key_data(_score_noise_key(score_spec, 1)),
    jax.random.key_data(_score_noise_key(score_spec, 0)),
  )


@pytest.mark.slow
@pytest.mark.requires_weights
def test_score_and_sample_noise_the_backbone_identically() -> None:
  """Acceptance 3: the encoder output at (structure 0, level d) is the same in score and sample.

  Seam: design encodes with ``plan.encode(bundle, fold_in(base_key, structure_idx), config)``;
  score encodes with ``make_score_split_fns(...).encode_structure(..., noise, noise_key)``.
  """
  from aminx.host.plan import make_inference_plan  # noqa: PLC0415
  from aminx.host.prep import prep_protein_stream_and_model  # noqa: PLC0415
  from aminx.inference.bundle_builder import build_inference_bundle  # noqa: PLC0415
  from aminx.scoring.score import make_score_split_fns  # noqa: PLC0415

  score_spec = _spec(backbone_noise=_LEVELS)
  sample_spec = _sample_spec()
  iterator, model = prep_protein_stream_and_model(score_spec)
  (ensemble,) = list(iterator)
  coords, mask = ensemble.coordinates[0], ensemble.mask[0]
  residue_index, chain_index = ensemble.residue_index[0], ensemble.chain_index[0]

  plan = make_inference_plan(model, sample_spec, purpose="sample")
  split_fns = make_score_split_fns(model)
  design_key = jax.random.fold_in(_base_sampling_key(sample_spec, grid_lineage=None), 0)
  score_key = _score_noise_key(score_spec, 0)

  encodings = []
  for level in _LEVELS:
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      backbone_noise=level,
      mode="score_conditional",
      inference=True,
    )
    designed = plan.encode(bundle, design_key, config)
    scored = split_fns.encode_structure(
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise=level,
      noise_key=score_key,
    )
    np.testing.assert_allclose(
      np.asarray(scored.node_features),
      np.asarray(designed.node_features),
      atol=_ATOL,
    )
    np.testing.assert_allclose(
      np.asarray(scored.edge_features),
      np.asarray(designed.edge_features),
      atol=_ATOL,
    )
    encodings.append(np.asarray(scored.node_features))
  # Negative control: noise really moved the encoding, so the equalities above are not vacuous.
  assert not np.allclose(encodings[0], encodings[2], atol=1e-4)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_multi_level_score_has_noise_axis_and_slices_match_single_level() -> None:
  """Acceptance 2 (+ N-D1): ``(S, C, D)`` scores, ``(S, C, D, L, 21)`` logits; slices match."""
  multi = score(_spec(backbone_noise=_LEVELS))
  n_candidates = len(_CANDIDATES)
  assert np.asarray(multi["scores"]).shape == (1, n_candidates, len(_LEVELS))
  logits = np.asarray(multi["logits"])
  assert logits.shape[:3] == (1, n_candidates, len(_LEVELS))
  assert logits.shape[-1] == 21
  assert tuple(multi["noise_levels"]) == _LEVELS
  assert tuple(multi["metadata"]["noise_levels"]) == _LEVELS
  assert multi["schema_version"] == "scoring_v2"

  for d, level in enumerate(_LEVELS):
    single = score(_spec(backbone_noise=(level,)))
    assert np.asarray(single["scores"]).shape == (1, n_candidates)  # one level keeps 2-D
    np.testing.assert_allclose(
      np.asarray(multi["scores"])[..., d],
      np.asarray(single["scores"]),
      atol=_ATOL,
    )
    np.testing.assert_allclose(logits[:, :, d], np.asarray(single["logits"]), atol=_ATOL)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_nonzero_noise_changes_score_and_zero_is_unchanged() -> None:
  """Acceptances 1 and 4: level 0 equals the default run; a nonzero level moves every candidate."""
  default = np.asarray(score(_spec())["scores"])
  explicit_zero = np.asarray(score(_spec(backbone_noise=0.0))["scores"])
  np.testing.assert_array_equal(explicit_zero, default)  # bit-identical, same code path
  noised = np.asarray(score(_spec(backbone_noise=0.3))["scores"])
  assert noised.shape == default.shape
  assert np.all(np.abs(noised - default) > 1e-4)  # the old silent-drop bug leaves these equal


@pytest.mark.slow
@pytest.mark.requires_weights
def test_noised_score_is_deterministic_in_the_seed() -> None:
  """Same spec/seed gives the same noised score; a different seed gives a different draw."""
  first = np.asarray(score(_spec(backbone_noise=0.3))["scores"])
  again = np.asarray(score(_spec(backbone_noise=0.3))["scores"])
  other = np.asarray(score(_spec(backbone_noise=0.3, random_seed=8))["scores"])
  np.testing.assert_array_equal(first, again)
  assert not np.allclose(first, other, atol=1e-5)


@pytest.mark.slow
@pytest.mark.requires_weights
def test_fused_multistate_score_gets_the_noise_axis() -> None:
  """Fused multi-state keeps its leading 1: ``(1, C)`` for one level, ``(1, C, D)`` for several."""
  # The fused path needs the map width to equal the padded length (as test_score_multistate_fusion does).
  length = len(_UBIQUITIN)
  positions = [list(range(length))]
  multi = score(_spec(backbone_noise=_LEVELS, state_position_map=positions, max_length=length))
  assert np.asarray(multi["scores"]).shape == (1, len(_CANDIDATES), len(_LEVELS))
  single = score(_spec(backbone_noise=(0.3,), state_position_map=positions, max_length=length))
  assert np.asarray(single["scores"]).shape == (1, len(_CANDIDATES))
  np.testing.assert_allclose(
    np.asarray(multi["scores"])[..., 2],
    np.asarray(single["scores"]),
    atol=_ATOL,
  )


@pytest.mark.slow
@pytest.mark.requires_weights
def test_averaged_score_with_two_levels_no_longer_crashes() -> None:
  """N-D4 / #2517: the R3 check is hoisted out of the vmap, so D >= 2 fuses to one ``(S, C)`` score."""
  result = score(_spec(average_node_features=True, backbone_noise=(0.1, 0.3)))
  assert np.asarray(result["scores"]).shape == (1, len(_CANDIDATES))
  assert tuple(result["noise_levels"]) == (0.1, 0.3)
