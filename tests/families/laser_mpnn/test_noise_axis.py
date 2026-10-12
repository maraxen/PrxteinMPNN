# ruff: noqa: S101
"""LASEr noise-level axis (spec 261011 N-D5, acceptance 1 and 5).

``backbone_noise`` takes every level. Levels share ONE draw (key ``0x6262``, the level is not folded in)
scaled by the level, so a single level keeps today's shape and values, and two or more levels add a
``D`` axis right after the candidate/sample axis whose slice ``d`` equals a single-level run at level ``d``.
Proofread results have no noise axis, so a multi-level sweep is refused there.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import LASErMPNN, LaserDriver  # noqa: E402
from aminx.host.family_driver import FAMILY_DRIVERS  # noqa: E402
from aminx.host.runner import sample, score  # noqa: E402
from aminx.run.options import LaserOptions  # noqa: E402
from aminx.run.specs import SamplingSpecification, ScoringSpecification  # noqa: E402
from tests.families.laser_mpnn.test_driver import _write_pdb  # noqa: E402

_LEVEL = 0.5


@pytest.fixture
def registered() -> Iterator[LaserDriver]:
  driver = LaserDriver()
  FAMILY_DRIVERS.register("lasermpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("lasermpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, LASErMPNN(key=jax.random.PRNGKey(0)))
  return path


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  path = tmp_path / "complex.pdb"
  _write_pdb(path)
  return path


def _score_spec(pdb: Path, model_path: Path, kind: str, **kwargs: Any) -> ScoringSpecification:  # noqa: ANN401
  return ScoringSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_test",
    model_local_path=model_path,
    output_kind=kind,  # type: ignore[arg-type]
    sequences_to_score=["AA", "AG"],
    random_seed=3,
    **kwargs,
  )


def _nll(pdb: Path, model_path: Path, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
  result = score(_score_spec(pdb, model_path, "nll", **kwargs))
  return np.asarray(result["structures"]["0"]["arrays"]["nll"], dtype=np.float64)


def _sample(pdb: Path, model_path: Path, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_test",
    model_local_path=model_path,
    num_samples=2,
    temperature=0.8,
    random_seed=5,
    laser=LaserOptions(chi_temp=1.0, disabled_residues=("X",)),
    **kwargs,
  )
  return sample(spec)["structures"]["0"]["arrays"]


def test_score_single_level_keeps_shape_and_values(
  registered: LaserDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  del registered
  clean = _nll(pdb, model_path)
  scalar = _nll(pdb, model_path, backbone_noise=_LEVEL)
  one = _nll(pdb, model_path, backbone_noise=(_LEVEL,))
  assert clean.shape == scalar.shape == one.shape  # (C, L)
  np.testing.assert_array_equal(scalar, one)
  assert not np.allclose(scalar, clean), "nonzero single-level noise must still reach the nll"


def test_score_two_levels_add_axis_and_slices_match_single_runs(
  registered: LaserDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  del registered
  clean = _nll(pdb, model_path)
  noised = _nll(pdb, model_path, backbone_noise=_LEVEL)
  both = _nll(pdb, model_path, backbone_noise=(0.0, _LEVEL))
  n_cand, length = clean.shape
  assert both.shape == (n_cand, 2, length), both.shape  # (C, D, L): D right after the candidate axis
  np.testing.assert_array_equal(both[:, 0], clean)
  np.testing.assert_array_equal(both[:, 1], noised)
  assert not np.allclose(both[:, 0], both[:, 1])


def test_score_schema_declares_noise_axis_only_for_two_levels(
  registered: LaserDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  one = _score_spec(pdb, model_path, "nll", backbone_noise=_LEVEL)
  two = _score_spec(pdb, model_path, "nll", backbone_noise=(0.0, _LEVEL))
  assert registered.result_schema(one, "score:nll")["nll"].dims == ("N_cand", "L_total")
  assert registered.result_schema(two, "score:nll")["nll"].dims == ("N_cand", "D_noise", "L_total")
  logits = registered.result_schema(two, "score:logits")["logits"].dims
  assert logits == ("N_cand", "D_noise", "L_total", "alphabet")


def test_proofread_refuses_a_noise_sweep(
  registered: LaserDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  """Proofread reduces over an ensemble; there is no honest noise axis, so D >= 2 is refused."""
  spec = _score_spec(pdb, model_path, "proofread_unconditional", backbone_noise=(0.0, _LEVEL))
  with pytest.raises(ValueError, match="proofread"):
    registered.result_schema(spec, "score:proofread_unconditional")


def test_sample_two_levels_add_axis_and_level_zero_is_noise_free(
  registered: LaserDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  del registered
  clean = _sample(pdb, model_path)
  noised = _sample(pdb, model_path, backbone_noise=_LEVEL)
  both = _sample(pdb, model_path, backbone_noise=(0.0, _LEVEL))

  n = clean["sequence"].shape[0]
  length = clean["sequence"].shape[1]
  assert both["sequence"].shape == (n, 2, length), both["sequence"].shape
  assert both["seq_log_prob"].shape == (n, 2, length)
  assert both["chi_deg"].shape == (n, 2, length, 4)
  np.testing.assert_array_equal(both["sequence"][:, 0], clean["sequence"])
  for name in ("seq_log_prob", "chi_deg"):
    np.testing.assert_allclose(both[name][:, 0], clean[name], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(both[name][:, 1], noised[name], rtol=0, atol=0, equal_nan=True)
  assert not np.allclose(both["seq_log_prob"][:, 0], both["seq_log_prob"][:, 1])


def test_sample_schema_declares_noise_axis_only_for_two_levels(registered: LaserDriver) -> None:
  def _spec(level: object) -> SamplingSpecification:
    return SamplingSpecification(
      inputs="a.pdb",
      model_family="lasermpnn",
      backbone_noise=level,  # type: ignore[arg-type]
    )

  one = registered.result_schema(_spec(_LEVEL), "sample")
  two = registered.result_schema(_spec((0.0, _LEVEL)), "sample")
  assert one["sequence"].dims == ("N", "L_total")
  assert two["sequence"].dims == ("N", "D_noise", "L_total")
  assert two["chi_deg"].dims == ("N", "D_noise", "L_total", "4")
  assert two["sidechain_coords"].dims == ("N", "D_noise", "L_total", "atom", "xyz")
