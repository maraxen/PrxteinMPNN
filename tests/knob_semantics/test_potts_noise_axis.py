# ruff: noqa: S101
"""PottsMPNN noise-level axis (spec 261011 N-D5, acceptance 1 and 5).

``backbone_noise`` takes every level. Levels share ONE draw (key ``0x9077``, the level is not folded in)
scaled by the level, so:

* one level, however large, is bit-identical to the old first-level-only behaviour and keeps its shape;
* two or more levels add a ``D`` axis right after the candidate/sample axis, and slice ``d`` equals a
  single-level run at level ``d`` (level 0.0 is the noise-free run).

The model is a random-init ``PottsMPNN`` serialised to ``tmp_path``; no checkpoint is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import PottsMPNNDriver
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import PottsMPNNOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

_THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY"}
_LEVEL = 0.5


def _write_pdb(path: Path, sequence: str) -> None:
  lines: list[str] = []
  serial = 1
  for index, amino in enumerate(sequence, start=1):
    for offset, atom in enumerate(("N", "CA", "C", "O")):
      x, y, z = float(index * 3 + offset), float(ord("A")), float(offset)
      lines.append(
        f"{'ATOM':<6.6}{serial:5d} {atom:>4.4}{' ':1.1}{_THREE[amino]:>3.3} {'A':1.1}"
        f"{index:4d}{' ':1.1}   {x:8.3f}{y:8.3f}{z:8.3f}",
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def registered() -> Iterator[PottsMPNNDriver]:
  driver = PottsMPNNDriver()
  FAMILY_DRIVERS.register("pottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("pottsmpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, PottsMPNN(key=jax.random.PRNGKey(0)))
  return path


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  path = tmp_path / "toy.pdb"
  _write_pdb(path, "ACDEFG")
  return path


def _score_spec(pdb: Path, model_path: Path, **kwargs: Any) -> ScoringSpecification:  # noqa: ANN401
  return ScoringSpecification(
    inputs=str(pdb),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=model_path,
    output_kind="energy",
    sequences_to_score=["ADDEFG"],
    potts_mpnn=PottsMPNNOptions(),
    **kwargs,
  )


def _energy(pdb: Path, model_path: Path, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
  spec = _score_spec(pdb, model_path, **kwargs)
  return np.asarray(score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64)


def _sample(
  pdb: Path,
  model_path: Path,
  options: PottsMPNNOptions,
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=model_path,
    num_samples=3,
    samples_chunk_size=2,
    return_logits=False,
    potts_mpnn=options,
    **kwargs,
  )
  return sample(spec)["structures"]["0"]["arrays"]


def test_score_single_level_keeps_shape_and_values(
  registered: PottsMPNNDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  """A one-level tuple is the old scalar: same shape, same numbers, and it still changes the energy."""
  del registered
  clean = _energy(pdb, model_path)
  scalar = _energy(pdb, model_path, backbone_noise=_LEVEL)
  one = _energy(pdb, model_path, backbone_noise=(_LEVEL,))
  assert clean.shape == scalar.shape == one.shape == (2,)
  np.testing.assert_array_equal(scalar, one)
  assert not np.allclose(scalar, clean), "nonzero single-level noise must still reach the energy"


def test_score_two_levels_add_axis_and_slices_match_single_runs(
  registered: PottsMPNNDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  del registered
  clean = _energy(pdb, model_path)
  noised = _energy(pdb, model_path, backbone_noise=_LEVEL)
  both = _energy(pdb, model_path, backbone_noise=(0.0, _LEVEL))
  assert both.shape == (2, 2), both.shape  # (C, D)
  np.testing.assert_array_equal(both[:, 0], clean)
  np.testing.assert_allclose(both[:, 1], noised, rtol=1e-6, atol=1e-6)
  assert not np.allclose(both[:, 0], both[:, 1])


def test_score_schema_declares_noise_axis_only_for_two_levels(
  registered: PottsMPNNDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  one_spec = _score_spec(pdb, model_path, backbone_noise=_LEVEL)
  two_spec = _score_spec(pdb, model_path, backbone_noise=(0.0, _LEVEL))
  assert registered.result_schema(one_spec, "score:energy")["energy"].dims == ("N_cand",)
  assert registered.result_schema(two_spec, "score:energy")["energy"].dims == ("N_cand", "D_noise")


def test_sample_two_levels_add_axis_and_level_zero_is_noise_free(
  registered: PottsMPNNDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  del registered
  options = PottsMPNNOptions(optimization_mode="none")
  clean = _sample(pdb, model_path, options)
  noised = _sample(pdb, model_path, options, backbone_noise=_LEVEL)
  both = _sample(pdb, model_path, options, backbone_noise=(0.0, _LEVEL))

  n = clean["sequence"].shape[0]
  assert clean["sample_energy"].shape == (n,)
  assert both["sequence"].shape == (n, 2, 6), both["sequence"].shape
  assert both["sample_energy"].shape == (n, 2), both["sample_energy"].shape
  # The runner ranks along axis N, so D >= 2 yields one ranking per level.
  assert both["sample_rank"].shape == (n, 2), both["sample_rank"].shape
  np.testing.assert_array_equal(both["sample_rank"][:, 0], clean["sample_rank"])
  np.testing.assert_array_equal(both["sequence"][:, 0], clean["sequence"])
  np.testing.assert_allclose(both["sample_energy"][:, 0], clean["sample_energy"], rtol=1e-6)
  np.testing.assert_allclose(both["sample_energy"][:, 1], noised["sample_energy"], rtol=1e-5)
  assert not np.allclose(both["sample_energy"][:, 0], both["sample_energy"][:, 1])


def test_sample_refuses_emit_etab_with_a_noise_sweep(
  registered: PottsMPNNDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  """``potts_etab`` has no sample axis to put a noise axis after, so D >= 2 is refused, not truncated."""
  del registered
  options = PottsMPNNOptions(optimization_mode="none", emit_etab=True)
  with pytest.raises(ValueError, match="emit_etab"):
    _sample(pdb, model_path, options, backbone_noise=(0.0, _LEVEL))
