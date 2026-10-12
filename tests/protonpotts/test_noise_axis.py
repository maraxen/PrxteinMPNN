"""ProtonPottsMPNN backbone-noise axis (spec 261011 N-D5, acceptance 1 and 5).

Before this, the ProtonPotts driver ignored ``backbone_noise`` for every purpose: a nonzero level changed
nothing and nothing said so. Upstream PottsMPNN-pH featurizes through the same ``structure_noise`` path as
PottsMPNN (its pH engine passes 0.0), so the driver now reuses ``apply_potts_backbone_noise`` on the featurized
coordinates with its own key (fold ``0x7070``). Levels share one draw scaled by the level.

* the control the old silent drop would fail: a nonzero level changes ``score:energy``;
* one level keeps today's shape; two or more add a ``D`` axis right after the candidate axis, slice 0 (level
  0.0) is the noise-free run and slice ``d`` equals a single-level run at level ``d``;
* pH design (``sample``) cannot carry the axis and refuses D >= 2.

The model is random, so these check the driver contract, not agreement with upstream.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ProtonPottsDriver
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import ProtonPottsOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR"]
_LEVEL = 0.5


def _atom(serial: int, name: str, resname: str, number: int, xyz: tuple[float, float, float]) -> str:
  padded = f" {name:<3}"
  return (
    f"ATOM  {serial:>5} {padded} {resname:>3} A{number:>4}    "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2}"
  )


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  rng = np.random.default_rng(0)
  lines: list[str] = []
  for i, resname in enumerate(RESIDUES, start=1):
    centre = np.array([3.8 * i, 2.0 * np.sin(i), 2.0 * np.cos(i)])
    for k, atom in enumerate(("N", "CA", "C", "O")):
      xyz = centre + rng.normal(scale=0.4, size=3) + np.array([0.6 * k, 0.0, 0.0])
      lines.append(_atom(len(lines) + 1, atom, resname, i, (xyz[0], xyz[1], xyz[2])))
  path = tmp_path / "toy.pdb"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


@pytest.fixture
def registered() -> Iterator[ProtonPottsDriver]:
  driver = ProtonPottsDriver()
  FAMILY_DRIVERS.register("protonpottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("protonpottsmpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6))
  return path


@pytest.fixture
def variants(tmp_path: Path) -> str:
  path = tmp_path / "variants.json"
  path.write_text(json.dumps({"his_protonated": {"A:2": "HIS-P"}}), encoding="utf-8")
  return str(path)


def _score(pdb: Path, model_path: Path, kind: str, **kwargs: Any) -> dict[str, np.ndarray]:  # noqa: ANN401
  options = kwargs.pop("options")
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind=kind,  # type: ignore[arg-type]
    protonpotts=options,
    **kwargs,
  )
  return score(spec)["structures"]["0"]["arrays"]


def _energy(pdb: Path, model_path: Path, variants: str, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
  options = ProtonPottsOptions(variants_json=variants)
  arrays = _score(pdb, model_path, "energy", options=options, **kwargs)
  return np.asarray(arrays["energy"], dtype=np.float64)


def test_nonzero_noise_changes_energy_which_the_old_silent_ignore_would_fail(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  del registered
  clean = _energy(pdb, model_path, variants)
  noised = _energy(pdb, model_path, variants, backbone_noise=_LEVEL)
  assert clean.shape == noised.shape == (2,)  # reference + one variant, no noise axis for one level
  assert not np.allclose(clean, noised), "backbone_noise reached nothing: ProtonPotts ignored it"
  np.testing.assert_array_equal(noised, _energy(pdb, model_path, variants, backbone_noise=(_LEVEL,)))
  np.testing.assert_array_equal(clean, _energy(pdb, model_path, variants, backbone_noise=0.0))


def test_noise_is_seeded(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  del registered
  first = _energy(pdb, model_path, variants, backbone_noise=_LEVEL, random_seed=3)
  again = _energy(pdb, model_path, variants, backbone_noise=_LEVEL, random_seed=3)
  other = _energy(pdb, model_path, variants, backbone_noise=_LEVEL, random_seed=4)
  np.testing.assert_array_equal(first, again)
  assert not np.allclose(first, other)


def test_energy_two_levels_add_axis_and_slices_match_single_runs(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  del registered
  clean = _energy(pdb, model_path, variants)
  noised = _energy(pdb, model_path, variants, backbone_noise=_LEVEL)
  both = _energy(pdb, model_path, variants, backbone_noise=(0.0, _LEVEL))
  assert both.shape == (2, 2), both.shape  # (C, D)
  np.testing.assert_array_equal(both[:, 0], clean)
  np.testing.assert_allclose(both[:, 1], noised, rtol=1e-6, atol=1e-6)
  assert not np.allclose(both[:, 0], both[:, 1])


def test_ddg_two_levels_add_axis(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  del registered
  options = ProtonPottsOptions(variants_json=variants)
  clean = _score(pdb, model_path, "ddg", options=options)["ddg"]
  both = _score(pdb, model_path, "ddg", options=options, backbone_noise=(0.0, _LEVEL))
  assert np.asarray(clean).shape == (1,)
  assert both["ddg"].shape == (1, 2)
  assert both["ddg_expt"].shape == (1,)  # reference values do not depend on the noise
  np.testing.assert_array_equal(both["ddg"][:, 0], clean)


def test_selectivity_two_levels_add_axis(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  del registered
  options = ProtonPottsOptions(
    binder_chain="A",
    explicit_centers=((2, "HIS-P"),),
    center_types=(),
    variants_json=variants,
  )
  clean = _score(pdb, model_path, "selectivity", options=options)
  noised = _score(pdb, model_path, "selectivity", options=options, backbone_noise=_LEVEL)
  both = _score(pdb, model_path, "selectivity", options=options, backbone_noise=(0.0, _LEVEL))
  assert clean["selectivity"].shape == (2,)
  assert clean["selectivity_per_centre"].shape == (2, 1)
  assert both["selectivity"].shape == (2, 2)
  assert both["selectivity_per_centre"].shape == (2, 2, 1)
  assert both["candidate_tokens"].shape == clean["candidate_tokens"].shape
  np.testing.assert_array_equal(both["selectivity"][:, 0], clean["selectivity"])
  np.testing.assert_allclose(both["selectivity"][:, 1], noised["selectivity"], rtol=1e-5, atol=1e-6)
  assert not np.allclose(clean["selectivity"], noised["selectivity"])


def test_schema_declares_noise_axis_only_for_two_levels(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
  variants: str,
) -> None:
  options = ProtonPottsOptions(variants_json=variants)

  def _spec(level: object) -> ScoringSpecification:
    return ScoringSpecification(
      inputs=str(pdb),
      model_family="protonpottsmpnn",
      model_local_path=model_path,
      output_kind="energy",
      protonpotts=options,
      backbone_noise=level,  # type: ignore[arg-type]
    )

  assert registered.result_schema(_spec(_LEVEL), "score:energy")["energy"].dims == ("N_cand",)
  two = registered.result_schema(_spec((0.0, _LEVEL)), "score:energy")["energy"].dims
  assert two == ("N_cand", "D_noise")


def test_ph_design_refuses_a_noise_sweep_but_takes_one_level(
  registered: ProtonPottsDriver,
  pdb: Path,
  model_path: Path,
) -> None:
  """Designs are energy-sorted and the placement plan is chosen from the noised table: no honest D axis."""
  options = ProtonPottsOptions(
    binder_chain="A",
    center_types=("HIS-P",),
    block_size=2,
    samples_per_site=2,
    neighbour_k=0,
    max_mutations=0,
    block_max_rounds=3,
    temperature=0.05,
  )

  def _spec(**kwargs: Any) -> SamplingSpecification:  # noqa: ANN401
    return SamplingSpecification(
      inputs=str(pdb),
      model_family="protonpottsmpnn",
      model_local_path=model_path,
      random_seed=42,
      protonpotts=options,
      **kwargs,
    )

  with pytest.raises(ValueError, match="noise axis"):
    registered.result_schema(_spec(backbone_noise=(0.0, _LEVEL)), "sample")
  with pytest.raises(ValueError, match="noise axis"):
    sample(_spec(backbone_noise=(0.0, _LEVEL)))
  clean = sample(_spec())["structures"]["0"]["arrays"]
  one = sample(_spec(backbone_noise=_LEVEL))["structures"]["0"]["arrays"]
  assert clean["sequence"].shape == one["sequence"].shape  # one level: unchanged shape
  assert not np.allclose(clean["final_potts_energy"], one["final_potts_energy"])
