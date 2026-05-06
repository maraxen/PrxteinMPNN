"""Tests for :mod:`prxteinmpnn.run.run_spec_portable_json`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from prxteinmpnn.run.spec import (
  AveragingConfig,
  BatchingConfig,
  GridLineageConfig,
  IOConfig,
  LigandConfig,
  MultistateConfig,
  PrecisionConfig,
  ResourceConfig,
  RunSpec,
  TiedPositionsConfig,
  build_run_spec,
)
from prxteinmpnn.run.run_spec_portable_json import (
  PORTABLE_RUN_SPEC_VERSION,
  run_spec_portable_from_dict,
  run_spec_portable_to_dict,
)
from prxteinmpnn.run.specs import RunSpecification, SamplingSpecification


def _full_run_spec(**portable: object) -> RunSpec:
  """Hand-built tree with non-portable fields distinct from placeholder defaults."""
  ms = portable.get(
    "multistate",
    MultistateConfig(mode="multi_state", n_states=2, combine_strategy="geometric_mean"),
  )
  res = portable.get(
    "resource",
    ResourceConfig(n_devices=2, sample_batch_size=16, structure_batch_size=8),
  )
  prec = portable.get("precision", PrecisionConfig(compute="bf16"))
  assert isinstance(ms, MultistateConfig)
  assert isinstance(res, ResourceConfig)
  assert isinstance(prec, PrecisionConfig)
  tree = RunSpec(
    io=IOConfig(output_dir=Path("/tmp/x"), sink_kind="hdf5", manifest_path=Path("/tmp/m.json")),
    resource=res,
    multistate=ms,
    ligand=LigandConfig(
      model_family="ligandmpnn",
      use_side_chain_context=True,
      ligand_conditioning=True,
      sidechain_conditioning=True,
      context_path=Path("/ctx"),
    ),
    tied=TiedPositionsConfig(
      tied_positions=((1, 2),),
      pass_mode="inter",
      multi_state_temperature=0.5,
      tie_group_map=None,
      structure_mapping=None,
    ),
    grid=GridLineageConfig(
      grid_mode=True,
      campaign_mode=True,
      job_id="j1",
      chunk_id=3,
      sample_start=10,
      sample_count=100,
    ),
    batching=BatchingConfig(
      batch_size=64,
      samples_batch_size=4,
      samples_chunk_size=2,
      noise_batch_size=1,
      temperature_batch_size=1,
      jacobian_batch_size=8,
      combine_batch_size=4,
      apc_batch_size=2,
      apc_residue_batch_size=500,
    ),
    averaging=AveragingConfig(
      average_node_features=True,
      average_encoding_mode="noise_levels",
      average_encodings=True,
      state_weights=None,
    ),
    precision=prec,
  )
  return cast("RunSpec", tree)


def test_portable_dict_roundtrip_preserves_subset_only() -> None:
  rs = _full_run_spec()
  d = run_spec_portable_to_dict(rs)
  assert d["version"] == PORTABLE_RUN_SPEC_VERSION
  assert d["multistate"]["mode"] == "multi_state"
  assert d["resource"]["n_devices"] == 2
  assert d["precision"]["compute"] == "bf16"

  rs2 = run_spec_portable_from_dict(d)
  assert run_spec_portable_to_dict(rs2) == d

  assert rs2.multistate == rs.multistate
  assert rs2.resource == rs.resource
  assert rs2.precision == rs.precision

  assert rs2.io.sink_kind == "none"
  assert rs2.io.output_dir is None
  assert rs2.ligand.model_family == "proteinmpnn"
  assert rs2.ligand.ligand_conditioning is False
  assert rs2.tied.pass_mode == "intra"
  assert rs2.grid.grid_mode is False
  assert rs2.batching.batch_size == 32
  assert rs2.averaging.average_node_features is False


def test_portable_roundtrip_from_build_run_spec_sampling() -> None:
  spec = SamplingSpecification(inputs=["/dev/null"], tied_positions=None, n_devices=1)
  rs = build_run_spec(spec)
  d = run_spec_portable_to_dict(rs)
  rs2 = run_spec_portable_from_dict(d)
  assert run_spec_portable_to_dict(rs2) == d
  assert rs2.multistate == rs.multistate
  assert rs2.resource == rs.resource
  assert rs2.precision == rs.precision


def test_portable_roundtrip_from_build_run_spec_run_specification() -> None:
  spec = RunSpecification(inputs=["a.pdb"], tied_positions=None, batch_size=48, n_devices=2)
  rs = build_run_spec(spec)
  d = run_spec_portable_to_dict(rs)
  rs2 = run_spec_portable_from_dict(d)
  assert run_spec_portable_to_dict(rs2) == d


_MIN_RES = {"n_devices": 1, "sample_batch_size": 1, "structure_batch_size": 1}
_MIN_PREC = {"compute": "fp32"}


@pytest.mark.parametrize(
  ("payload", "snippet"),
  [
    (
      {
        "version": 2,
        "multistate": {"mode": "a", "n_states": 1, "combine_strategy": "x"},
        "resource": _MIN_RES,
        "precision": _MIN_PREC,
      },
      "version",
    ),
    (
      {"version": 1, "resource": _MIN_RES, "precision": _MIN_PREC},
      "multistate",
    ),
    (
      {
        "version": 1,
        "multistate": {"mode": 1, "n_states": 1, "combine_strategy": "x"},
        "resource": _MIN_RES,
        "precision": _MIN_PREC,
      },
      "multistate.mode",
    ),
    (
      {
        "version": 1,
        "multistate": {"mode": "a", "n_states": True, "combine_strategy": "x"},
        "resource": _MIN_RES,
        "precision": _MIN_PREC,
      },
      "multistate.n_states",
    ),
    (
      {
        "version": 1,
        "multistate": {"mode": "a", "n_states": 1, "combine_strategy": "x"},
        "resource": [],
        "precision": _MIN_PREC,
      },
      "resource",
    ),
    (
      {
        "version": 1,
        "multistate": {"mode": "a", "n_states": 1, "combine_strategy": "x"},
        "resource": _MIN_RES,
        "precision": {"compute": "fp128"},
      },
      "precision.compute",
    ),
  ],
)
def test_from_dict_rejects_invalid(payload: dict[str, Any], snippet: str) -> None:
  with pytest.raises((ValueError, TypeError)) as ei:
    run_spec_portable_from_dict(payload)
  assert snippet in str(ei.value)
