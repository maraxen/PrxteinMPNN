"""JSON round-trip for run specifications."""

from __future__ import annotations

import json

import pytest

from prxteinmpnn.run.spec_json import (
  SpecJSONDecodeError,
  SpecJSONEncodeError,
  run_specification_from_json,
  run_specification_from_json_dict,
  run_specification_to_json,
  run_specification_to_json_dict,
)
from prxteinmpnn.run.specs import RunSpecification, SamplingSpecification, ScoringSpecification


def test_scoring_spec_json_roundtrip() -> None:
  spec = ScoringSpecification(
    inputs=["/tmp/struct.pdb"],
    sequences_to_score=["ACDE", "FGHI"],
    batch_size=4,
    output_h5_path="/tmp/out.h5",
  )
  blob = run_specification_to_json(spec, indent=None)
  restored = run_specification_from_json(blob)
  assert isinstance(restored, ScoringSpecification)
  assert restored.inputs == spec.inputs
  assert list(restored.sequences_to_score) == list(spec.sequences_to_score)
  assert restored.batch_size == spec.batch_size
  assert restored.output_h5_path == spec.output_h5_path


def test_sampling_spec_dict_has_discriminator() -> None:
  spec = SamplingSpecification(
    inputs=["x.pdb"],
    num_samples=2,
    backbone_noise=(0.0, 0.1),
    temperature=(0.2, 0.3),
  )
  d = run_specification_to_json_dict(spec)
  assert d["_spec_class"] == "SamplingSpecification"
  assert d["num_samples"] == 2
  assert d["backbone_noise"] == [0.0, 0.1]
  again = run_specification_from_json_dict(d)
  assert isinstance(again, SamplingSpecification)
  assert again.num_samples == 2
  assert tuple(again.backbone_noise) == (0.0, 0.1)


def test_run_specification_tied_positions_lists() -> None:
  spec = RunSpecification(
    inputs=["a.pdb"],
    tied_positions=[(0, 1), (2, 3)],
    pass_mode="inter",
  )
  d = run_specification_to_json_dict(spec)
  assert d["tied_positions"] == [[0, 1], [2, 3]]
  restored = run_specification_from_json_dict(d)
  assert restored.tied_positions == [(0, 1), (2, 3)]


def test_encode_rejects_text_stream_inputs() -> None:
  from io import StringIO

  spec = RunSpecification(inputs=StringIO("ATOM"), tied_positions=None)
  with pytest.raises(SpecJSONEncodeError, match="str"):
    run_specification_to_json_dict(spec)


def test_json_root_must_be_object() -> None:
  with pytest.raises(SpecJSONDecodeError, match="object"):
    run_specification_from_json(json.dumps(["not", "an", "object"]))
