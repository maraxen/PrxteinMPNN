"""Tests for aminx.run public exports (RS-4 spec-driven routing + tensor deprecation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import pytest

from aminx.run import InspectionSpecification, sample, score
from aminx.run.specs import SamplingSpecification, ScoringSpecification


class _DummyModel(eqx.Module):
  """Minimal eqx.Module stand-in for tensor-signature detection."""


class TestRunExports:
  """AC-RS-4: spec-driven sample/score via host.runner; tensor path deprecated."""

  def test_inspection_specification_importable_from_run_package(self) -> None:
    assert InspectionSpecification is not None
    spec = InspectionSpecification(inputs="test.pdb")
    assert spec.inputs == "test.pdb"

  def test_sample_with_spec_routes_to_host_runner(self) -> None:
    spec = SamplingSpecification(inputs="test.pdb")
    sentinel = {"sequences": "dict-result"}
    with patch("aminx.host.runner.sample", return_value=sentinel) as mock_runner:
      result = sample(spec)
    mock_runner.assert_called_once_with(spec)
    assert result is sentinel

  def test_sample_with_spec_kwargs_routes_to_host_runner(self) -> None:
    sentinel = {"schema_version": "sampling_v1"}
    with patch("aminx.host.runner.sample", return_value=sentinel) as mock_runner:
      result = sample(inputs="test.pdb")
    mock_runner.assert_called_once_with(inputs="test.pdb")
    assert result is sentinel

  def test_score_with_spec_routes_to_host_runner(self) -> None:
    spec = ScoringSpecification(
      inputs="test.pdb",
      sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"],
    )
    sentinel = {"scores": "dict-result"}
    with patch("aminx.host.runner.score", return_value=sentinel) as mock_runner:
      result = score(spec)
    mock_runner.assert_called_once_with(spec)
    assert result is sentinel

  def test_score_with_spec_kwargs_routes_to_host_runner(self) -> None:
    sentinel = {"schema_version": "scoring_v1"}
    with patch("aminx.host.runner.score", return_value=sentinel) as mock_runner:
      result = score(inputs="test.pdb", sequences_to_score=["MKVL"])
    mock_runner.assert_called_once_with(inputs="test.pdb", sequences_to_score=["MKVL"])
    assert result is sentinel

  def test_sample_tensor_call_emits_deprecation_and_routes_to_sampling(self) -> None:
    key = jax.random.PRNGKey(0)
    model = _DummyModel()
    coords = jax.numpy.zeros((4, 4, 3))
    mask = jax.numpy.ones(4, dtype=jax.numpy.int32)
    residue_index = jax.numpy.arange(4)
    chain_index = jax.numpy.zeros(4, dtype=jax.numpy.int32)
    tensor_result = (MagicMock(), MagicMock(), MagicMock())
    with (
      patch("aminx.sampling.sample", return_value=tensor_result) as mock_tensor,
      pytest.warns(DeprecationWarning, match="aminx.sampling.sample"),
    ):
      result = sample(key, model, coords, mask, residue_index, chain_index)
    mock_tensor.assert_called_once_with(key, model, coords, mask, residue_index, chain_index)
    assert result is tensor_result

  def test_score_tensor_call_emits_deprecation_and_routes_to_scoring(self) -> None:
    key = jax.random.PRNGKey(1)
    model = _DummyModel()
    coords = jax.numpy.zeros((4, 4, 3))
    mask = jax.numpy.ones(4, dtype=jax.numpy.int32)
    residue_index = jax.numpy.arange(4)
    chain_index = jax.numpy.zeros(4, dtype=jax.numpy.int32)
    sequence = jax.numpy.zeros(4, dtype=jax.numpy.int32)
    tensor_result = (MagicMock(), MagicMock(), MagicMock())
    with (
      patch("aminx.scoring.score.score", return_value=tensor_result) as mock_tensor,
      pytest.warns(DeprecationWarning, match="aminx.scoring.score"),
    ):
      result = score(key, model, coords, mask, residue_index, chain_index, sequence)
    mock_tensor.assert_called_once_with(
      key, model, coords, mask, residue_index, chain_index, sequence
    )
    assert result is tensor_result

  def test_tensor_apis_remain_importable_from_native_modules(self) -> None:
    from aminx.sampling import sample as sampling_sample
    from aminx.scoring.score import score as scoring_score

    assert sampling_sample.__module__ == "aminx.sampling"
    assert scoring_score.__module__ == "aminx.scoring.score"
