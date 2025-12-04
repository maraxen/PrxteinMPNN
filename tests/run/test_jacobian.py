"""Test suite for the jacobian module."""
import tempfile

import chex
import h5py
import pytest

from prxteinmpnn.run.jacobian import categorical_jacobian
from prxteinmpnn.run.specs import JacobianSpecification


@pytest.mark.skip(reason="Jacobian tests are too slow and time out.")
def test_categorical_jacobian_in_memory(protein_structure, mock_model_parameters):
    spec = JacobianSpecification(
        inputs="tests/data/1ubq.pdb",
        model_weights="soluble",
        average_encodings=False,
        backbone_noise=[0.0],
    )
    results = categorical_jacobian(spec=spec)
    chex.assert_tree_all_finite(results["categorical_jacobians"])
    assert "categorical_jacobians" in results
    assert results["categorical_jacobians"] is not None

@pytest.mark.skip(reason="Jacobian tests are too slow and time out.")
def test_categorical_jacobian_streaming(protein_structure, mock_model_parameters):
  with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
    spec = JacobianSpecification(
      inputs="tests/data/1ubq.pdb",
      model_weights="soluble",
      average_encodings=False,
      backbone_noise=[0.0],
      output_h5_path=tmpfile.name,
    )
    results = categorical_jacobian(spec=spec)
    assert "output_h5_path" in results
    assert results["output_h5_path"] == tmpfile.name
    with h5py.File(tmpfile.name, "r") as f:
      assert results["spec_hash"] in f

@pytest.mark.skip(reason="Jacobian tests are too slow and time out.")
def test_categorical_jacobian_in_memory_avg_encodings(protein_structure, mock_model_parameters):
    spec = JacobianSpecification(
        inputs="tests/data/1ubq.pdb",
        model_weights="soluble",
        average_encodings=True,
        backbone_noise=[0.0],
    )
    results = categorical_jacobian(spec=spec)
    assert "categorical_jacobians" in results
    assert results["categorical_jacobians"] is not None
