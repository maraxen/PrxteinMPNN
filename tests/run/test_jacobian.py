"""Test suite for the jacobian module."""
from prxteinmpnn.run.jacobian import categorical_jacobian
from prxteinmpnn.run.specs import JacobianSpecification

import tempfile
import h5py

def test_categorical_jacobian_in_memory(protein_structure, mock_model_parameters):
    spec = JacobianSpecification(
        inputs="tests/data/1ubq.pdb",
        model_weights="soluble",
        average_encodings=False,
        backbone_noise=[0.0],
        extract_dihedrals=True,
    )
    results = categorical_jacobian(spec=spec)
    assert "categorical_jacobians" in results
    assert results["categorical_jacobians"] is not None

def test_categorical_jacobian_streaming(protein_structure, mock_model_parameters):
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
        spec = JacobianSpecification(
            inputs="tests/data/1ubq.pdb",
            model_weights="soluble",
            average_encodings=False,
            backbone_noise=[0.0],
            output_h5_path=tmpfile.name,
            extract_dihedrals=True,
        )
        results = categorical_jacobian(spec=spec)
        assert "output_h5_path" in results
        assert results["output_h5_path"] == tmpfile.name
        with h5py.File(tmpfile.name, "r") as f:
            assert results["spec_hash"] in f

def test_categorical_jacobian_in_memory_avg_encodings(protein_structure, mock_model_parameters):
    spec = JacobianSpecification(
        inputs="tests/data/1ubq.pdb",
        model_weights="soluble",
        average_encodings=True,
        backbone_noise=[0.0],
        extract_dihedrals=True,
    )
    results = categorical_jacobian(spec=spec)
    assert "categorical_jacobians" in results
    assert results["categorical_jacobians"] is not None
