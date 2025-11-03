
import pytest
from prxteinmpnn.run.specs import RunSpecification

def test_run_spec_instantiation_tied_modes():
    """Test that RunSpecification can be successfully created with all valid combinations."""
    # Test default values
    spec = RunSpecification(inputs="test.pdb")
    assert spec.tied_positions is None
    assert spec.pass_mode == "intra"

    # Test valid combinations
    RunSpecification(inputs="test.pdb", tied_positions=None, pass_mode="intra")
    RunSpecification(inputs="test.pdb", tied_positions=None, pass_mode="inter")
    RunSpecification(inputs="test.pdb", tied_positions="auto", pass_mode="inter")
    RunSpecification(inputs="test.pdb", tied_positions="direct", pass_mode="inter")
    RunSpecification(inputs="test.pdb", tied_positions=[(0, 1)], pass_mode="intra")
    RunSpecification(inputs="test.pdb", tied_positions=[(0, 1)], pass_mode="inter")

def test_run_spec_validation_error():
    """Assert that pytest.raises(ValueError) correctly catches invalid combinations."""
    with pytest.raises(ValueError):
        RunSpecification(inputs="test.pdb", tied_positions="auto", pass_mode="intra")
    with pytest.raises(ValueError):
        RunSpecification(inputs="test.pdb", tied_positions="direct", pass_mode="intra")
