import pytest

from prxteinmpnn.run.specs import RunSpecification


def test_run_spec_instantiation_tied_modes():
    # Valid combinations
    dummy_inputs = ["dummy.pdb"]
    RunSpecification(inputs=dummy_inputs, tied_positions=None, pass_mode="intra")
    RunSpecification(inputs=dummy_inputs, tied_positions=None, pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions="auto", pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions="direct", pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions=[(0, 1)], pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions=[(0, 1)], pass_mode="intra")

def test_run_spec_validation_error():
    dummy_inputs = ["dummy.pdb"]
    # Invalid: tied_positions="auto" and pass_mode="intra"
    with pytest.raises(ValueError) as e1:
        RunSpecification(inputs=dummy_inputs, tied_positions="auto", pass_mode="intra")
    assert "pass_mode must be 'inter'" in str(e1.value)
    # Invalid: tied_positions="direct" and pass_mode="intra"
    with pytest.raises(ValueError) as e2:
        RunSpecification(inputs=dummy_inputs, tied_positions="direct", pass_mode="intra")
    assert "pass_mode must be 'inter'" in str(e2.value)
