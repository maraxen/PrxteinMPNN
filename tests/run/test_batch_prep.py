"""Tests for batch_prep."""
import jax.numpy as jnp
import pytest
from prxteinmpnn.run.batch_prep import (
    ProteinMPNNInput,
    prepare_inter_mode_batch,
)
from prxteinmpnn.run.specs import RunSpecification


@pytest.fixture
def mock_protein_mpnn_inputs():
    """Generates a mock ProteinMPNNInput object."""
    input1 = ProteinMPNNInput(
        X=jnp.ones((10, 4, 3)),
        S=jnp.zeros(10, dtype=jnp.int32),
        mask=jnp.ones(10, dtype=jnp.int32),
        chain_id=jnp.array([0] * 5 + [1] * 5, dtype=jnp.int32),
        residue_index=jnp.arange(10, dtype=jnp.int32),
        structure_to_sequence_mappings=[{}],
    )
    input2 = ProteinMPNNInput(
        X=jnp.ones((12, 4, 3)),
        S=jnp.zeros(12, dtype=jnp.int32),
        mask=jnp.ones(12, dtype=jnp.int32),
        chain_id=jnp.array([0] * 6 + [1] * 6, dtype=jnp.int32),
        residue_index=jnp.arange(12, dtype=jnp.int32),
        structure_to_sequence_mappings=[{}],
    )
    return [input1, input2]


def test_prepare_inter_mode_batch_concatenation(mock_protein_mpnn_inputs):
    """Test that the batch preparation correctly concatenates inputs."""
    spec = RunSpecification(inputs=[], pass_mode="inter")
    combined, _ = prepare_inter_mode_batch(mock_protein_mpnn_inputs, spec)

    assert combined.X.shape == (22, 4, 3)
    assert combined.S.shape == (22,)
    assert combined.mask.shape == (22,)
    assert combined.chain_id.shape == (22,)
    assert combined.residue_index.shape == (22,)


def test_prepare_inter_mode_batch_chain_remapping(mock_protein_mpnn_inputs):
    """Test that the batch preparation correctly remaps chain IDs."""
    spec = RunSpecification(inputs=[], pass_mode="inter")
    _, inter_mode_map = prepare_inter_mode_batch(mock_protein_mpnn_inputs, spec)

    expected_chain_map = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    assert inter_mode_map.chain_map == expected_chain_map


def test_prepare_inter_mode_batch_raises_value_error(mock_protein_mpnn_inputs):
    """Test that a ValueError is raised for incorrect pass_mode."""
    spec = RunSpecification(inputs=[], pass_mode="intra")
    with pytest.raises(ValueError):
        prepare_inter_mode_batch(mock_protein_mpnn_inputs, spec)
