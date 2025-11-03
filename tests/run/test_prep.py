
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.run.prep import prepare_inter_mode_batch
from prxteinmpnn.run.specs import RunSpecification
from prxteinmpnn.utils.data_structures import Protein

def make_mock_protein(L, chain_ids):
    """Creates a mock Protein object for testing."""
    return Protein(
        coordinates=jnp.ones((L, 5, 3)),
        aatype=jnp.ones((L,)),
        one_hot_sequence=jnp.ones((L, 21)),
        mask=jnp.ones((L,)),
        residue_index=jnp.arange(L),
        chain_index=jnp.array(chain_ids),
    )

def test_inter_mode_batch_concatenation():
    """Test concatenation of Protein objects."""
    p1 = make_mock_protein(10, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p2 = make_mock_protein(20, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    spec = RunSpecification(inputs=["p1", "p2"], pass_mode="inter")
    
    combined, _ = prepare_inter_mode_batch([p1, p2], spec)
    
    assert combined.coordinates.shape[0] == 30
    assert combined.aatype.shape[0] == 30
    assert combined.one_hot_sequence.shape[0] == 30
    assert combined.mask.shape[0] == 30
    assert combined.residue_index.shape[0] == 30
    assert combined.chain_index.shape[0] == 30

def test_inter_mode_chain_reindexing():
    """Test chain re-indexing."""
    p1 = make_mock_protein(10, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p2 = make_mock_protein(20, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    spec = RunSpecification(inputs=["p1", "p2"], pass_mode="inter")
    
    combined, _ = prepare_inter_mode_batch([p1, p2], spec)
    
    unique_chains = np.unique(combined.chain_index)
    assert len(unique_chains) == 3
    assert all(c in unique_chains for c in [0, 1, 2])
    
def test_inter_mode_back_mapping():
    """Test back-mapping of chain IDs."""
    p1 = make_mock_protein(10, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p2 = make_mock_protein(20, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    spec = RunSpecification(inputs=["p1", "p2"], pass_mode="inter")
    
    _, inter_mode_map = prepare_inter_mode_batch([p1, p2], spec)
    
    assert inter_mode_map == {0: (0, 0), 1: (0, 1), 2: (1, 0)}

def test_prepare_inter_mode_batch_error():
    """Test that prepare_inter_mode_batch raises an error for 'intra' pass mode."""
    p1 = make_mock_protein(10, [0])
    spec = RunSpecification(inputs=["p1"], pass_mode="intra")
    
    with pytest.raises(ValueError):
        prepare_inter_mode_batch([p1], spec)
