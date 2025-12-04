
import numpy as np
import pytest
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.io.operations import pad_and_collate_proteins

def create_dummy_protein(length=100):
    return ProteinTuple(
        coordinates=np.zeros((length, 37, 3)),
        aatype=np.zeros((length,), dtype=np.int32),
        atom_mask=np.ones((length, 37)),
        residue_index=np.arange(length),
        chain_index=np.zeros((length,), dtype=np.int32),
        full_coordinates=None,
        dihedrals=None,
        source="test",
        mapping=None,
        charges=None,
        radii=None,
        sigmas=None,
        epsilons=None,
        estat_backbone_mask=None,
        estat_resid=None,
        estat_chain_index=None,
        physics_features=None,
    )

def test_fixed_padding_with_max_length():
    """Test that all batches are padded to the same fixed max_length."""
    # Create proteins of varying lengths
    p1 = create_dummy_protein(100)
    p2 = create_dummy_protein(200)
    p3 = create_dummy_protein(150)
    
    # Batch with max_length=512
    batch = pad_and_collate_proteins([p1, p2, p3], max_length=512)
    
    # All proteins should be padded to 512
    assert batch.coordinates.shape[1] == 512, f"Expected shape[1]=512, got {batch.coordinates.shape[1]}"
    assert batch.coordinates.shape[0] == 3, f"Expected batch size 3, got {batch.coordinates.shape[0]}"

def test_variable_padding_without_max_length():
    """Test that without max_length, padding uses max in batch."""
    p1 = create_dummy_protein(100)
    p2 = create_dummy_protein(200)
    
    batch = pad_and_collate_proteins([p1, p2], max_length=None)
    
    # Should pad to max in batch (200)
    assert batch.coordinates.shape[1] == 200

def test_different_batches_same_shape_with_max_length():
    """Test that different batches have the same shape when max_length is set."""
    # Batch 1
    batch1 = pad_and_collate_proteins(
        [create_dummy_protein(100), create_dummy_protein(150)],
        max_length=512
    )
    
    # Batch 2 with different lengths
    batch2 = pad_and_collate_proteins(
        [create_dummy_protein(300), create_dummy_protein(400)],
        max_length=512
    )
    
    # Both batches should have the same shape
    assert batch1.coordinates.shape[1] == batch2.coordinates.shape[1] == 512
