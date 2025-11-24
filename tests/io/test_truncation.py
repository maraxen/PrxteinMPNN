
import numpy as np
import pytest
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.io.operations import truncate_protein

def create_dummy_protein(length=100):
    return ProteinTuple(
        coordinates=np.zeros((length, 37, 3)),
        aatype=np.zeros((length,), dtype=np.int32),
        atom_mask=np.ones((length, 37)),
        residue_index=np.arange(length),
        chain_index=np.zeros((length,), dtype=np.int32),
        full_coordinates=np.zeros((length, 37, 3)), # Optional
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

def test_truncation_center_crop():
    p = create_dummy_protein(100)
    cropped_center = truncate_protein(p, max_length=50, strategy="center_crop")
    assert cropped_center.coordinates.shape[0] == 50
    # Center of 100 is 50. Range [25, 75).
    assert cropped_center.residue_index[0] == 25 
    assert cropped_center.residue_index[-1] == 74

def test_truncation_random_crop():
    p = create_dummy_protein(100)
    # Mock numpy random to ensure deterministic behavior for test if needed, 
    # but for simple length check it's fine.
    # To test randomness we might need to seed or mock.
    cropped_random = truncate_protein(p, max_length=50, strategy="random_crop")
    assert cropped_random.coordinates.shape[0] == 50
    assert 0 <= cropped_random.residue_index[0] <= 50

def test_no_truncation_needed():
    p = create_dummy_protein(100)
    cropped_none = truncate_protein(p, max_length=150, strategy="random_crop")
    assert cropped_none.coordinates.shape[0] == 100

def test_strategy_none():
    p = create_dummy_protein(100)
    cropped_strat_none = truncate_protein(p, max_length=50, strategy="none")
    assert cropped_strat_none.coordinates.shape[0] == 100

def test_invalid_strategy():
    p = create_dummy_protein(100)
    with pytest.raises(ValueError, match="Unknown truncation strategy"):
        truncate_protein(p, max_length=50, strategy="invalid")
