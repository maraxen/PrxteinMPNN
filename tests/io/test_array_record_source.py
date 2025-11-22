import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import msgpack
import msgpack_numpy as m

from prxteinmpnn.io.array_record_source import ArrayRecordDataSource

m.patch()

@pytest.fixture
def mock_index_content():
    return {
        "p1": {"idx": [0], "set": "train"},
        "p2": {"idx": [1], "set": "valid"},
        "p3": {"idx": [2, 3], "set": "train"},
    }

@pytest.fixture
def mock_record_data():
    # Minimal record data
    return {
        "coordinates": np.zeros((10, 37, 3), dtype=np.float32),
        "aatype": np.zeros((10,), dtype=np.int8),
        "atom_mask": np.ones((10, 37), dtype=bool),
        "residue_index": np.arange(10, dtype=np.int32),
        "chain_index": np.zeros((10,), dtype=np.int32),
        "source_file": "test.pqr",
        # Missing full_coordinates and physics_features to test robustness
    }

def test_array_record_source_splits(tmp_path, mock_index_content, mock_record_data):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(mock_index_content))
    
    record_path = tmp_path / "data.array_record"
    record_path.touch()

    with patch("prxteinmpnn.io.array_record_source.ArrayRecordReader") as MockReader:
        mock_reader = MockReader.return_value
        # We have 4 records total (0, 1, 2, 3)
        mock_reader.num_records.return_value = 4
        
        # Mock read to return packed data
        packed_data = msgpack.packb(mock_record_data, use_bin_type=True)
        mock_reader.read.return_value = [packed_data]

        # Test Train Split
        ds_train = ArrayRecordDataSource(record_path, index_path, split="train")
        # p1 (1 record) + p3 (2 records) = 3 records
        assert len(ds_train) == 3
        
        # Test Valid Split
        ds_valid = ArrayRecordDataSource(record_path, index_path, split="valid")
        # p2 (1 record) = 1 record
        assert len(ds_valid) == 1
        
        # Test Test Split (empty)
        ds_test = ArrayRecordDataSource(record_path, index_path, split="test")
        assert len(ds_test) == 0

def test_array_record_source_robustness(tmp_path, mock_index_content, mock_record_data):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(mock_index_content))
    
    record_path = tmp_path / "data.array_record"
    record_path.touch()

    with patch("prxteinmpnn.io.array_record_source.ArrayRecordReader") as MockReader:
        mock_reader = MockReader.return_value
        mock_reader.num_records.return_value = 4
        packed_data = msgpack.packb(mock_record_data, use_bin_type=True)
        mock_reader.read.return_value = [packed_data]

        ds = ArrayRecordDataSource(record_path, index_path, split="train")
        
        # Load a record
        protein = ds[0]
        
        # Check that missing fields were filled with defaults
        assert protein.physics_features.shape == (10, 5)
        assert np.all(protein.physics_features == 0)
        
        assert protein.full_coordinates.shape == (0, 3)
        assert protein.charges.shape == (0,)
