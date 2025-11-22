import json
import pathlib
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest
import jax.numpy as jnp

from prxteinmpnn.training.dataloading.preprocess import (
    PreprocessingSpecification,
    _merge_shards_to_final,
    _worker_process_protein,
    preprocess_dataset,
    _load_checkpoint_metadata,
)

@pytest.fixture
def mock_force_field():
    ff = MagicMock()
    ff.charges_by_id = [0.1] * 10
    ff.sigmas_by_id = [1.0] * 10
    ff.epsilons_by_id = [0.5] * 10
    ff.atom_key_to_id = {"N": 0, "C": 1, "O": 2, "CA": 3}
    return ff

@pytest.fixture
def dummy_pqr_content():
    return """ATOM      1  N   ALA A   1      10.000  10.000  10.000  0.1000 1.5000
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  0.2000 1.8000
ATOM      3  C   ALA A   1      12.000  10.000  10.000  0.3000 1.7000
ATOM      4  O   ALA A   1      13.000  10.000  10.000 -0.3000 1.6000
END
"""

def test_worker_process_protein(tmp_path, dummy_pqr_content):
    pqr_path = tmp_path / "test.pqr"
    pqr_path.write_text(dummy_pqr_content)

    output_file = tmp_path / "output.array_record"
    spec = PreprocessingSpecification(
        input_dir=tmp_path,
        output_file=output_file,
        validate_features=False
    )

    force_field_data = {
        "charges": np.array([0.1] * 10),
        "sigmas": np.array([1.0] * 10),
        "epsilons": np.array([0.5] * 10),
        "atom_key_to_id": {"N": 0, "C": 1, "O": 2, "CA": 3}
    }

    with patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordWriter") as MockWriter, \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_electrostatic_node_features") as mock_estat:

        # Mock compute_electrostatic_node_features to return dummy features
        # It returns (N, 5) array
        mock_estat.return_value = np.zeros((4, 5))

        mock_writer_instance = MockWriter.return_value

        protein_id, shard_path = _worker_process_protein((pqr_path, spec, tmp_path, force_field_data))

        assert protein_id == "test"
        assert shard_path is not None
        # The function constructs shard path using temp_dir_path (tmp_path)
        assert str(shard_path).startswith(str(tmp_path))

        # Verify writer was called
        assert mock_writer_instance.write.called

        # Verify compute_electrostatic_node_features was called
        assert mock_estat.called

def test_preprocess_dataset_serial(tmp_path, mock_force_field, dummy_pqr_content):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "protein1.pqr").write_text(dummy_pqr_content)

    output_file = tmp_path / "output.array_record"
    spec = PreprocessingSpecification(
        input_dir=input_dir,
        output_file=output_file,
        num_workers=0, # Serial
        resume_from_checkpoint=False,
        validate_features=False
    )

    with patch("prxteinmpnn.training.dataloading.preprocess.load_force_field_from_hub", return_value=mock_force_field), \
         patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordWriter") as MockWriter, \
         patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordReader") as MockReader, \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_electrostatic_node_features", return_value=np.zeros((4, 5))), \
         patch("prxteinmpnn.training.dataloading.preprocess.msgpack.unpackb") as mock_unpack:

        # Ensure the shard file is created when writer is instantiated
        def create_dummy_shard(path, *args, **kwargs):
            path_obj = pathlib.Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.touch()
            return MagicMock()

        MockWriter.side_effect = create_dummy_shard

        # Mock Reader to return something for merge phase
        mock_reader = MockReader.return_value
        mock_reader.num_records.return_value = 1
        mock_reader.read.return_value = [b"dummy_record"]

        # Mock unpack to return protein_id
        mock_unpack.return_value = {"protein_id": "protein1"}

        result = preprocess_dataset(spec)

        assert result["num_proteins"] == 1
        assert result["num_failed"] == 0
        assert result["output_file"] == output_file

        # Verify metadata file was created/updated
        assert spec.metadata_file.exists()
        with spec.metadata_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) >= 1
            data = json.loads(lines[0])
            assert data["protein_id"] == "protein1"
            assert data["status"] == "success"

def test_merge_shards_to_final(tmp_path):
    shard1 = tmp_path / "shard1"
    shard1.touch()
    shard2 = tmp_path / "shard2"
    shard2.touch()

    output_file = tmp_path / "final.array_record"
    metadata_file = tmp_path / "metadata.jsonl"
    index_file = tmp_path / "index.json"

    with patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordWriter") as MockWriter, \
         patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordReader") as MockReader, \
         patch("prxteinmpnn.training.dataloading.preprocess.msgpack.unpackb") as mock_unpack:

         mock_reader = MockReader.return_value
         mock_reader.num_records.return_value = 1
         mock_reader.read.return_value = [b"data"]

         # Mock msgpack unpack to return protein_id for indexing
         mock_unpack.side_effect = [{"protein_id": "p1"}, {"protein_id": "p2"}]

         index = _merge_shards_to_final([shard1, shard2], output_file, metadata_file, index_file, "zstd", 1)

         assert MockWriter.call_count == 1
         # Reader called for each shard
         assert MockReader.call_count == 2

         assert index == {"p1": 0, "p2": 1}
         assert index_file.exists()

         # Check metadata
         assert metadata_file.exists()
         with metadata_file.open() as f:
             lines = f.readlines()
             assert len(lines) == 2
             assert json.loads(lines[0])["protein_id"] == "p1"
             assert json.loads(lines[1])["protein_id"] == "p2"

def test_load_checkpoint_metadata(tmp_path):
    metadata_file = tmp_path / "meta.jsonl"

    # Test non-existent
    data = _load_checkpoint_metadata(metadata_file)
    assert data["total_records"] == 0

    # Test with content
    with metadata_file.open("w") as f:
        f.write(json.dumps({"protein_id": "p1", "status": "success"}) + "\n")
        f.write(json.dumps({"protein_id": "p2", "status": "failed"}) + "\n")
        f.write("garbage\n") # Should be skipped

    data = _load_checkpoint_metadata(metadata_file)
    assert "p1" in data["processed_files"]
    assert "p2" in data["failed_files"]
    assert data["total_records"] == 1

def test_worker_process_protein_error(tmp_path):
    # Test with invalid PQR content
    pqr_path = tmp_path / "bad.pqr"
    pqr_path.write_text("INVALID CONTENT")

    spec = PreprocessingSpecification(
        input_dir=tmp_path,
        output_file=tmp_path / "out",
    )

    # We mock force field data
    force_field_data = {}

    # It should return None as path
    protein_id, shard_path = _worker_process_protein((pqr_path, spec, tmp_path, force_field_data))

    assert protein_id == "bad"
    assert shard_path is None
