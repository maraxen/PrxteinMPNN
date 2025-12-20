import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prxteinmpnn.training.dataloading.preprocess import (
    PreprocessingSpecification,
    _worker_process_protein,
    run_preprocessing_pipeline,
    _load_checkpoint_metadata,
)
from prxteinmpnn.utils.data_structures import ProteinTuple

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

@pytest.fixture
def dummy_protein_tuple():
    return ProteinTuple(
        coordinates=np.zeros((4, 5, 3)),
        aatype=np.zeros(4, dtype=np.int8),
        atom_mask=np.ones((4, 37)),
        residue_index=np.arange(4),
        chain_index=np.zeros(4),
        full_coordinates=np.zeros((4, 3)),
        dihedrals=None,
        source="test",
        mapping=None,
        charges=np.zeros(4),
        radii=np.zeros(4),
        sigmas=np.zeros(4),
        epsilons=np.zeros(4),
        estat_backbone_mask=None,
        estat_resid=None,
        estat_chain_index=None,
        physics_features=None,
    )

def test_worker_process_protein(tmp_path, dummy_pqr_content, dummy_protein_tuple):
    pqr_path = tmp_path / "test.pqr"
    pqr_path.write_text(dummy_pqr_content)

    output_file = tmp_path / "output.array_record"
    spec = PreprocessingSpecification(
        input_dir=tmp_path,
        output_file=output_file,
        validate_features=False,
        compute_lj=True,
        compute_estat=True,
    )

    force_field_data = {}

    with patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordWriter") as MockWriter, \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_electrostatic_node_features") as mock_estat, \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_vdw_node_features") as mock_vdw, \
         patch("prxteinmpnn.training.dataloading.preprocess.parse_pqr_to_processed_structure") as mock_parse_pqr, \
         patch("prxteinmpnn.training.dataloading.preprocess.processed_structure_to_protein_tuples") as mock_to_tuples:

        # Mock parsers
        mock_parse_pqr.return_value = MagicMock()
        mock_to_tuples.return_value = iter([dummy_protein_tuple])

        # Mock features
        mock_estat.return_value = np.zeros((4, 5))
        mock_vdw.return_value = np.zeros((4, 5))

        mock_writer_instance = MockWriter.return_value

        protein_id, shard_path = _worker_process_protein((pqr_path, spec, tmp_path, force_field_data))

        assert protein_id == "test"
        assert shard_path is not None
        assert str(shard_path).startswith(str(tmp_path))

        # Verify writer was called
        assert mock_writer_instance.write.called

        # Verify feature computations were called
        assert mock_estat.called
        assert mock_vdw.called

def test_preprocess_dataset_serial(tmp_path, mock_force_field, dummy_pqr_content, dummy_protein_tuple):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "protein1.pqr").write_text(dummy_pqr_content)

    output_file = tmp_path / "output.array_record"
    spec = PreprocessingSpecification(
        input_dir=input_dir,
        output_file=output_file,
        num_workers=0, # Serial
        resume_from_checkpoint=False,
        validate_features=False,
        compute_lj=True,
        compute_estat=True,
    )

    with patch("prxteinmpnn.training.dataloading.preprocess.load_force_field", return_value=mock_force_field), \
         patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordWriter") as MockWriter, \
         patch("prxteinmpnn.training.dataloading.preprocess.ArrayRecordReader") as MockReader, \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_electrostatic_node_features", return_value=np.zeros((4, 5))), \
         patch("prxteinmpnn.training.dataloading.preprocess.compute_vdw_node_features", return_value=np.zeros((4, 5))), \
         patch("prxteinmpnn.training.dataloading.preprocess.parse_pqr_to_processed_structure") as mock_parse_pqr, \
         patch("prxteinmpnn.training.dataloading.preprocess.processed_structure_to_protein_tuples") as mock_to_tuples, \
         patch("prxteinmpnn.training.dataloading.preprocess.msgpack.unpackb") as mock_unpack:

        mock_parse_pqr.return_value = MagicMock()
        mock_to_tuples.return_value = iter([dummy_protein_tuple])

        # Ensure the shard file is created when writer is instantiated
        def create_dummy_shard(path, *args, **kwargs):
            path_obj = pathlib.Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.touch()
            return MagicMock()

        MockWriter.side_effect = create_dummy_shard

        # Mock Reader to return something for merge phase
        mock_reader = MockReader.return_value
        mock_reader.__iter__.return_value = [b"dummy_record"]

        # Mock unpack to return protein_id
        mock_unpack.return_value = {"protein_id": "protein1"}

        result = run_preprocessing_pipeline(spec)

        assert result["success_count"] == 1
        assert result["failure_count"] == 0
        assert result["output_file"] == output_file

        # Verify metadata file was created/updated
        assert spec.metadata_file.exists()
        with spec.metadata_file.open("r") as f:
            lines = [line for line in f if line.strip()]
            assert len(lines) >= 1
            data = json.loads(lines[-1]) # Last one should be from gather phase success
            assert data["id"] == "protein1"
            assert data["status"] == "success"

@pytest.mark.skip(reason="_merge_shards_to_final integrated into run_preprocessing_pipeline")
def test_merge_shards_to_final(tmp_path):
    pass

def test_load_checkpoint_metadata(tmp_path):
    metadata_file = tmp_path / "meta.jsonl"

    # Test non-existent
    data = _load_checkpoint_metadata(metadata_file)
    assert data["total_records"] == 0

    # Test with content
    with metadata_file.open("w") as f:
        f.write(json.dumps({"id": "p1", "status": "success"}) + "\n")
        f.write(json.dumps({"id": "p2", "status": "failed"}) + "\n")
        f.write("garbage\n") # Should be skipped

    data = _load_checkpoint_metadata(metadata_file)
    assert "p1" in data["processed_files"]
    assert "p2" in data["failed_files"]
    assert data["total_records"] == 1

def test_worker_process_protein_error(tmp_path):
    pqr_path = tmp_path / "bad.pqr"
    pqr_path.write_text("INVALID CONTENT")

    spec = PreprocessingSpecification(
        input_dir=tmp_path,
        output_file=tmp_path / "out",
    )

    force_field_data = {}

    with patch("prxteinmpnn.training.dataloading.preprocess.parse_pqr_to_processed_structure") as mock_parse:
        mock_parse.side_effect = Exception("Parsing failed")
        protein_id, shard_path = _worker_process_protein((pqr_path, spec, tmp_path, force_field_data))

    assert protein_id == "bad"
    assert shard_path is None
