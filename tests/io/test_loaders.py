"""Tests for the high-level Grain data loader."""

from unittest.mock import MagicMock, patch, ANY

import grain.python as grain
import pytest
import jax.numpy as jnp

from prxteinmpnn.io.loaders import create_protein_dataset
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch, ProteinTuple


@pytest.fixture
def mock_protein_batch() -> ProteinBatch:
    """Fixture for a mock ProteinBatch."""
    protein = Protein(
        coordinates=MagicMock(),
        aatype=MagicMock(),
        one_hot_sequence=MagicMock(),
        atom_mask=MagicMock(),
        residue_index=MagicMock(),
        chain_index=MagicMock(),
    )
    return [protein]


@pytest.fixture
def mock_protein_tuple() -> ProteinTuple:
    """Fixture for a mock ProteinTuple."""
    return ProteinTuple(
        coordinates=MagicMock(),
        aatype=MagicMock(),
        atom_mask=MagicMock(),
        residue_index=MagicMock(),
        chain_index=MagicMock(),
        full_coordinates=None,
        dihedrals=None,
        source=None,
        mapping=None,
    )


@patch("prxteinmpnn.io.loaders.preprocess_inputs_to_hdf5")
@patch("prxteinmpnn.io.loaders.sources.HDF5DataSource")
@patch("prxteinmpnn.io.loaders.operations.LoadHDF5Frame")
@patch("prxteinmpnn.io.loaders.operations.pad_and_collate_proteins")
def test_create_protein_dataset(
    mock_pad_and_collate: MagicMock,
    mock_load_op_class: MagicMock,
    mock_source_class: MagicMock,
    mock_preprocess: MagicMock,
    mock_protein_tuple: ProteinTuple,
    mock_protein_batch: ProteinBatch,
) -> None:
    """Test the end-to-end creation of a protein dataset."""
    # 1. Setup mocks
    dummy_h5_path = "/tmp/dummy.h5"
    mock_preprocess.return_value = dummy_h5_path

    mock_source_instance = MagicMock(spec=grain.RandomAccessDataSource)
    mock_source_instance.__len__.return_value = 1
    mock_source_instance.__getitem__.return_value = 0
    mock_source_class.return_value = mock_source_instance

    mock_load_op_instance = MagicMock()
    mock_load_op_instance.return_value = mock_protein_tuple
    mock_load_op_class.return_value = mock_load_op_instance

    mock_pad_and_collate.return_value = mock_protein_batch

    # 2. Define inputs
    inputs = ["test.pdb"]
    batch_size = 1

    # 3. Call the function
    dataset = create_protein_dataset(inputs, batch_size, cache_path=None)

    # 4. Assertions
    assert isinstance(dataset, grain.IterDataset)

    mock_preprocess.assert_called_once_with(
        inputs,
        output_path=None,
        foldcomp_database=None,
        parse_kwargs={},
    )
    mock_source_class.assert_called_once_with(dummy_h5_path)
    mock_load_op_class.assert_called_once_with(hdf5_path=dummy_h5_path)

    # Iterate to trigger pipeline
    output_batches = list(dataset)

    mock_load_op_instance.assert_called_once_with(0)
    mock_pad_and_collate.assert_called_once_with([mock_protein_tuple])

    assert len(output_batches) == 1
    assert output_batches[0] == mock_protein_batch


@patch("prxteinmpnn.io.loaders.preprocess_inputs_to_hdf5")
def test_create_protein_dataset_with_workers(mock_preprocess: MagicMock) -> None:
    """Test that multiprocessing is enabled when num_workers > 0."""
    mock_preprocess.return_value = "/tmp/dummy.h5"
    with patch("prxteinmpnn.io.loaders.sources.HDF5DataSource") as mock_source_class, \
         patch.object(grain.IterDataset, "mp_prefetch") as mock_prefetch:
        mock_source_instance = MagicMock()
        mock_source_instance.__len__.return_value = 1
        mock_source_class.return_value = mock_source_instance

        create_protein_dataset(["test.pdb"], batch_size=4, num_workers=2)
        mock_prefetch.assert_called_once_with(ANY)
        args, kwargs = mock_prefetch.call_args
        assert isinstance(args[0], grain.MultiprocessingOptions)
        assert args[0].num_workers == 2