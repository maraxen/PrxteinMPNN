"""Tests for the high-level Grain data loader."""

from unittest.mock import MagicMock, patch

import grain.python as grain
import pytest

from prxteinmpnn.io.loaders import create_protein_dataset
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch


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


@patch("prxteinmpnn.io.loaders.sources.MixedInputDataSource")
@patch("prxteinmpnn.io.loaders.operations.ParseStructure")
@patch("prxteinmpnn.io.loaders.operations.pad_and_collate_proteins")
def test_create_protein_dataset(
    mock_pad_and_collate: MagicMock,
    mock_parse_op: MagicMock,
    mock_source: MagicMock,
    mock_protein_batch: ProteinBatch,
) -> None:
    """Test the end-to-end creation of a protein dataset."""
    # 1. Mock the Data Source to return a real list, which behaves like a sequence.
    #    This is the key change to fix the test.
    mock_source.return_value = ["mock_raw_protein"]

    # 2. Mock the Parsing operation to process the raw data into a Protein object
    mock_protein = mock_protein_batch[0]
    mock_parse_instance = MagicMock()
    mock_parse_instance.return_value = mock_protein
    mock_parse_op.return_value = mock_parse_instance

    # 3. Mock the collate function to return the final batch
    mock_pad_and_collate.return_value = mock_protein_batch

    # Define inputs for the function under test
    inputs = ["test.pdb"]
    batch_size = 32

    # Create the dataset
    dataset = create_protein_dataset(inputs, batch_size)

    # Assert that the dataset is the correct type
    assert isinstance(dataset, grain.IterDataset)

    # Iterate the dataset to trigger the pipeline and check the output
    output_batches = list(dataset)
    assert len(output_batches) == 1
    assert output_batches[0] == mock_protein_batch

    # Check that our mocked components were called as expected
    mock_source.assert_called_with(inputs, None)
    mock_parse_op.assert_called_with(parse_kwargs={})
    mock_parse_instance.assert_called_with("mock_raw_protein")
    mock_pad_and_collate.assert_called_with([mock_protein])


def test_create_protein_dataset_with_workers() -> None:
    """Test that multiprocessing is enabled when num_workers > 0."""
    with patch.object(grain.IterDataset, "mp_prefetch") as mock_prefetch:
        create_protein_dataset(["test.pdb"], batch_size=4, num_workers=2)
        mock_prefetch.assert_called_once()