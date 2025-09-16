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
    # Mock the return values of the components
    mock_source_instance = MagicMock()
    mock_source.return_value = mock_source_instance

    mock_parse_instance = MagicMock()
    mock_parse_op.return_value = mock_parse_instance

    mock_pad_and_collate.return_value = mock_protein_batch

    # Define some inputs
    inputs = ["test.pdb"]
    batch_size = 32

    # Create the dataset
    dataset = create_protein_dataset(inputs, batch_size)

    # Assert that the dataset is an IterDataset
    assert isinstance(dataset, grain.IterDataset)

    # To test the pipeline, we can iterate through it.
    # We need to mock the underlying data source to return something.
    # Let's assume the source would yield two items that the parser handles.
    with patch.object(grain.MapDataset, "source") as mock_map_source:
        # Create a mock MapDataset that we can control
        mock_map_ds = MagicMock()
        mock_map_source.return_value = mock_map_ds

        # Simulate the pipeline steps
        mock_map_ds.map.return_value.filter.return_value.batch.return_value.to_iter_dataset.return_value = [
            mock_protein_batch
        ]

        # Re-create the dataset to use the patched source
        dataset = create_protein_dataset(inputs, batch_size)

        # Iterate and check the output
        output_batches = list(dataset)
        assert len(output_batches) == 1
        assert output_batches[0] == mock_protein_batch

        # Check that our components were called
        mock_source.assert_called_with(inputs, None)
        mock_parse_op.assert_called_with(parse_kwargs={})
        # The batch_fn is not directly callable in this mock setup,
        # but we can check that the batch method was called with it.
        mock_map_ds.map.return_value.filter.return_value.batch.assert_called_with(
            batch_size, batch_fn=mock_pad_and_collate
        )


def test_create_protein_dataset_with_workers() -> None:
    """Test that multiprocessing is enabled when num_workers > 0."""
    with patch.object(grain.IterDataset, "mp_prefetch") as mock_prefetch:
        create_protein_dataset(["test.pdb"], batch_size=4, num_workers=2)
        mock_prefetch.assert_called_once()
