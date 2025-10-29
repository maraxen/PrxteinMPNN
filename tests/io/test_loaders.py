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
        mask=MagicMock(),
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


@patch("prxteinmpnn.io.loaders.dataset.ProteinDataSource")
@patch("prxteinmpnn.io.loaders.operations.pad_and_collate_proteins")
def test_create_protein_dataset(
    mock_pad_and_collate: MagicMock,
    mock_source_class: MagicMock,
    mock_protein_tuple: ProteinTuple,
    mock_protein_batch: ProteinBatch,
) -> None:
    """Test the end-to-end creation of a protein dataset."""
    # 1. Setup mocks
    mock_source_instance = MagicMock(spec=grain.RandomAccessDataSource)
    mock_source_instance.__len__.return_value = 1
    mock_source_instance.__getitem__.return_value = mock_protein_tuple
    mock_source_class.return_value = mock_source_instance

    mock_pad_and_collate.return_value = mock_protein_batch

    # 2. Define inputs
    inputs = ["test.pdb"]
    batch_size = 1

    # 3. Call the function
    dataset = create_protein_dataset(inputs, batch_size, parse_kwargs={})

    # 4. Assertions
    assert isinstance(dataset, grain.IterDataset)

    mock_source_class.assert_called_once_with(
        inputs=inputs,
        parse_kwargs={},
        foldcomp_database=None,
    )

    # Iterate to trigger pipeline
    output_batches = list(dataset)

    mock_pad_and_collate.assert_called_once_with([mock_protein_tuple])

    assert len(output_batches) == 1
    assert output_batches[0] == mock_protein_batch


@patch("prxteinmpnn.io.loaders.dataset.ProteinDataSource")
@patch("prxteinmpnn.io.loaders.prefetch_autotune.pick_performance_config")
@patch.object(grain.MapDataset, "to_iter_dataset")
def test_create_protein_dataset_with_workers(
    mock_to_iter_dataset: MagicMock,
    mock_pick_perf: MagicMock,
    mock_source_class: MagicMock,
) -> None:
    """Test that multiprocessing is enabled when num_workers > 0."""
    mock_source_instance = MagicMock(spec=grain.RandomAccessDataSource)
    mock_source_instance.__len__.return_value = 1
    mock_source_class.return_value = mock_source_instance

    mock_iter_dataset = MagicMock(spec=grain.IterDataset)
    mock_to_iter_dataset.return_value = mock_iter_dataset

    mock_perf_config = MagicMock()
    mock_perf_config.read_options = grain.ReadOptions(
        num_threads=4, prefetch_buffer_size=1
    )
    mock_pick_perf.return_value = mock_perf_config

    create_protein_dataset(["test.pdb"], batch_size=4)

    mock_to_iter_dataset.assert_called_once_with(
        read_options=mock_perf_config.read_options
    )
