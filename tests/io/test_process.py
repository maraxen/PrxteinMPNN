"""Tests for prxteinmpnn.io.process."""

from io import StringIO
from unittest.mock import MagicMock, patch

import anyio
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.process import load
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture
def dummy_protein() -> Protein:
  """Create a dummy Protein for testing."""
  return Protein(
    coordinates=jnp.zeros((10, 37, 3)),
    aatype=jnp.zeros(10, dtype=jnp.int8),
    atom_mask=jnp.ones((10, 37)),
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros(10, dtype=jnp.int32),
  )


async def async_gen_from_list(items):
  """Helper to create an async generator from a list."""
  for item in items:
    yield item


@pytest.mark.anyio
class TestLoadFunction:
  """Tests for the load function."""

  async def test_load_single_input(self, dummy_protein):
    """Test loading from a single valid input string."""
    mock_handler = MagicMock()
    mock_handler.process.return_value = async_gen_from_list([(dummy_protein, "file.pdb")])

    with patch("prxteinmpnn.io.process.get_source_handler", return_value=mock_handler) as mock_get_handler, patch(
      "prxteinmpnn.io.process.ProcessPoolExecutor",
    ):
      inputs = "file.pdb"
      results = [p async for p in load(inputs)]

      mock_get_handler.assert_called_once_with(inputs, **{})
      mock_handler.process.assert_called_once()
      assert len(results) == 1
      assert results[0] == (dummy_protein, "file.pdb")

  async def test_load_stringio_input(self, dummy_protein):
    """Test loading from a single StringIO input."""
    mock_handler = MagicMock()
    mock_handler.process.return_value = async_gen_from_list([(dummy_protein, "StringIO")])

    string_io_input = StringIO("PDB data")

    with patch("prxteinmpnn.io.process.get_source_handler", return_value=mock_handler) as mock_get_handler, patch(
      "prxteinmpnn.io.process.ProcessPoolExecutor",
    ):
      results = [p async for p in load(string_io_input)]

      mock_get_handler.assert_called_once_with(string_io_input, **{})
      mock_handler.process.assert_called_once()
      assert len(results) == 1
      assert results[0] == (dummy_protein, "StringIO")

  async def test_load_multiple_inputs(self, dummy_protein):
    """Test loading from a list of multiple inputs."""
    mock_handler1 = MagicMock()
    mock_handler1.process.return_value = async_gen_from_list([(dummy_protein, "file1.pdb")])

    mock_handler2 = MagicMock()
    mock_handler2.process.return_value = async_gen_from_list([(dummy_protein, "file2.pdb")])

    def side_effect(item, **kwargs):
      if item == "file1.pdb":
        return mock_handler1
      if item == "file2.pdb":
        return mock_handler2
      return None

    with patch("prxteinmpnn.io.process.get_source_handler", side_effect=side_effect) as mock_get_handler, patch(
      "prxteinmpnn.io.process.ProcessPoolExecutor",
    ):
      inputs = ["file1.pdb", "file2.pdb"]
      results = [p async for p in load(inputs)]

      assert mock_get_handler.call_count == 2
      mock_handler1.process.assert_called_once()
      mock_handler2.process.assert_called_once()
      assert len(results) == 2
      # Note: order is preserved due to sequential processing
      assert results[0] == (dummy_protein, "file1.pdb")
      assert results[1] == (dummy_protein, "file2.pdb")

  async def test_load_foldcomp_ids(self, dummy_protein):
    """Test loading from FoldComp IDs."""
    mock_foldcomp_handler = MagicMock()
    mock_foldcomp_handler.process.return_value = async_gen_from_list(
      [
        (dummy_protein, "AF-P12345-F1-model_v4"),
        (dummy_protein, "AF-Q67890-F1-model_v4"),
      ],
    )

    with patch("prxteinmpnn.io.process.get_source_handler", return_value=None), patch(
      "prxteinmpnn.io.process.FoldCompSource",
      return_value=mock_foldcomp_handler,
    ) as mock_foldcomp_source_class, patch("prxteinmpnn.io.process.ProcessPoolExecutor"):
      inputs = ["AF-P12345-F1-model_v4", "AF-Q67890-F1-model_v4"]
      results = [p async for p in load(inputs, foldcomp_database="afdb_rep_v4")]

      mock_foldcomp_source_class.assert_called_once_with(inputs, "afdb_rep_v4", **{})
      mock_foldcomp_handler.process.assert_called_once()
      assert len(results) == 2

  async def test_load_mixed_inputs(self, dummy_protein):
    """Test loading from a mix of file paths and FoldComp IDs."""
    mock_file_handler = MagicMock()
    mock_file_handler.process.return_value = async_gen_from_list([(dummy_protein, "file1.pdb")])

    mock_foldcomp_handler = MagicMock()
    mock_foldcomp_handler.process.return_value = async_gen_from_list([(dummy_protein, "AF-P12345-F1-model_v4")])

    with patch("prxteinmpnn.io.process.get_source_handler", return_value=mock_file_handler) as mock_get_handler, patch(
      "prxteinmpnn.io.process.FoldCompSource",
      return_value=mock_foldcomp_handler,
    ) as mock_foldcomp_source_class, patch("prxteinmpnn.io.process.ProcessPoolExecutor"):
      foldcomp_id = "AF-P12345-F1-model_v4"
      inputs = ["file1.pdb", foldcomp_id]
      results = [p async for p in load(inputs, foldcomp_database="afdb_rep_v4")]

      mock_get_handler.assert_called_once_with("file1.pdb", **{})
      mock_foldcomp_source_class.assert_called_once_with([foldcomp_id], "afdb_rep_v4", **{})
      mock_file_handler.process.assert_called_once()
      mock_foldcomp_handler.process.assert_called_once()
      assert len(results) == 2

  async def test_load_foldcomp_without_database(self, dummy_protein):
    """Test that FoldComp IDs are ignored if no database is provided."""
    with patch("prxteinmpnn.io.process.get_source_handler", return_value=None) as mock_get_handler, patch(
      "prxteinmpnn.io.process.FoldCompSource",
    ) as mock_foldcomp_source_class, patch("prxteinmpnn.io.process.ProcessPoolExecutor"):
      inputs = ["AF-P12345-F1-model_v4"]
      results = [p async for p in load(inputs, foldcomp_database=None)]

      mock_get_handler.assert_not_called()
      mock_foldcomp_source_class.assert_not_called()
      assert len(results) == 0

  async def test_load_empty_input_list(self):
    """Test loading from an empty list of inputs."""
    results = [p async for p in load([])]
    assert len(results) == 0

  async def test_load_invalid_input(self):
    """Test that invalid inputs that don't match any handler are ignored."""
    with patch("prxteinmpnn.io.process.get_source_handler", return_value=None) as mock_get_handler, patch(
      "prxteinmpnn.io.process.ProcessPoolExecutor",
    ):
      results = [p async for p in load(["invalid_input"])]
      mock_get_handler.assert_called_once_with("invalid_input", **{})
      assert len(results) == 0

  async def test_load_kwargs_are_passed(self, dummy_protein):
    """Test that kwargs are passed down to handlers."""
    mock_file_handler = MagicMock()
    mock_file_handler.process.return_value = async_gen_from_list([])

    mock_foldcomp_handler = MagicMock()
    mock_foldcomp_handler.process.return_value = async_gen_from_list([])

    with patch("prxteinmpnn.io.process.get_source_handler", return_value=mock_file_handler) as mock_get_handler, patch(
      "prxteinmpnn.io.process.FoldCompSource",
      return_value=mock_foldcomp_handler,
    ) as mock_foldcomp_source_class, patch("prxteinmpnn.io.process.ProcessPoolExecutor"):
      kwargs = {"model": 2, "chain_id": "A"}
      inputs = ["file.pdb", "AF-P12345-F1-model_v4"]
      _ = [p async for p in load(inputs, foldcomp_database="afdb_rep_v4", **kwargs)]

      mock_get_handler.assert_called_once_with("file.pdb", **kwargs)
      mock_foldcomp_source_class.assert_called_once_with(["AF-P12345-F1-model_v4"], "afdb_rep_v4", **kwargs)

  async def test_load_sequential_processing_of_handlers(self, dummy_protein):
    """Test that handlers are processed sequentially, one after another.

    This test verifies the current implementation detail where concurrency
    is not applied across different handlers.
    """
    order = []

    async def process1(executor):
      order.append("start1")
      await anyio.sleep(0.02)
      yield (dummy_protein, "h1")
      order.append("end1")

    async def process2(executor):
      order.append("start2")
      await anyio.sleep(0.01)
      yield (dummy_protein, "h2")
      order.append("end2")

    mock_handler1 = MagicMock()
    mock_handler1.process.side_effect = process1

    mock_handler2 = MagicMock()
    mock_handler2.process.side_effect = process2

    def side_effect(item, **kwargs):
      if item == "h1":
        return mock_handler1
      if item == "h2":
        return mock_handler2
      return None

    with patch("prxteinmpnn.io.process.get_source_handler", side_effect=side_effect), patch(
      "prxteinmpnn.io.process.ProcessPoolExecutor",
    ):
      _ = [p async for p in load(["h1", "h2"])]

    assert order == ["start1", "end1", "start2", "end2"]