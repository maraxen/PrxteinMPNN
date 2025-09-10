import pathlib
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponseError

from prxteinmpnn.io.input_sources import (
  DirectorySource,
  FilePathSource,
  FoldCompSource,
  PDBIDSource,
  StringIOSource,
  get_source_handler,
)
from prxteinmpnn.utils.data_structures import Protein


async def _collect(async_gen):
  """Collect all items from an async generator into a list."""
  items = []
  async for it in async_gen:
    items.append(it)
  return items


# --- Fixtures ---


@pytest.fixture
def mock_executor():
  """Fixture for a mock ProcessPoolExecutor."""
  return ProcessPoolExecutor()


@pytest.fixture
def mock_protein():
  """Fixture for a mock protein object."""
  return MagicMock(spec=Protein)


@pytest.fixture
def sample_pdb_string() -> str:
  """A simple PDB string for testing."""
  return """
ATOM      1  N   ALA A   1      27.230  36.324  24.562  1.00  0.00           N
ATOM      2  CA  ALA A   1      28.150  35.200  24.340  1.00  0.00           C
"""


@pytest.fixture
def sample_cif_string() -> str:
  """A simple CIF string for testing."""
  return """
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N N ALA A 1 1 ? 27.230 36.324 24.562
ATOM 2 C CA ALA A 1 1 ? 28.150 35.200 24.340
"""


# --- FilePathSource Tests ---


@pytest.mark.anyio
async def test_filepathsource_process_success(mock_executor, sample_pdb_string, tmp_path):
  """Test FilePathSource successfully processes a file."""
  pdb_file = tmp_path / "test.pdb"
  pdb_file.write_text(sample_pdb_string)

  source = FilePathSource(str(pdb_file))
  results = await _collect(source.process(mock_executor))

  assert len(results) == 1
  protein, path = results[0]
  assert isinstance(protein, Protein)
  assert path == str(pdb_file)


@pytest.mark.anyio
async def test_filepathsource_process_failure(mock_executor, tmp_path):
  """Test FilePathSource handles exceptions during processing."""
  bad_file = tmp_path / "bad.pdb"
  bad_file.touch()  # Create an empty file that will fail parsing

  source = FilePathSource(str(bad_file))
  with pytest.warns(UserWarning, match=f"Failed to process file '{bad_file}'"):
    results = await _collect(source.process(mock_executor))

  assert results == []


# --- DirectorySource Tests ---


@pytest.mark.anyio
async def test_directorysource_process(mock_executor, sample_pdb_string, sample_cif_string, tmp_path):
  """Test DirectorySource recursively finds and processes files."""
  d = tmp_path / "sub"
  d.mkdir()
  p1 = tmp_path / "test1.pdb"
  p1.write_text(sample_pdb_string)
  p2 = d / "test2.cif"
  p2.write_text(sample_cif_string)
  ignored_file = tmp_path / "ignored.txt"
  ignored_file.touch()

  source = DirectorySource(tmp_path)
  results = await _collect(source.process(mock_executor))

  assert len(results) == 2
  yielded_paths = {item[1] for item in results}
  assert yielded_paths == {str(p1), str(p2)}


# --- PDBIDSource Tests ---


@patch("prxteinmpnn.io.input_sources.parse_input")
@pytest.mark.anyio
async def test_pdbidsource_process_success(mock_parse_input, mock_executor, mock_protein, monkeypatch):
  """Test PDBIDSource successfully fetches and processes a PDB ID."""
  pdb_id = "1ABC"
  pdb_content = "ATOM..."

  mock_response = AsyncMock()
  mock_response.raise_for_status = AsyncMock()  # Must be async
  mock_response.text = AsyncMock(return_value=pdb_content)

  # This is the context manager returned by session.get()
  mock_get_context_manager = AsyncMock()
  mock_get_context_manager.__aenter__.return_value = mock_response

  mock_session = AsyncMock()
  mock_session.get = MagicMock(return_value=mock_get_context_manager)

  # The ClientSession itself is an async context manager
  mock_client_session_context_manager = AsyncMock()
  mock_client_session_context_manager.__aenter__.return_value = mock_session

  monkeypatch.setattr(
    "aiohttp.ClientSession", MagicMock(return_value=mock_client_session_context_manager)
  )

  mock_parse_input.return_value = [mock_protein]

  source = PDBIDSource(pdb_id)
  results = await _collect(source.process(mock_executor))

  assert results == [(mock_protein, pdb_id)]
  mock_session.get.assert_called_once_with(f"https://files.rcsb.org/download/{pdb_id}.pdb")
  mock_parse_input.assert_called_once()
  string_io_arg = mock_parse_input.call_args[0][0]
  assert isinstance(string_io_arg, StringIO)
  assert string_io_arg.getvalue() == pdb_content


@pytest.mark.anyio
async def test_pdbidsource_process_fetch_failure(mock_executor, monkeypatch):
  """Test PDBIDSource handles HTTP fetch failures."""
  pdb_id = "1XYZ"

  mock_response = AsyncMock()
  mock_response.raise_for_status.side_effect = ClientResponseError(MagicMock(), MagicMock(), status=404)

  # This is the context manager returned by session.get()
  mock_get_context_manager = AsyncMock()
  mock_get_context_manager.__aenter__.return_value = mock_response

  mock_session = AsyncMock()
  mock_session.get = MagicMock(return_value=mock_get_context_manager)

  # The ClientSession itself is an async context manager
  mock_client_session_context_manager = AsyncMock()
  mock_client_session_context_manager.__aenter__.return_value = mock_session

  monkeypatch.setattr(
    "aiohttp.ClientSession", MagicMock(return_value=mock_client_session_context_manager)
  )
  source = PDBIDSource(pdb_id)
  with pytest.warns(UserWarning, match=f"Failed to fetch or process PDB ID '{pdb_id}'"):
    results = await _collect(source.process(mock_executor))

  assert results == []


# --- StringIOSource Tests ---


@patch("prxteinmpnn.io.input_sources.parse_input")
@pytest.mark.anyio
async def test_stringiosource_process_success(mock_parse_input, mock_executor, mock_protein):
  """Test StringIOSource successfully processes a StringIO object."""
  string_io = StringIO("ATOM...")
  mock_parse_input.return_value = [mock_protein]

  source = StringIOSource(string_io)
  results = await _collect(source.process(mock_executor))

  assert results == [(mock_protein, "StringIO")]
  mock_parse_input.assert_called_once_with(string_io, **{})


@patch("prxteinmpnn.io.input_sources.parse_input")
@pytest.mark.anyio
async def test_stringiosource_process_failure(mock_parse_input, mock_executor):
  """Test StringIOSource handles exceptions during processing."""
  string_io = StringIO("ATOM...")
  mock_parse_input.side_effect = ValueError("Parsing failed")

  source = StringIOSource(string_io)
  with pytest.warns(UserWarning, match="Failed to process StringIO input: Parsing failed"):
    results = await _collect(source.process(mock_executor))

  assert results == []


# --- FoldCompSource Tests ---


@pytest.mark.anyio
async def test_foldcomp_process_no_db_warns():
  """Test that process warns and yields nothing when no FoldComp database is provided."""
  source = FoldCompSource(["AF-ABCDEF-1-model_v1"], foldcomp_database=None)
  with pytest.warns(UserWarning, match="FoldComp IDs provided but no database specified."):
    results = await _collect(source.process(None))
  assert results == []


@patch("prxteinmpnn.io.input_sources.get_protein_structures")
@pytest.mark.anyio
async def test_foldcomp_process_success_yields(mock_get_structures, mock_executor):
  """Test that process yields protein objects paired with the provided IDs on success."""
  fake_proteins = [MagicMock(spec=Protein), MagicMock(spec=Protein)]
  ids = ["AF-ABCDE1-1-model_v1"]
  db = "fake_db"

  with patch.object(mock_executor, "submit") as mock_submit:
    future = MagicMock()
    future.result.return_value = fake_proteins
    mock_submit.return_value = future

    source = FoldCompSource(ids, foldcomp_database=db)
    results = await _collect(source.process(mock_executor))

    expected = [(fake_proteins[0], ids), (fake_proteins[1], ids)]
    assert results == expected
    mock_submit.assert_called_once_with(mock_get_structures, ids, db)


@patch("prxteinmpnn.io.input_sources.get_protein_structures")
@pytest.mark.anyio
async def test_foldcomp_process_exception_warns(mock_get_structures, mock_executor):
  """Test that exceptions from the foldcomp fetch are caught and produce a warning."""
  ids = ["AF-ERR01-1-model_v1"]
  db = "db"

  with patch.object(mock_executor, "submit") as mock_submit:
    future = MagicMock()
    future.result.side_effect = RuntimeError("simulated failure")
    mock_submit.return_value = future

    source = FoldCompSource(ids, foldcomp_database=db)
    with pytest.warns(UserWarning, match="Failed to process FoldComp IDs:"):
      results = await _collect(source.process(mock_executor))
    assert results == []
    mock_submit.assert_called_once_with(mock_get_structures, ids, db)


# --- get_source_handler Tests ---


def test_get_source_handler_stringio():
  """Test get_source_handler for StringIO."""
  item = StringIO()
  handler = get_source_handler(item)
  assert isinstance(handler, StringIOSource)
  assert handler.value is item


def test_get_source_handler_foldcomp():
  """Test get_source_handler for FoldComp ID."""
  item = "AF-P12345-F1-model_v4"
  handler = get_source_handler(item)
  assert handler is None


def test_get_source_handler_pdb_id():
  """Test get_source_handler for a PDB ID."""
  item = "1ABC"
  handler = get_source_handler(item)
  assert isinstance(handler, PDBIDSource)
  assert handler.value == item


def test_get_source_handler_pdb_id_as_file(tmp_path):
  """Test get_source_handler for a string that looks like a PDB ID but is a file."""
  p = tmp_path / "1ABC"
  p.touch()
  handler = get_source_handler(str(p))
  assert isinstance(handler, FilePathSource)
  assert handler.value == str(p)


def test_get_source_handler_directory(tmp_path):
  """Test get_source_handler for a directory."""
  handler = get_source_handler(str(tmp_path))
  assert isinstance(handler, DirectorySource)
  assert handler.value == tmp_path


def test_get_source_handler_uncategorized():
  """Test get_source_handler for an uncategorized input."""
  item = "this-is-not-a-valid-input"
  with pytest.warns(UserWarning, match=f"Input '{item}' could not be categorized and will be ignored."):
    handler = get_source_handler(item)
  assert handler is None
