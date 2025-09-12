import pathlib
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
async def test_filepathsource_process_success(mock_protein, sample_pdb_string, tmp_path):
  """Test FilePathSource successfully processes a file."""
  pdb_file = tmp_path / "test.pdb"
  pdb_file.write_text(sample_pdb_string)

  source = FilePathSource(str(pdb_file))

  async def mock_async_gen():
    yield (mock_protein, str(pdb_file))

  with patch(
    "prxteinmpnn.io.input_sources._parse_input_worker",
    return_value=mock_async_gen(),
  ):
    results = await _collect(source.process())

  assert len(results) == 1
  protein, path = results[0]
  assert protein is mock_protein
  assert path == str(pdb_file)


@pytest.mark.anyio
async def test_filepathsource_process_failure(tmp_path):
  """Test FilePathSource handles exceptions during processing."""
  bad_file = tmp_path / "bad.pdb"
  bad_file.touch()  # Create an empty file that will fail parsing

  source = FilePathSource(str(bad_file))
  with pytest.warns(UserWarning, match=f"Failed to process file '{bad_file}'"):
    with patch(
      "prxteinmpnn.io.input_sources._parse_input_worker",
      new_callable=AsyncMock,
    ) as mock_worker:
      mock_worker.side_effect = Exception("parsing failed")
      results = await _collect(source.process())

  assert results == []


# --- DirectorySource Tests ---


@pytest.mark.anyio
async def test_directorysource_process(
  mock_protein, sample_pdb_string, sample_cif_string, tmp_path
):
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

  async def mock_gen_p1():
    yield (mock_protein, str(p1))

  async def mock_gen_p2():
    yield (mock_protein, str(p2))

  with patch(
    "prxteinmpnn.io.input_sources._parse_input_worker",
    side_effect=[mock_gen_p1(), mock_gen_p2()],
  ):
    results = await _collect(source.process())

  assert len(results) == 2
  yielded_paths = {item[1] for item in results}
  assert yielded_paths == {str(p1), str(p2)}


# --- PDBIDSource Tests ---


@pytest.mark.anyio
async def test_pdbidsource_process_success(mock_protein, monkeypatch):
  """Test PDBIDSource successfully fetches and processes a PDB ID."""
  pdb_id = "1ABC"
  pdb_content = "ATOM..."

  mock_response = AsyncMock()
  mock_response.raise_for_status = AsyncMock()
  mock_response.text = AsyncMock(return_value=pdb_content)

  mock_get_context_manager = AsyncMock()
  mock_get_context_manager.__aenter__.return_value = mock_response

  mock_session = AsyncMock()
  mock_session.get = MagicMock(return_value=mock_get_context_manager)

  mock_client_session_context_manager = AsyncMock()
  mock_client_session_context_manager.__aenter__.return_value = mock_session

  monkeypatch.setattr(
    "aiohttp.ClientSession", MagicMock(return_value=mock_client_session_context_manager)
  )

  source = PDBIDSource(pdb_id)

  async def mock_async_gen():
    yield (mock_protein, pdb_id)

  with patch(
    "prxteinmpnn.io.input_sources._parse_input_worker",
    return_value=mock_async_gen(),
  ) as mock_worker:
    results = await _collect(source.process())

  assert results == [(mock_protein, pdb_id)]
  mock_session.get.assert_called_once_with(f"https://files.rcsb.org/download/{pdb_id}.pdb")
  mock_worker.assert_called_once()
  string_io_arg = mock_worker.call_args[0][0]
  assert isinstance(string_io_arg, StringIO)
  assert string_io_arg.getvalue() == pdb_content


@pytest.mark.anyio
async def test_pdbidsource_process_fetch_failure(monkeypatch):
  """Test PDBIDSource handles HTTP fetch failures."""
  pdb_id = "1XYZ"

  mock_response = AsyncMock()
  mock_response.raise_for_status.side_effect = ClientResponseError(
    MagicMock(), MagicMock(), status=404
  )

  mock_get_context_manager = AsyncMock()
  mock_get_context_manager.__aenter__.return_value = mock_response

  mock_session = AsyncMock()
  mock_session.get = MagicMock(return_value=mock_get_context_manager)

  mock_client_session_context_manager = AsyncMock()
  mock_client_session_context_manager.__aenter__.return_value = mock_session

  monkeypatch.setattr(
    "aiohttp.ClientSession", MagicMock(return_value=mock_client_session_context_manager)
  )
  source = PDBIDSource(pdb_id)
  with pytest.warns(UserWarning, match=f"Failed to fetch or process PDB ID '{pdb_id}'"):
    results = await _collect(source.process())

  assert results == []


# --- StringIOSource Tests ---


@pytest.mark.anyio
async def test_stringiosource_process_success(mock_protein):
  """Test StringIOSource successfully processes a StringIO object."""
  string_io = StringIO("ATOM...")

  source = StringIOSource(string_io)

  async def mock_async_gen():
    yield (mock_protein, "StringIO")

  with patch(
    "prxteinmpnn.io.input_sources._parse_input_worker",
    return_value=mock_async_gen(),
  ) as mock_worker:
    results = await _collect(source.process())

  assert results == [(mock_protein, "StringIO")]
  mock_worker.assert_called_once_with(string_io, **{})


@pytest.mark.anyio
async def test_stringiosource_process_failure():
  """Test StringIOSource handles exceptions during processing."""
  string_io = StringIO("ATOM...")

  source = StringIOSource(string_io)
  with pytest.warns(UserWarning, match="Failed to process StringIO input: Parsing failed"):
    with patch(
      "prxteinmpnn.io.input_sources._parse_input_worker",
      new_callable=AsyncMock,
    ) as mock_worker:
      mock_worker.side_effect = ValueError("Parsing failed")
      results = await _collect(source.process())

  assert results == []


# --- FoldCompSource Tests ---


@pytest.mark.anyio
async def test_foldcomp_process_no_db_warns():
  """Test that process warns and yields nothing when no FoldComp database is provided."""
  source = FoldCompSource(["AF-ABCDEF-1-model_v1"], foldcomp_database=None)  # type: ignore
  with pytest.warns(UserWarning, match="FoldComp IDs provided but no database specified."):
    results = await _collect(source.process())
  assert results == []


@pytest.mark.anyio
async def test_foldcomp_process_success_yields():
  """Test that process yields protein objects paired with the provided IDs on success."""
  fake_proteins = [MagicMock(spec=Protein), MagicMock(spec=Protein)]
  ids = ["AF-ABCDE1-1-model_v1"]
  db = "esmatlas"

  source = FoldCompSource(ids, foldcomp_database=db)

  async def mock_async_gen():
    yield (fake_proteins[0], ids[0])
    yield (fake_proteins[1], ids[0])

  with patch(
    "prxteinmpnn.io.input_sources._get_protein_structures_worker",
    return_value=mock_async_gen(),
  ) as mock_worker:
    results = await _collect(source.process())

    expected = [(fake_proteins[0], ids[0]), (fake_proteins[1], ids[0])]
    assert results == expected
    mock_worker.assert_called_once_with(ids, db)


@pytest.mark.anyio
async def test_foldcomp_process_exception_warns():
  """Test that exceptions from the foldcomp fetch are caught and produce a warning."""
  ids = ["AF-ERR01-1-model_v1"]
  db = "esmatlas"

  source = FoldCompSource(ids, foldcomp_database=db)
  with pytest.warns(UserWarning, match="Failed to process FoldComp IDs:"):
    with patch(
      "prxteinmpnn.io.input_sources._get_protein_structures_worker",
      new_callable=AsyncMock,
    ) as mock_worker:
      mock_worker.side_effect = RuntimeError("simulated failure")
      results = await _collect(source.process())
  assert results == []


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
