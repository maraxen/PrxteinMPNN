"""Tests for the io.process module."""

import pathlib
from io import StringIO
from unittest.mock import Mock, patch
from unittest.mock import mock_open as mock_file_open

import pytest
import requests

from prxteinmpnn.io.process import (
  _fetch_md_cath,
  _fetch_pdb,
  _fetch_with_retry,
  _resolve_inputs,
  frame_iterator_from_inputs,
)


class TestFetchWithRetry:
  """Tests for the _fetch_with_retry function."""

  def test_successful_fetch_first_attempt(self) -> None:
    """Test successful fetch on the first attempt.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If the response does not match the expected mock.

    Example:
      >>> test_successful_fetch_first_attempt()

    """
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response) as mock_get:
      result = _fetch_with_retry("https://example.com/test", max_retries=3)

      assert result == mock_response
      mock_get.assert_called_once_with("https://example.com/test", timeout=60)
      mock_response.raise_for_status.assert_called_once()

  def test_successful_fetch_after_retries(self) -> None:
    """Test successful fetch after one or more retries.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If the response does not match after retries.

    Example:
      >>> test_successful_fetch_after_retries()

    """
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    # First two calls fail, third succeeds
    side_effects = [
      requests.RequestException("Connection error"),
      requests.RequestException("Timeout"),
      mock_response,
    ]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep") as mock_sleep,
    ):
      result = _fetch_with_retry(
        "https://example.com/test",
        max_retries=3,
        initial_delay=0.5,
        backoff_factor=2.0,
      )

      assert result == mock_response
      # Verify exponential backoff delays
      expected_retry_count = 2
      assert mock_sleep.call_count == expected_retry_count
      mock_sleep.assert_any_call(0.5)  # First retry delay
      mock_sleep.assert_any_call(1.0)  # Second retry delay (0.5 * 2.0)

  def test_all_retries_fail(self) -> None:
    """Test that RequestException is raised when all retries fail.

    Args:
      None

    Returns:
      None

    Raises:
      pytest.raises: If RequestException is not raised as expected.

    Example:
      >>> test_all_retries_fail()

    """
    side_effects = [
      requests.RequestException("Connection error"),
      requests.RequestException("Timeout"),
      requests.RequestException("Server error"),
    ]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
      pytest.raises(requests.RequestException, match=r"Failed to fetch.*after 3 attempts"),
    ):
      _fetch_with_retry("https://example.com/test", max_retries=3)

  def test_custom_timeout(self) -> None:
    """Test that custom timeout is passed to requests.get.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If timeout is not passed correctly.

    Example:
      >>> test_custom_timeout()

    """
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response) as mock_get:
      _fetch_with_retry("https://example.com/test", timeout=120)

      mock_get.assert_called_once_with("https://example.com/test", timeout=120)

  def test_http_error_triggers_retry(self) -> None:
    """Test that HTTP errors (e.g., 404, 500) trigger retries.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If retries are not triggered correctly.

    Example:
      >>> test_http_error_triggers_retry()

    """
    mock_response_fail = Mock()
    mock_response_fail.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

    mock_response_success = Mock()
    mock_response_success.raise_for_status = Mock()

    side_effects = [mock_response_fail, mock_response_success]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
    ):
      result = _fetch_with_retry("https://example.com/test", max_retries=2)

      assert result == mock_response_success


class TestFetchPdb:
  """Tests for the _fetch_pdb function."""

  def test_fetch_pdb_success(self) -> None:
    """Test successful PDB fetching.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If the fetched content does not match expected.

    Example:
      >>> test_fetch_pdb_success()

    """
    mock_response = Mock()
    mock_response.text = "ATOM   1  CA  ALA A   1      10.000  20.000  30.000"
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
      result = _fetch_pdb("1abc")

      assert result == mock_response.text
      assert "ATOM" in result

  def test_fetch_pdb_with_retry(self) -> None:
    """Test PDB fetching with retry logic.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If retry logic is not working correctly.

    Example:
      >>> test_fetch_pdb_with_retry()

    """
    mock_response = Mock()
    mock_response.text = "ATOM   1  CA  ALA A   1      10.000  20.000  30.000"
    mock_response.raise_for_status = Mock()

    side_effects = [requests.RequestException("Timeout"), mock_response]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
    ):
      result = _fetch_pdb("1abc")

      assert result == mock_response.text

  def test_fetch_pdb_failure(self) -> None:
    """Test PDB fetching failure after all retries.

    Args:
      None

    Returns:
      None

    Raises:
      pytest.raises: If RequestException is not raised as expected.

    Example:
      >>> test_fetch_pdb_failure()

    """
    side_effects = [
      requests.RequestException("Error 1"),
      requests.RequestException("Error 2"),
      requests.RequestException("Error 3"),
    ]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
      pytest.raises(requests.RequestException),
    ):
      _fetch_pdb("1abc")


class TestFetchMdCath:
  """Tests for the _fetch_md_cath function."""

  def test_fetch_md_cath_success(self) -> None:
    """Test successful MD-CATH fetching and file saving.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If the file is not saved correctly.

    Example:
      >>> test_fetch_md_cath_success()

    """
    mock_response = Mock()
    mock_response.content = b"HDF5 file content"
    mock_response.raise_for_status = Mock()

    m_open = mock_file_open()

    with (
      patch("requests.get", return_value=mock_response),
      patch("pathlib.Path.mkdir"),
      patch("pathlib.Path.open", m_open),
    ):
      result = _fetch_md_cath("1abc00")

      # Verify the function returns a Path and has the correct name
      assert isinstance(result, pathlib.Path)
      assert str(result).endswith("mdcath_dataset_1abc00.h5")
      # Verify that the file was written
      m_open.assert_called_once_with("wb")

  def test_fetch_md_cath_with_retry(self) -> None:
    """Test MD-CATH fetching with retry logic.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If retry logic is not working correctly.

    Example:
      >>> test_fetch_md_cath_with_retry()

    """
    mock_response = Mock()
    mock_response.content = b"HDF5 file content"
    mock_response.raise_for_status = Mock()

    side_effects = [requests.RequestException("Timeout"), mock_response]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
      patch("pathlib.Path.mkdir"),
      patch("pathlib.Path.open"),
    ):
      result = _fetch_md_cath("1abc00")

      # Verify the function returns a Path
      assert isinstance(result, pathlib.Path)

  def test_fetch_md_cath_failure(self) -> None:
    """Test MD-CATH fetching failure after all retries.

    Args:
      None

    Returns:
      None

    Raises:
      pytest.raises: If RequestException is not raised as expected.

    Example:
      >>> test_fetch_md_cath_failure()

    """
    side_effects = [
      requests.RequestException("Error 1"),
      requests.RequestException("Error 2"),
      requests.RequestException("Error 3"),
    ]

    with (
      patch("requests.get", side_effect=side_effects),
      patch("time.sleep"),
      pytest.raises(requests.RequestException),
    ):
      _fetch_md_cath("1abc00")


class TestResolveInputs:
  """Tests for the _resolve_inputs function."""

  def test_resolve_pdb_id(self) -> None:
    """Test resolving a PDB ID input.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If PDB ID is not resolved correctly.

    Example:
      >>> test_resolve_pdb_id()

    """
    mock_response = Mock()
    mock_response.text = "ATOM   1  CA  ALA A   1"
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
      result = list(_resolve_inputs(["1abc"]))

      assert len(result) == 1
      assert isinstance(result[0], StringIO)

  def test_resolve_file_path(self, tmp_path: pathlib.Path) -> None:
    """Test resolving a file path input.

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If file path is not resolved correctly.

    Example:
      >>> test_resolve_file_path()

    """
    test_file = tmp_path / "test.pdb"
    test_file.write_text("ATOM   1  CA  ALA A   1")

    result = list(_resolve_inputs([str(test_file)]))

    assert len(result) == 1
    assert result[0] == test_file

  def test_resolve_stringio(self) -> None:
    """Test resolving a StringIO input.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If StringIO is not resolved correctly.

    Example:
      >>> test_resolve_stringio()

    """
    string_io = StringIO("ATOM   1  CA  ALA A   1")

    result = list(_resolve_inputs([string_io]))

    assert len(result) == 1
    assert result[0] is string_io

  def test_resolve_directory(self, tmp_path: pathlib.Path) -> None:
    """Test resolving a directory input (should yield all files in directory).

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If directory resolution is not correct.

    Example:
      >>> test_resolve_directory()

    """
    file1 = tmp_path / "test1.pdb"
    file2 = tmp_path / "test2.pdb"
    file1.write_text("ATOM 1")
    file2.write_text("ATOM 2")

    result = list(_resolve_inputs([str(tmp_path)]))

    expected_file_count = 2
    assert len(result) == expected_file_count
    assert all(isinstance(r, pathlib.Path) for r in result)


class TestFrameIteratorFromInputs:
  """Tests for the frame_iterator_from_inputs function."""

  def test_frame_iterator_basic(self, tmp_path: pathlib.Path) -> None:
    """Test basic frame iterator functionality.

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If frame iteration does not work correctly.

    Example:
      >>> test_frame_iterator_basic()

    """
    # This is a placeholder test - actual implementation would depend on
    # parse_input behavior and ProteinTuple structure
    test_file = tmp_path / "test.pdb"
    test_file.write_text("ATOM   1  CA  ALA A   1      10.000  20.000  30.000")

    with patch("prxteinmpnn.io.process.parse_input") as mock_parse:
      mock_parse.return_value = iter([])  # Mock empty iterator

      result = list(frame_iterator_from_inputs([str(test_file)]))

      # Verify parse_input was called
      mock_parse.assert_called()
      assert isinstance(result, list)
