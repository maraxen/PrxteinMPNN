"""Tests for runner-side unresolved-URI guard in _loader_inputs (T7b, G4b)."""

from __future__ import annotations

import pytest

from aminx.io.proxide_fetch import InputResolutionError
from aminx.run.specs import _loader_inputs


class TestLoaderInputsGuard:
    """Guard tests: _loader_inputs raises InputResolutionError on unresolved URIs.

    Gate G4b: Any spec.inputs token carrying a :// scheme that was not resolved
    to a local path raises InputResolutionError, preventing offline clusters from
    attempting fetches at runtime.
    """

    def test_unresolved_pdb_uri_raises(self) -> None:
        """Unresolved pdb:// URI raises InputResolutionError with token name."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(["pdb://1A3A"])

        error_msg = str(exc_info.value)
        assert "pdb://1A3A" in error_msg, f"Error should name the token, got: {error_msg}"
        assert "local paths" in error_msg.lower(), f"Error should mention local paths, got: {error_msg}"

    def test_unresolved_afdb_uri_raises(self) -> None:
        """Unresolved afdb:// URI raises InputResolutionError."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(["afdb://P12345"])

        error_msg = str(exc_info.value)
        assert "afdb://P12345" in error_msg
        assert "local paths" in error_msg.lower()

    def test_unresolved_mdcath_uri_raises(self) -> None:
        """Unresolved mdcath:// URI raises InputResolutionError."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(["mdcath://1abcA00"])

        error_msg = str(exc_info.value)
        assert "mdcath://1abcA00" in error_msg
        assert "local paths" in error_msg.lower()

    def test_resolved_local_path_no_raise(self) -> None:
        """Resolved local path does not raise."""
        result = _loader_inputs(["/tmp/test.pdb"])
        assert result == ["/tmp/test.pdb"]

    def test_resolved_local_path_relative_no_raise(self) -> None:
        """Resolved relative local path does not raise."""
        result = _loader_inputs(["./test.pdb"])
        assert result == ["./test.pdb"]

    def test_file_uri_scheme_raises(self) -> None:
        """Even file:// scheme raises (not resolved)."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(["file:///tmp/test.pdb"])

        error_msg = str(exc_info.value)
        assert "file://" in error_msg or "/tmp/test.pdb" in error_msg

    def test_mixed_resolved_and_unresolved_raises_on_first(self) -> None:
        """Mixed list with resolved + unresolved raises on first unresolved."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(["/tmp/test.pdb", "pdb://1A3A", "./other.pdb"])

        error_msg = str(exc_info.value)
        assert "pdb://1A3A" in error_msg

    def test_string_input_no_scheme_no_raise(self) -> None:
        """String input (which is a Sequence) without scheme passes through."""
        result = _loader_inputs("/tmp/test.pdb")
        # strings are Sequences in Python, so returned unchanged
        assert result == "/tmp/test.pdb"

    def test_string_input_with_unresolved_uri_raises(self) -> None:
        """String input (which is a Sequence) with unresolved URI raises."""
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs("pdb://1A3A")

        error_msg = str(exc_info.value)
        assert "pdb://1A3A" in error_msg
