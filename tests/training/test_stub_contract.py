"""Tests for aminx.training's not-yet-ready module stub (PEP 562 __getattr__)."""

from __future__ import annotations

import pytest

import aminx.training


def test_attribute_access_raises_attribute_error() -> None:
  with pytest.raises(AttributeError, match="aminx.training is not yet updated"):
    aminx.training.anything


def test_getattr_with_default_does_not_raise() -> None:
  # A module __getattr__ (PEP 562) must raise AttributeError, not some other
  # exception, so that getattr(..., default) and hasattr() degrade gracefully
  # instead of propagating. This is what pickle.whichmodule relies on when it
  # scans sys.modules -- see src/aminx/training/__init__.py's __getattr__ docstring.
  sentinel = object()
  assert getattr(aminx.training, "anything", sentinel) is sentinel


def test_hasattr_returns_false() -> None:
  assert hasattr(aminx.training, "anything") is False
