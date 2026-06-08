"""Parity suite defaults (roadmap §13 Q5)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _parity_enable_runtime_verify() -> None:
  """Turn on ``AMINX_VERIFY`` only for tests under ``tests/parity/``."""
  previous = os.environ.get("AMINX_VERIFY")
  os.environ["AMINX_VERIFY"] = "1"
  yield
  if previous is None:
    os.environ.pop("AMINX_VERIFY", None)
  else:
    os.environ["AMINX_VERIFY"] = previous
