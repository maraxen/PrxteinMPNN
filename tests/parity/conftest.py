"""Parity suite defaults (roadmap §13 Q5)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _parity_enable_runtime_verify() -> None:
  """Turn on ``PRXTEINMPNN_VERIFY`` only for tests under ``tests/parity/``."""
  previous = os.environ.get("PRXTEINMPNN_VERIFY")
  os.environ["PRXTEINMPNN_VERIFY"] = "1"
  yield
  if previous is None:
    os.environ.pop("PRXTEINMPNN_VERIFY", None)
  else:
    os.environ["PRXTEINMPNN_VERIFY"] = previous
