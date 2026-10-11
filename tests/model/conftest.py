"""Model-suite guards."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.matmul_precision import matmul_precision_guard


@pytest.fixture(autouse=True)
def _matmul_precision_unchanged() -> Iterator[None]:
  """Debt #2428: fail any model test that leaves jax_default_matmul_precision changed."""
  yield from matmul_precision_guard()
