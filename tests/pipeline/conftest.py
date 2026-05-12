"""Shared test fixtures for pipeline tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def registry_snapshot():
    """Snapshot pipeline_registry._REGISTRY before each test, restore after.

    Prevents cloudpickle-UID pollution from test ordering and enables
    registry isolation across tests.
    """
    import prxteinmpnn.pipeline_registry as _reg

    snap = dict(_reg._REGISTRY)
    yield
    _reg._REGISTRY.clear()
    _reg._REGISTRY.update(snap)
