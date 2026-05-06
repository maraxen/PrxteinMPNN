"""Multistate mode string constants and registry scaffold."""

import pytest

from prxteinmpnn.registry import (
  FROZEN_MULTISTATE_MODES,
  MULTISTATE_MODE_FLAT,
  MULTISTATE_MODE_STATE_VMAP,
  MULTISTATE_MODE_STATE_VMAP_EXACT,
  MULTISTATE_MODES,
  assert_known_multistate_mode,
)


def test_multistate_mode_constants_are_distinct_strings() -> None:
  modes = (
    MULTISTATE_MODE_FLAT,
    MULTISTATE_MODE_STATE_VMAP,
    MULTISTATE_MODE_STATE_VMAP_EXACT,
  )
  assert len(modes) == len(set(modes))
  assert all(isinstance(m, str) and len(m) > 0 for m in modes)


def test_frozen_multistate_modes_membership() -> None:
  assert FROZEN_MULTISTATE_MODES == frozenset(
    {
      MULTISTATE_MODE_FLAT,
      MULTISTATE_MODE_STATE_VMAP,
      MULTISTATE_MODE_STATE_VMAP_EXACT,
    },
  )
  for m in (
    MULTISTATE_MODE_FLAT,
    MULTISTATE_MODE_STATE_VMAP,
    MULTISTATE_MODE_STATE_VMAP_EXACT,
  ):
    assert m in FROZEN_MULTISTATE_MODES


def test_assert_known_multistate_mode_accepts_flat_and_state_vmap_exact() -> None:
  assert assert_known_multistate_mode(MULTISTATE_MODE_FLAT) == MULTISTATE_MODE_FLAT
  assert (
    assert_known_multistate_mode(MULTISTATE_MODE_STATE_VMAP_EXACT)
    == MULTISTATE_MODE_STATE_VMAP_EXACT
  )
  assert assert_known_multistate_mode(MULTISTATE_MODE_STATE_VMAP) == MULTISTATE_MODE_STATE_VMAP


def test_assert_known_multistate_mode_rejects_garbage() -> None:
  with pytest.raises(ValueError, match="Unknown multistate_mode"):
    assert_known_multistate_mode("not_a_supported_mode")


def test_multistate_modes_registry_empty_scaffold() -> None:
  assert MULTISTATE_MODES.keys() == []
