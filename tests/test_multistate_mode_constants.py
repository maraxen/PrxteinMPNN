"""Multistate mode string constants and ``MULTISTATE_MODES`` registry."""

import pytest

from prxteinmpnn.registry import (
  FROZEN_MULTISTATE_MODES,
  MULTISTATE_MODE_FLAT,
  MULTISTATE_MODE_STATE_VMAP,
  MULTISTATE_MODE_STATE_VMAP_EXACT,
  MULTISTATE_MODES,
  MultistateModeDescriptor,
  assert_known_multistate_mode,
  multistate_mode_descriptor,
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


def test_multistate_modes_registry_default_keys() -> None:
  assert MULTISTATE_MODES.keys() == [
    MULTISTATE_MODE_FLAT,
    MULTISTATE_MODE_STATE_VMAP,
    MULTISTATE_MODE_STATE_VMAP_EXACT,
  ]


def test_multistate_mode_descriptor_routing_flags() -> None:
  flat = multistate_mode_descriptor(MULTISTATE_MODE_FLAT)
  assert flat == MultistateModeDescriptor(
    uses_stacked_exact_model_call=False,
    uses_stacked_exact_sample_wave=False,
    uses_stacked_exact_score_factory=False,
    allows_ligand_flat_encoder_path=True,
  )
  sv = multistate_mode_descriptor(MULTISTATE_MODE_STATE_VMAP)
  assert sv.allows_ligand_flat_encoder_path is False
  assert sv.uses_stacked_exact_model_call is False
  exact = multistate_mode_descriptor(MULTISTATE_MODE_STATE_VMAP_EXACT)
  assert exact.uses_stacked_exact_model_call
  assert exact.uses_stacked_exact_sample_wave
  assert exact.uses_stacked_exact_score_factory
  assert exact.allows_ligand_flat_encoder_path is False
