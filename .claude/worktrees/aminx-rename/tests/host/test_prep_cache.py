"""Tests for JAX compilation cache configuration in prep.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import pytest

from aminx.host.prep import prep_protein_stream_and_model
from aminx.run.specs import SamplingSpecification


class TestPrepCacheConfig:
  """Test suite for JAX compilation cache configuration."""

  def test_cache_path_set_updates_jax_config(self, tmp_path: Path) -> None:
    """When cache_path is set on spec, jax_compilation_cache_dir should be updated."""
    cache_dir = tmp_path / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=cache_dir,
    )

    with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
         patch("aminx.host.prep.load_model") as mock_load_model:
      mock_dataset.return_value = MagicMock()
      mock_load_model.return_value = MagicMock()

      try:
        prep_protein_stream_and_model(spec)
      except Exception:
        # We expect exceptions due to mocking, but cache config should have been set
        pass

      # Verify that jax.config was updated with the cache path
      cache_config = jax.config.jax_compilation_cache_dir
      assert str(cache_dir) == cache_config

  def test_cache_path_none_does_not_update_jax_config(self) -> None:
    """When cache_path is None, jax_compilation_cache_dir should not be modified."""
    # Store original config value
    original_cache_dir = jax.config.jax_compilation_cache_dir

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=None,
    )

    with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
         patch("aminx.host.prep.load_model") as mock_load_model:
      mock_dataset.return_value = MagicMock()
      mock_load_model.return_value = MagicMock()

      try:
        prep_protein_stream_and_model(spec)
      except Exception:
        # Expected due to mocking
        pass

      # Verify that jax.config was not changed
      cache_config = jax.config.jax_compilation_cache_dir
      assert cache_config == original_cache_dir

  def test_cache_path_converts_string_to_path(self, tmp_path: Path) -> None:
    """When cache_path is a string, it should be converted to Path and used."""
    cache_dir = tmp_path / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=str(cache_dir),
    )

    # Verify that cache_path was converted to Path in __post_init__
    assert isinstance(spec.cache_path, Path)

    with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
         patch("aminx.host.prep.load_model") as mock_load_model:
      mock_dataset.return_value = MagicMock()
      mock_load_model.return_value = MagicMock()

      try:
        prep_protein_stream_and_model(spec)
      except Exception:
        pass

      # Verify jax.config was set with the Path converted to string
      cache_config = jax.config.jax_compilation_cache_dir
      assert cache_config == str(cache_dir)
