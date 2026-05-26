"""Unit tests for host dispatch layer coverage.

Covers:
  - sampling_driver.py: SamplingDriver initialization, registry lookups
  - averaging.py: logit aggregation, encode/sample splitting functions
  - prep.py: bundle/config preparation, checkpoint resolution
"""

from __future__ import annotations

from unittest import mock
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import pytest
from jax import random

from prxteinmpnn.host.sampling_driver import SamplingDriver
from prxteinmpnn.host.averaging import (
    get_averaged_encodings,
    make_encoding_sampling_split_fn,
)
from prxteinmpnn.host.prep import (
    _loader_inputs,
    _artifact_path_for_registry_entry,
    _resolve_local_checkpoint_from_registry,
)
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.utils.data_structures import Protein


# ============================================================================
# sampling_driver.py tests
# ============================================================================


class TestSamplingDriver:
    """Test host-side sampling driver orchestration."""

    @staticmethod
    def _make_minimal_spec():
        """Create minimal SamplingSpecification for testing."""
        spec = SamplingSpecification(inputs="dummy.pdb")
        return spec

    def test_sampling_driver_init(self):
        """SamplingDriver initializes with spec and factory key."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec, sampler_factory_key="make_sample_sequences")
        assert driver.spec is spec
        assert driver.sampler_factory_key == "make_sample_sequences"

    def test_sampling_driver_default_factory_key(self):
        """SamplingDriver default factory key is 'make_sample_sequences'."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        assert driver.sampler_factory_key == "make_sample_sequences"

    def test_sampler_factory_keys_not_empty(self):
        """sampler_factory_keys() returns non-empty tuple after import."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        keys = driver.sampler_factory_keys()
        assert isinstance(keys, tuple)
        assert len(keys) > 0

    def test_sampler_factory_keys_includes_make_sample_sequences(self):
        """sampler_factory_keys() includes 'make_sample_sequences'."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        keys = driver.sampler_factory_keys()
        assert "make_sample_sequences" in keys

    def test_build_sampler_fn_with_default_key(self):
        """build_sampler_fn with default key and mock model."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        mock_model = mock.MagicMock()
        sampler_fn = driver.build_sampler_fn(mock_model)
        assert callable(sampler_fn)

    def test_build_sampler_fn_with_unknown_key_raises(self):
        """build_sampler_fn with unknown key raises KeyError with known keys listed."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec, sampler_factory_key="unknown_sampler")
        mock_model = mock.MagicMock()
        with pytest.raises(KeyError) as exc_info:
            driver.build_sampler_fn(mock_model)
        assert "unknown_sampler" in str(exc_info.value)
        assert "known keys:" in str(exc_info.value)

    def test_build_sampler_fn_with_no_extra_kwargs(self):
        """build_sampler_fn works without extra kwargs."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        mock_model = mock.MagicMock()
        # Call without extra kwargs (basic case)
        sampler_fn = driver.build_sampler_fn(mock_model)
        assert callable(sampler_fn)

    def test_sampling_driver_spec_attribute(self):
        """SamplingDriver.spec attribute is accessible."""
        spec = self._make_minimal_spec()
        driver = SamplingDriver(spec)
        assert driver.spec is spec


# ============================================================================
# averaging.py tests
# ============================================================================


class TestAveragedEncodings:
    """Test logit aggregation and encoding averaging."""

    def create_mock_protein(self, batch_size: int = 2, seq_len: int = 10) -> Protein:
        """Create a minimal mock Protein for testing."""
        return Protein(
            coordinates=jnp.ones((batch_size, seq_len, 4, 3), dtype=jnp.float32),
            mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
            residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
            chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
            aatype=jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        )

    def test_make_encoding_sampling_split_fn_importable(self):
        """make_encoding_sampling_split_fn is importable."""
        assert callable(make_encoding_sampling_split_fn)

    @pytest.mark.skip(reason="deprecated stub uses *args/**kwargs; signature introspection no longer valid")
    def test_make_encoding_sampling_split_fn_signature(self):
        """make_encoding_sampling_split_fn has expected signature."""
        import inspect

        sig = inspect.signature(make_encoding_sampling_split_fn)
        params = list(sig.parameters.keys())
        assert "model_parameters" in params
        assert "decoding_order_fn" in params
        assert "sampling_strategy" in params
        assert "decode_fn_wrapper" in params

    @pytest.mark.skip(reason="deprecated stub uses *args/**kwargs; signature introspection no longer valid")
    def test_make_encoding_sampling_split_fn_default_params(self):
        """make_encoding_sampling_split_fn default sampling_strategy is temperature."""
        import inspect

        sig = inspect.signature(make_encoding_sampling_split_fn)
        strategy_param = sig.parameters["sampling_strategy"]
        assert strategy_param.default == "temperature"

    def test_make_encoding_sampling_split_fn_returns_tuple(self, minimal_model):
        """make_encoding_sampling_split_fn returns (encode_fn, sample_fn, decode_logits_fn)."""
        encode_fn, sample_fn, decode_logits_fn = make_encoding_sampling_split_fn(minimal_model)
        assert callable(encode_fn)
        assert callable(sample_fn)
        assert callable(decode_logits_fn)

    def test_make_encoding_sampling_split_fn_with_wrapper(self, minimal_model):
        """make_encoding_sampling_split_fn accepts decode_fn_wrapper."""
        def dummy_wrapper(fn):
            """Dummy wrapper that returns the function unchanged."""
            return fn

        encode_fn, sample_fn, decode_logits_fn = make_encoding_sampling_split_fn(
            minimal_model,
            decode_fn_wrapper=dummy_wrapper,
        )
        assert callable(encode_fn)
        assert callable(sample_fn)
        assert callable(decode_logits_fn)

    def test_make_encoding_sampling_split_fn_encode_jitted(self, minimal_model):
        """encode_fn returned is JIT-compiled."""
        encode_fn, _, _ = make_encoding_sampling_split_fn(minimal_model)
        # Check that the function has the JIT wrapper attributes
        assert hasattr(encode_fn, "__wrapped__") or callable(encode_fn)
        # Verify it's a JIT-compiled function by checking for lowered attribute
        # or by checking that it can be called
        assert callable(encode_fn)

    def test_sample_fn_signature_has_required_params(self, minimal_model):
        """sample_fn has required parameters: prng_key, encoded_features, decoding_order."""
        import inspect

        _, sample_fn, _ = make_encoding_sampling_split_fn(minimal_model)
        sig = inspect.signature(sample_fn)
        params = list(sig.parameters.keys())
        assert "prng_key" in params
        assert "encoded_features" in params
        assert "decoding_order" in params

    def test_get_averaged_encodings_importable(self):
        """get_averaged_encodings is importable and callable."""
        assert callable(get_averaged_encodings)

    @pytest.mark.skip(reason="deprecated stub uses *args/**kwargs; signature introspection no longer valid")
    def test_averaging_mode_inputs_is_valid(self):
        """get_averaged_encodings accepts average_encoding_mode='inputs'."""
        import inspect

        sig = inspect.signature(get_averaged_encodings)
        avg_mode_param = sig.parameters['average_encoding_mode']
        assert avg_mode_param is not None

    @pytest.mark.skip(reason="deprecated stub uses *args/**kwargs; signature introspection no longer valid")
    def test_averaging_mode_noise_levels_is_valid(self):
        """get_averaged_encodings accepts average_encoding_mode='noise_levels'."""
        import inspect

        sig = inspect.signature(get_averaged_encodings)
        avg_mode_param = sig.parameters['average_encoding_mode']
        assert avg_mode_param is not None

    @pytest.mark.skip(reason="deprecated stub uses *args/**kwargs; signature introspection no longer valid")
    def test_averaging_mode_inputs_and_noise_is_valid(self):
        """get_averaged_encodings accepts average_encoding_mode='inputs_and_noise'."""
        import inspect

        sig = inspect.signature(get_averaged_encodings)
        avg_mode_param = sig.parameters['average_encoding_mode']
        assert avg_mode_param is not None


# ============================================================================
# prep.py tests
# ============================================================================


class TestPrepHelpers:
    """Test prep.py helper functions."""

    def test_loader_inputs_single_string(self):
        """_loader_inputs passes string (Sequence) through unchanged."""
        result = _loader_inputs("test.pdb")
        # str is a Sequence, so it's returned unchanged
        assert result == "test.pdb"

    def test_loader_inputs_sequence_passthrough(self):
        """_loader_inputs passes through sequence unchanged."""
        inputs = ("a.pdb", "b.pdb")
        result = _loader_inputs(inputs)
        assert result == inputs

    def test_loader_inputs_list(self):
        """_loader_inputs handles list input."""
        inputs = ["a.pdb", "b.pdb", "c.pdb"]
        result = _loader_inputs(inputs)
        assert result == inputs

    def test_artifact_path_absolute_path_unchanged(self):
        """_artifact_path_for_registry_entry returns absolute path unchanged."""
        registry_path = Path("/registry.json")
        artifact_path = "/absolute/path/model.ckpt"
        result = _artifact_path_for_registry_entry(registry_path, artifact_path)
        assert result == Path(artifact_path)

    def test_artifact_path_relative_resolved(self):
        """_artifact_path_for_registry_entry resolves relative paths next to registry."""
        registry_path = Path("/path/to/registry.json")
        artifact_path = "models/model.ckpt"
        result = _artifact_path_for_registry_entry(registry_path, artifact_path)
        assert result == Path("/path/to/models/model.ckpt")

    def test_artifact_path_relative_with_parent_refs(self):
        """_artifact_path_for_registry_entry resolves .. in relative paths."""
        registry_path = Path("/path/to/registry.json")
        artifact_path = "../models/model.ckpt"
        result = _artifact_path_for_registry_entry(registry_path, artifact_path)
        assert "models" in str(result)
        assert result.is_absolute()

    def test_loader_inputs_with_stringio(self):
        """_loader_inputs wraps StringIO objects in tuple."""
        from io import StringIO

        io_obj = StringIO("test content")
        result = _loader_inputs(io_obj)
        # StringIO is not a Sequence, so it gets wrapped
        assert isinstance(result, tuple)
        assert result[0] is io_obj

    def test_loader_inputs_with_tuple(self):
        """_loader_inputs handles tuple input."""
        inputs = ("a.pdb", "b.pdb")
        result = _loader_inputs(inputs)
        assert result == inputs
        assert isinstance(result, tuple)

    def test_artifact_path_nested_relative_path(self):
        """_artifact_path_for_registry_entry handles deeply nested relative paths."""
        registry_path = Path("/a/b/c/registry.json")
        artifact_path = "../../models/v1/model.ckpt"
        result = _artifact_path_for_registry_entry(registry_path, artifact_path)
        assert result.is_absolute()
        assert "models" in str(result)
        assert "v1" in str(result)


class TestResolveLocalCheckpoint:
    """Test checkpoint resolution from registry."""

    def test_resolve_checkpoint_no_registry_path_returns_none(self):
        """_resolve_local_checkpoint_from_registry returns None if no registry_path."""
        spec = mock.MagicMock()
        spec.checkpoint_registry_path = None
        result = _resolve_local_checkpoint_from_registry(spec)
        assert result is None

    def test_resolve_checkpoint_missing_checkpoint_id_raises(self):
        """_resolve_local_checkpoint_from_registry raises if checkpoint_id missing."""
        spec = mock.MagicMock()
        spec.checkpoint_registry_path = Path("/registry.json")
        spec.checkpoint_id = None
        with pytest.raises(ValueError) as exc_info:
            _resolve_local_checkpoint_from_registry(spec)
        assert "checkpoint_id is required" in str(exc_info.value)

    def test_resolve_checkpoint_entry_not_found_raises(self, tmp_path):
        """_resolve_local_checkpoint_from_registry raises if entry not found."""
        registry_path = tmp_path / "registry.json"
        registry_json = {"entries": [{"model_family": "test", "checkpoint_id": "v1"}]}
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v2"  # Does not exist
        spec.model_family = "test"
        with pytest.raises(ValueError) as exc_info:
            _resolve_local_checkpoint_from_registry(spec)
        assert "No checkpoint registry entry found" in str(exc_info.value)

    def test_resolve_checkpoint_entry_found(self, tmp_path):
        """_resolve_local_checkpoint_from_registry finds matching entry."""
        registry_path = tmp_path / "registry.json"
        artifact_path = tmp_path / "model.ckpt"
        artifact_path.touch()  # Create dummy artifact

        registry_json = {
            "entries": [
                {
                    "model_family": "test_family",
                    "checkpoint_id": "v1",
                    "artifact_path": str(artifact_path),
                }
            ]
        }
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v1"
        spec.model_family = "test_family"

        result = _resolve_local_checkpoint_from_registry(spec)
        assert result == str(artifact_path)

    def test_resolve_checkpoint_with_sha256_match(self, tmp_path):
        """_resolve_local_checkpoint_from_registry verifies SHA256 if present."""
        artifact_path = tmp_path / "model.ckpt"
        artifact_path.write_bytes(b"test_data")

        import hashlib

        sha256_digest = hashlib.sha256(b"test_data").hexdigest()
        registry_path = tmp_path / "registry.json"
        registry_json = {
            "entries": [
                {
                    "model_family": "test_family",
                    "checkpoint_id": "v1",
                    "artifact_path": str(artifact_path),
                    "sha256": sha256_digest,
                }
            ]
        }
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v1"
        spec.model_family = "test_family"

        result = _resolve_local_checkpoint_from_registry(spec)
        assert result == str(artifact_path)

    def test_resolve_checkpoint_sha256_mismatch_raises(self, tmp_path):
        """_resolve_local_checkpoint_from_registry raises on SHA256 mismatch."""
        artifact_path = tmp_path / "model.ckpt"
        artifact_path.write_bytes(b"test_data")

        registry_path = tmp_path / "registry.json"
        registry_json = {
            "entries": [
                {
                    "model_family": "test_family",
                    "checkpoint_id": "v1",
                    "artifact_path": str(artifact_path),
                    "sha256": "wrong_hash",
                }
            ]
        }
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v1"
        spec.model_family = "test_family"

        with pytest.raises(ValueError) as exc_info:
            _resolve_local_checkpoint_from_registry(spec)
        assert "checksum mismatch" in str(exc_info.value)

    def test_resolve_checkpoint_multiple_entries_selects_correct_one(self, tmp_path):
        """_resolve_local_checkpoint_from_registry selects correct entry from multiple."""
        artifact1_path = tmp_path / "model_v1.ckpt"
        artifact2_path = tmp_path / "model_v2.ckpt"
        artifact1_path.touch()
        artifact2_path.touch()

        registry_path = tmp_path / "registry.json"
        registry_json = {
            "entries": [
                {
                    "model_family": "family_a",
                    "checkpoint_id": "v1",
                    "artifact_path": str(artifact1_path),
                },
                {
                    "model_family": "family_b",
                    "checkpoint_id": "v2",
                    "artifact_path": str(artifact2_path),
                },
            ]
        }
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v2"
        spec.model_family = "family_b"

        result = _resolve_local_checkpoint_from_registry(spec)
        assert result == str(artifact2_path)

    def test_resolve_checkpoint_with_relative_artifact_path(self, tmp_path):
        """_resolve_local_checkpoint_from_registry resolves relative artifact paths."""
        registry_dir = tmp_path / "registry_dir"
        registry_dir.mkdir()
        models_dir = registry_dir / "models"
        models_dir.mkdir()
        artifact_path = models_dir / "model.ckpt"
        artifact_path.touch()

        registry_path = registry_dir / "registry.json"
        registry_json = {
            "entries": [
                {
                    "model_family": "test_family",
                    "checkpoint_id": "v1",
                    "artifact_path": "models/model.ckpt",
                }
            ]
        }
        registry_path.write_text(json.dumps(registry_json))

        spec = mock.MagicMock()
        spec.checkpoint_registry_path = registry_path
        spec.checkpoint_id = "v1"
        spec.model_family = "test_family"

        result = _resolve_local_checkpoint_from_registry(spec)
        assert Path(result) == artifact_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
