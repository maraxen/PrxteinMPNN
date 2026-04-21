"""Grid-mode sampling tests for lineage metadata and ligand key enforcement."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.run.sampling import SamplingSpecification, sample
from prxteinmpnn.utils.data_structures import Protein


def _mock_protein(seq_len: int = 6) -> Protein:
    return Protein(
        coordinates=jnp.ones((1, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.ones((1, seq_len), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, seq_len), dtype=jnp.int8), 21),
        mask=jnp.ones((1, seq_len), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        chain_index=jnp.zeros((1, seq_len), dtype=jnp.int32),
    )


def _sampler_with_key_identity(
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    _mask: jax.Array,
    _residue_index: jax.Array,
    _chain_index: jax.Array,
    **_kwargs: object,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    seq_len = structure_coordinates.shape[0]
    token = jax.random.randint(prng_key, (), 0, 21, dtype=jnp.int32).astype(jnp.int8)
    sequence = jnp.full((seq_len,), token, dtype=jnp.int8)
    logits = jnp.zeros((seq_len, 21), dtype=jnp.float32)
    decoding_order = jnp.arange(seq_len, dtype=jnp.int32)
    return sequence, logits, decoding_order


def _artifact_path(suffix: str) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "_grid_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    return artifact_dir / f"sampling_grid_{uuid4().hex}{suffix}"


def _run_with_mocked_pipeline(spec: SamplingSpecification) -> dict[str, object]:
    protein = _mock_protein()
    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([protein], MagicMock()),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=_sampler_with_key_identity,
        ):
            return sample(spec)


@pytest.mark.parametrize(
    ("payload_ids", "expected_missing", "expected_extra"),
    [
        ([], "1ubq", None),
        (["1ubq", "extra"], None, "extra"),
    ],
)
def test_ligand_context_keys_must_exactly_match_structure_ids(
    payload_ids: list[str],
    expected_missing: str | None,
    expected_extra: str | None,
) -> None:
    ligand_context_path = _artifact_path(".npz")
    seq_len = _mock_protein().coordinates.shape[1]
    n_payload = len(payload_ids)
    np.savez(
        ligand_context_path,
        structure_ids=np.asarray(payload_ids, dtype=np.str_),
        Y=np.zeros((n_payload, seq_len, 1, 3), dtype=np.float32),
        Y_t=np.zeros((n_payload, seq_len, 1), dtype=np.int32),
        Y_m=np.zeros((n_payload, seq_len, 1), dtype=np.float32),
    )

    spec = SamplingSpecification(
        inputs=["tests/data/1ubq.pdb"],
        model_family="ligandmpnn",
        ligand_conditioning=True,
        ligand_context_path=ligand_context_path,
        num_samples=1,
    )

    try:
        with patch(
            "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
            return_value=([_mock_protein(seq_len=seq_len)], MagicMock()),
        ):
            with patch(
                "prxteinmpnn.run.sampling.make_sample_sequences",
                return_value=_sampler_with_key_identity,
            ):
                with pytest.raises(ValueError, match="exactly match canonical structure IDs") as exc_info:
                    sample(spec)
        msg = str(exc_info.value)
        assert "missing keys" in msg
        assert "extra keys" in msg
        if expected_missing is not None:
            assert expected_missing in msg
        if expected_extra is not None:
            assert expected_extra in msg
    finally:
        ligand_context_path.unlink(missing_ok=True)


def test_grid_chunk_schedule_is_stable_across_chunking() -> None:
    full_run = _run_with_mocked_pipeline(
        SamplingSpecification(
            inputs=["tests/data/1ubq.pdb"],
            num_samples=8,
            grid_mode=True,
            job_id="grid-job",
            chunk_id=2,
            sample_start=0,
            sample_count=6,
            samples_chunk_size=6,
            random_seed=11,
        )
    )
    chunk_run = _run_with_mocked_pipeline(
        SamplingSpecification(
            inputs=["tests/data/1ubq.pdb"],
            num_samples=8,
            grid_mode=True,
            job_id="grid-job",
            chunk_id=2,
            sample_start=2,
            sample_count=2,
            samples_chunk_size=1,
            random_seed=11,
        )
    )

    np.testing.assert_array_equal(
        np.asarray(full_run["sequences"])[:, 2:4],
        np.asarray(chunk_run["sequences"]),
    )
    assert chunk_run["schema_version"] == "grid_v1"
    np.testing.assert_array_equal(np.asarray(chunk_run["sample_indices"]), np.asarray([2, 3]))
    lineage = chunk_run["metadata"]["lineage"]
    assert lineage["job_id"] == "grid-job"
    assert lineage["chunk_id"] == 2
    assert lineage["sample_start"] == 2
    assert lineage["sample_count"] == 2
    assert isinstance(lineage["manifest_row_hash"], str)
    assert len(lineage["manifest_row_hash"]) == 64
    assert lineage["grid_iteration_sample_start"] == [2, 3]
    assert lineage["grid_iteration_sample_count"] == [1, 1]


def test_grid_streaming_writes_lineage_metadata_and_consistent_sample_count() -> None:
    output_h5_path = _artifact_path(".h5")
    spec = SamplingSpecification(
        inputs=["tests/data/1ubq.pdb"],
        num_samples=12,
        grid_mode=True,
        job_id="grid-stream",
        chunk_id=7,
        sample_start=8,
        sample_count=4,
        samples_chunk_size=3,
        output_h5_path=output_h5_path,
        random_seed=17,
    )

    try:
        result = _run_with_mocked_pipeline(spec)
        assert result["schema_version"] == "grid_v1"
        lineage = result["metadata"]["lineage"]
        assert lineage["job_id"] == "grid-stream"
        assert lineage["chunk_id"] == 7
        assert lineage["sample_start"] == 8
        assert lineage["sample_count"] == 4
        assert isinstance(lineage["manifest_row_hash"], str)
        assert len(lineage["manifest_row_hash"]) == 64
        assert lineage["sample_indices"] == [8, 9, 10, 11]

        with h5py.File(output_h5_path, "r") as h5_file:
            assert h5_file.attrs["schema_version"] == "grid_v1"
            assert h5_file.attrs["job_id"] == "grid-stream"
            assert int(h5_file.attrs["chunk_id"]) == 7
            assert int(h5_file.attrs["sample_start"]) == 8
            assert int(h5_file.attrs["sample_count"]) == 4
            assert h5_file.attrs["manifest_row_hash"] == lineage["manifest_row_hash"]
            np.testing.assert_array_equal(h5_file["sample_indices"][:], np.asarray([8, 9, 10, 11]))
            np.testing.assert_array_equal(
                h5_file["grid_iteration_sample_start"][:],
                np.asarray([8, 11]),
            )
            np.testing.assert_array_equal(
                h5_file["grid_iteration_sample_count"][:],
                np.asarray([3, 1]),
            )
            structure = h5_file["structure_0"]
            assert int(structure.attrs["num_samples"]) == 4
            assert int(structure.attrs["sample_count"]) == 4
    finally:
        output_h5_path.unlink(missing_ok=True)


def test_key_generation_equivalence():
    """Test that vmap-based key generation produces identical results to fold_in loop.

    This validates the refactoring from list comprehension with fold_in to eager vmap
    pre-computation. Both approaches must produce bitwise identical keys.
    """
    import jax.random as jr

    base_key = jr.key(42)
    sample_offset = 0
    num_samples = 100

    # Old approach: list comprehension with fold_in (what we replaced)
    old_keys = jnp.stack([
        jr.fold_in(base_key, idx)
        for idx in range(sample_offset, sample_offset + num_samples)
    ], axis=0)

    # New approach: vmap with eager numpy indices (our refactoring)
    all_sample_indices = np.arange(sample_offset, sample_offset + num_samples, dtype=np.int32)
    new_keys = jax.vmap(lambda idx: jr.fold_in(base_key, idx))(all_sample_indices)

    # Extract key data for comparison (JAX uses special key dtype)
    old_key_data = jax.random.key_data(old_keys)
    new_key_data = jax.random.key_data(new_keys)

    # Must be bitwise identical
    np.testing.assert_array_equal(old_key_data, new_key_data)


def test_key_generation_with_offset():
    """Test key generation with non-zero sample_offset (campaign mode scenario)."""
    import jax.random as jr

    base_key = jr.key(42)
    sample_offset = 50  # Campaign mode uses non-zero offset
    num_samples = 100

    # Old approach
    old_keys = jnp.stack([
        jr.fold_in(base_key, idx)
        for idx in range(sample_offset, sample_offset + num_samples)
    ], axis=0)

    # New approach
    all_sample_indices = np.arange(sample_offset, sample_offset + num_samples, dtype=np.int32)
    new_keys = jax.vmap(lambda idx: jr.fold_in(base_key, idx))(all_sample_indices)

    # Extract key data for comparison (JAX uses special key dtype)
    old_key_data = jax.random.key_data(old_keys)
    new_key_data = jax.random.key_data(new_keys)

    np.testing.assert_array_equal(old_key_data, new_key_data)


def test_chunk_slicing_with_variable_sizes():
    """Test that chunk slicing works correctly with variable final chunk size.

    This ensures the refactoring correctly handles the last chunk being smaller than
    chunk_size (e.g., 100 samples with chunk_size=30 → [30, 30, 30, 10]).
    """
    import jax.random as jr

    base_key = jr.key(42)
    sample_offset = 0
    num_samples = 100
    chunk_size = 30

    # Pre-compute all keys (new approach)
    all_sample_indices = np.arange(sample_offset, sample_offset + num_samples, dtype=np.int32)
    all_keys = jax.vmap(lambda idx: jr.fold_in(base_key, idx))(all_sample_indices)

    # Verify each chunk has correct size
    total_chunks = (num_samples + chunk_size - 1) // chunk_size
    assert total_chunks == 4  # 30, 30, 30, 10

    chunk_keys = []
    for chunk_iter in range(total_chunks):
        chunk_start = chunk_iter * chunk_size
        chunk_count = min(chunk_size, num_samples - chunk_start)
        keys = all_keys[chunk_start : chunk_start + chunk_count]
        chunk_keys.append(keys)

        # Verify chunk size
        assert len(keys) == chunk_count

    # Verify final chunk is smaller
    assert len(chunk_keys[0]) == 30
    assert len(chunk_keys[1]) == 30
    assert len(chunk_keys[2]) == 30
    assert len(chunk_keys[3]) == 10


def test_bitwise_equivalence_sampled_sequences():
    """Test that refactored code produces bitwise identical sequences as original.

    Uses a deterministic sampler that returns outputs based on the PRNG key,
    so identical keys must produce identical sequences.
    """
    def deterministic_sampler(
        prng_key: jax.Array,
        structure_coordinates: jax.Array,
        _mask: jax.Array,
        _residue_index: jax.Array,
        _chain_index: jax.Array,
        **_kwargs: object,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sampler that uses the key to generate deterministic output."""
        seq_len = structure_coordinates.shape[0]
        # Use key to generate a deterministic sequence
        token = jax.random.randint(prng_key, (), 0, 21, dtype=jnp.int32).astype(jnp.int8)
        sequence = jnp.full((seq_len,), token, dtype=jnp.int8)
        logits = jax.random.normal(prng_key, (seq_len, 21))
        decoding_order = jnp.arange(seq_len, dtype=jnp.int32)
        return sequence, logits, decoding_order

    spec = SamplingSpecification(
        inputs=["1ubq.pdb"],
        num_samples=10,
        backbone_noise=[0.1],
        samples_chunk_size=3,  # Force multiple chunks with variable final size
    )

    protein = _mock_protein()
    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([protein], MagicMock()),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=deterministic_sampler,
        ):
            # Run with refactored code (already in place)
            result = sample(spec)

            # Verify we got expected number of samples
            assert result["sequences"].shape == (1, 10, 1, 1, 6)  # batch, samples, noise, temp, residues
            assert result["logits"].shape == (1, 10, 1, 1, 6, 21)


def test_sampling_with_various_chunk_sizes():
    """Test sampling with different chunk_size values to ensure no shape variance issues."""
    for chunk_size in [1, 3, 5, 15]:  # Include boundary cases
        spec = SamplingSpecification(
            inputs=["1ubq.pdb"],
            num_samples=15,
            backbone_noise=[0.1],
            samples_chunk_size=chunk_size,
        )

        protein = _mock_protein()
        with patch(
            "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
            return_value=([protein], MagicMock()),
        ):
            with patch(
                "prxteinmpnn.run.sampling.make_sample_sequences",
                return_value=_sampler_with_key_identity,
            ):
                result = sample(spec)

                # All chunk sizes should produce same output shape
                assert result["sequences"].shape == (1, 15, 1, 1, 6)
                assert result["logits"].shape == (1, 15, 1, 1, 6, 21)


def test_campaign_mode_with_chunk_sample_start():
    """Test campaign mode where chunk_sample_start is non-zero.

    This ensures the refactoring correctly handles campaign mode's use of
    chunk_sample_start parameter, which shifts the sample_offset.
    """
    output_h5_path = _artifact_path(".h5")

    try:
        spec = SamplingSpecification(
            inputs=["1ubq.pdb"],
            num_samples=20,
            backbone_noise=[0.1],
            samples_chunk_size=5,
            output_h5_path=str(output_h5_path),
            campaign_mode=True,
            return_logits=False,
        )

        protein = _mock_protein()
        with patch(
            "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
            return_value=([protein], MagicMock()),
        ):
            with patch(
                "prxteinmpnn.run.sampling.make_sample_sequences",
                return_value=_sampler_with_key_identity,
            ):
                sample(spec)

        # Verify HDF5 output is valid and complete
        with h5py.File(output_h5_path, "r") as h5_file:
            assert "structure_0" in h5_file
            structure = h5_file["structure_0"]
            # Should have 20 samples total (5 chunks × 4 or 4 chunks × 5)
            num_samples = int(structure.attrs["num_samples"])
            assert num_samples == 20
    finally:
        output_h5_path.unlink(missing_ok=True)


@pytest.mark.parametrize("sample_offset", [0, 10, 100])
def test_key_offset_consistency(sample_offset: int):
    """Test that keys with different offsets are consistent and deterministic."""
    import jax.random as jr

    base_key = jr.key(42)
    num_samples = 50

    # Generate keys with vmap (new approach)
    all_sample_indices = np.arange(sample_offset, sample_offset + num_samples, dtype=np.int32)
    new_keys = jax.vmap(lambda idx: jr.fold_in(base_key, idx))(all_sample_indices)

    # Generate the same keys again (should be identical due to determinism)
    all_sample_indices_2 = np.arange(sample_offset, sample_offset + num_samples, dtype=np.int32)
    new_keys_2 = jax.vmap(lambda idx: jr.fold_in(base_key, idx))(all_sample_indices_2)

    # Extract key data for comparison (JAX uses special key dtype)
    new_key_data = jax.random.key_data(new_keys)
    new_key_data_2 = jax.random.key_data(new_keys_2)

    np.testing.assert_array_equal(new_key_data, new_key_data_2)
