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
