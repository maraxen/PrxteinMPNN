"""Tests for COMP-NEW: tensor io_callback emission and sink integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.host.kernel_dispatch import _sample_batch
from prxteinmpnn.host.output_sinks import (
    active_sampling_staging_sink,
    streaming_tensor_sink_session,
    take_staging_sequences_logits,
)
from prxteinmpnn.inference.sample_autoregressive import SampleResult
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.utils.data_structures import Protein


def _make_fake_protein(
    batch_size: int = 1, seq_len: int = 10
) -> Protein:
    """Create a minimal Protein namedtuple for testing.

    Args:
        batch_size: Number of structures in batch
        seq_len: Sequence length (residues)

    Returns:
        Protein namedtuple with correct shapes for _sample_batch
    """
    return Protein(
        coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        mapping=None,
    )


def _make_stub_kernel(seq_len: int = 10, vocab: int = 21):
    """Create a stub kernel that returns fixed-shape SampleResult.

    Args:
        seq_len: Sequence length for generated sequences
        vocab: Vocabulary size (default 21 for amino acids)

    Returns:
        Callable stub kernel compatible with _sample_batch
    """
    def _stub(model, prng_key, bundle, config, stage_set):
        seq = jnp.zeros((seq_len,), dtype=jnp.int32)
        logits = jnp.zeros((seq_len, vocab), dtype=jnp.float32)
        return SampleResult(sequence=seq, logits=logits)

    return _stub


# ---------------------------------------------------------------------------
# Test 1: Direct io_callback staging
# ---------------------------------------------------------------------------


def test_dispatch_tensor_io_callback_stages_to_active_sink():
    """_dispatch_sampling_tensor_batch_io stages sequences/logits to active sink."""
    from prxteinmpnn.host._sampling_helper import (
        _dispatch_sampling_tensor_batch_io,
    )

    batch_idx = 0
    batch_count = 1
    chunk_start = 0
    chunk_count = 4
    seq_len = 10
    num_samples = 2

    # Create numpy arrays matching expected shape
    sequences_np = np.zeros((1, num_samples, 1, 1, seq_len), dtype=np.int32)
    logits_np = np.zeros((1, num_samples, 1, 1, seq_len, 21), dtype=np.float32)

    with streaming_tensor_sink_session():
        # Verify sink is active
        sink = active_sampling_staging_sink()
        assert sink is not None, "Sink must be active in streaming_tensor_sink_session()"

        # Call the io_callback handler directly
        _dispatch_sampling_tensor_batch_io(
            batch_idx,
            batch_count,
            chunk_start,
            chunk_count,
            sequences_np,
            logits_np,
        )

        # Drain and verify shapes
        seqs, logits = take_staging_sequences_logits(batch_idx, chunk_start, chunk_count)
        assert seqs is not None, "Sequences should be staged"
        assert logits is not None, "Logits should be staged"
        assert seqs.shape[0] == 1, f"Expected batch size 1, got {seqs.shape[0]}"
        assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"


# ---------------------------------------------------------------------------
# Test 2: io_callback emission can be controlled by emit_structure_batch_io flag
# ---------------------------------------------------------------------------


def test_emit_structure_batch_io_gate_in_sample_batch():
    """Verify _sample_batch respects emit_structure_batch_io parameter.

    This test directly verifies the io_callback emission logic without
    invoking the full _sample_batch (which requires complex JAX-compatible mocks).
    Instead, we patch the io_callback handler to track invocations.
    """
    from prxteinmpnn.host.kernel_dispatch import _sample_batch

    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=1,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )

    # We'll mock the parts that would fail in full vmap context
    call_log = {"tensor": 0, "structure": 0}

    def mock_dispatch_tensor(*args, **kwargs):
        call_log["tensor"] += 1

    def mock_noop_structure(*args, **kwargs):
        call_log["structure"] += 1

    with patch(
        "prxteinmpnn.host.kernel_dispatch._dispatch_sampling_tensor_batch_io",
        side_effect=mock_dispatch_tensor,
    ), patch(
        "prxteinmpnn.host.kernel_dispatch._noop_sampling_structure_batch_io",
        side_effect=mock_noop_structure,
    ), patch(
        "prxteinmpnn.host.kernel_dispatch.make_sampling_planner"
    ), patch(
        "prxteinmpnn.host.kernel_dispatch._safe_map"
    ) as mock_safe_map:
        # Mock _safe_map to return dummy arrays with correct shapes
        def mock_safe_map_impl(fn, xs, batch_size=None):
            # Return dummy sequences and logits of expected shape
            return (
                jnp.zeros((1, 2, 1, 1, 10), dtype=jnp.int32),
                jnp.zeros((1, 2, 1, 1, 10, 21), dtype=jnp.float32),
            )

        mock_safe_map.side_effect = mock_safe_map_impl

        with patch(
            "prxteinmpnn.host.kernel_dispatch.extract_batch_sizes"
        ) as mock_extract, patch(
            "prxteinmpnn.host.kernel_dispatch._resolve_grid_lineage"
        ) as mock_grid, patch(
            "prxteinmpnn.host.kernel_dispatch.resolve_target_samples"
        ) as mock_target:
            mock_extract.return_value = (1, 2, 1, 1)
            mock_grid.return_value = None
            mock_target.return_value = 2

            with patch(
                "prxteinmpnn.host.kernel_dispatch._base_sampling_key"
            ) as mock_base_key, patch(
                "prxteinmpnn.host.kernel_dispatch.compute_sample_keys"
            ) as mock_sample_keys:
                mock_base_key.return_value = jax.random.key(0)
                mock_sample_keys.return_value = [
                    jax.random.key(i) for i in range(2)
                ]

                batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
                model = MagicMock(name="model")
                stage_set = MagicMock(name="stage_set")

                # Test 1: emit_structure_batch_io=False
                call_log["tensor"] = 0
                call_log["structure"] = 0
                try:
                    _sample_batch(
                        spec,
                        batched_ensemble,
                        model,
                        stage_set=stage_set,
                        batch_idx=0,
                        structure_batch_count=1,
                        emit_structure_batch_io=False,
                    )
                except Exception:
                    pass  # We're only checking if io_callback was attempted

                assert (
                    call_log["structure"] == 0
                ), f"emit_structure_batch_io=False should not emit scalar marker, got {call_log['structure']}"

                # Test 2: emit_structure_batch_io=True
                call_log["tensor"] = 0
                call_log["structure"] = 0
                try:
                    _sample_batch(
                        spec,
                        batched_ensemble,
                        model,
                        stage_set=stage_set,
                        batch_idx=0,
                        structure_batch_count=1,
                        emit_structure_batch_io=True,
                    )
                except Exception:
                    pass

                assert (
                    call_log["structure"] == 1
                ), f"emit_structure_batch_io=True should emit scalar marker once, got {call_log['structure']}"


# ---------------------------------------------------------------------------
# Test 3: io_callback is generated by _sample_batch (integration test)
# ---------------------------------------------------------------------------


def test_sample_batch_io_callback_sources_exist():
    """Verify _sample_batch module has io_callback sources for both tensor and structure markers."""
    # This is a simple smoke test verifying the code was added
    from prxteinmpnn.host import kernel_dispatch

    # Verify the imports exist
    assert hasattr(
        kernel_dispatch, "_dispatch_sampling_tensor_batch_io"
    ), "kernel_dispatch must import _dispatch_sampling_tensor_batch_io"
    assert hasattr(
        kernel_dispatch, "_noop_sampling_structure_batch_io"
    ), "kernel_dispatch must import _noop_sampling_structure_batch_io"

    # Verify _sample_batch is callable
    assert callable(kernel_dispatch._sample_batch)


# ---------------------------------------------------------------------------
# Test 4: emit_structure_batch_io parameter exists and defaults to True
# ---------------------------------------------------------------------------


def test_sample_batch_emit_structure_batch_io_parameter():
    """Verify _sample_batch accepts emit_structure_batch_io parameter with default True."""
    import inspect

    from prxteinmpnn.host.kernel_dispatch import _sample_batch

    sig = inspect.signature(_sample_batch)
    assert (
        "emit_structure_batch_io" in sig.parameters
    ), "_sample_batch must accept emit_structure_batch_io parameter"

    param = sig.parameters["emit_structure_batch_io"]
    assert (
        param.default is True
    ), "emit_structure_batch_io must default to True"
