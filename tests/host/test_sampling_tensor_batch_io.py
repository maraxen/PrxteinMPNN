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
# Test 2: _sample_batch stages tensors when sink is active
# ---------------------------------------------------------------------------


def test_sample_batch_stages_when_sink_active():
    """_sample_batch emits io_callbacks that stage tensors to active sink."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )

    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    model = MagicMock(name="model")
    stage_set = MagicMock(name="stage_set")

    with patch(
        "prxteinmpnn.host.kernel_dispatch.build_inference_bundle"
    ) as mock_bundle, patch(
        "prxteinmpnn.host.kernel_dispatch.resolve_kernel_fn"
    ) as mock_resolve_kernel:
        # Mock bundle builder to return dummy bundle/config
        mock_bundle.return_value = (
            MagicMock(name="bundle"),
            MagicMock(name="config"),
        )

        # Mock kernel resolver to return stub kernel
        stub_kernel = _make_stub_kernel(seq_len=10)
        mock_resolve_kernel.return_value = stub_kernel

        with streaming_tensor_sink_session():
            # Call _sample_batch; should emit io_callbacks
            _, _, _ = _sample_batch(
                spec,
                batched_ensemble,
                model,
                stage_set=stage_set,
                batch_idx=0,
                structure_batch_count=1,
            )

            # Flush pending io_callbacks
            jax.effects_barrier()

            # Drain and verify
            seqs, logits = take_staging_sequences_logits(0, 0, 2)
            assert seqs is not None, "Sequences should be staged by io_callback"
            assert logits is not None, "Logits should be staged by io_callback"
            assert seqs.shape[0] == 1, f"Expected batch size 1, got {seqs.shape[0]}"


# ---------------------------------------------------------------------------
# Test 3: _sample_batch does not raise without active sink
# ---------------------------------------------------------------------------


def test_sample_batch_does_not_raise_without_active_sink():
    """_sample_batch does not raise when no streaming sink is active."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )

    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    model = MagicMock(name="model")
    stage_set = MagicMock(name="stage_set")

    with patch(
        "prxteinmpnn.host.kernel_dispatch.build_inference_bundle"
    ) as mock_bundle, patch(
        "prxteinmpnn.host.kernel_dispatch.resolve_kernel_fn"
    ) as mock_resolve_kernel:
        # Mock bundle builder
        mock_bundle.return_value = (
            MagicMock(name="bundle"),
            MagicMock(name="config"),
        )

        # Mock kernel
        stub_kernel = _make_stub_kernel(seq_len=10)
        mock_resolve_kernel.return_value = stub_kernel

        # Call _sample_batch WITHOUT streaming_tensor_sink_session()
        # Should not raise RuntimeError
        _, _, _ = _sample_batch(
            spec,
            batched_ensemble,
            model,
            stage_set=stage_set,
            batch_idx=0,
            structure_batch_count=1,
        )

        # Flush and verify no error
        jax.effects_barrier()


# ---------------------------------------------------------------------------
# Test 4: emit_structure_batch_io=False suppresses scalar marker io_callback
# ---------------------------------------------------------------------------


def test_emit_structure_batch_io_false_skips_scalar_marker():
    """emit_structure_batch_io=False suppresses _noop_sampling_structure_batch_io call."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )

    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    model = MagicMock(name="model")
    stage_set = MagicMock(name="stage_set")

    # Create a counting stub for the noop handler
    call_count = {"count": 0}

    def counting_noop(*args, **kwargs):
        call_count["count"] += 1

    with patch(
        "prxteinmpnn.host.kernel_dispatch.build_inference_bundle"
    ) as mock_bundle, patch(
        "prxteinmpnn.host.kernel_dispatch.resolve_kernel_fn"
    ) as mock_resolve_kernel, patch(
        "prxteinmpnn.host.kernel_dispatch._noop_sampling_structure_batch_io",
        side_effect=counting_noop,
    ):
        # Mock bundle and kernel
        mock_bundle.return_value = (
            MagicMock(name="bundle"),
            MagicMock(name="config"),
        )
        stub_kernel = _make_stub_kernel(seq_len=10)
        mock_resolve_kernel.return_value = stub_kernel

        # Test 1: emit_structure_batch_io=False should NOT call the noop handler
        call_count["count"] = 0
        _, _, _ = _sample_batch(
            spec,
            batched_ensemble,
            model,
            stage_set=stage_set,
            batch_idx=0,
            structure_batch_count=1,
            emit_structure_batch_io=False,
        )
        jax.effects_barrier()
        assert call_count["count"] == 0, (
            f"emit_structure_batch_io=False should not call noop handler, "
            f"but was called {call_count['count']} times"
        )

        # Test 2: emit_structure_batch_io=True SHOULD call the noop handler
        call_count["count"] = 0
        _, _, _ = _sample_batch(
            spec,
            batched_ensemble,
            model,
            stage_set=stage_set,
            batch_idx=0,
            structure_batch_count=1,
            emit_structure_batch_io=True,
        )
        jax.effects_barrier()
        assert call_count["count"] == 1, (
            f"emit_structure_batch_io=True should call noop handler once, "
            f"but was called {call_count['count']} times"
        )
