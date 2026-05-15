"""COMP-1: Tests for EncoderOutput named container.

RED phase — all tests must fail until EncoderOutput is implemented and the
four encode sites are updated to return it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.types.encodings import EncoderOutput  # does not exist yet


# ---------------------------------------------------------------------------
# 1. Construction and field access
# ---------------------------------------------------------------------------

def test_encoder_output_has_named_fields():
    """EncoderOutput exposes node_features, edge_features, neighbor_indices."""
    L, K, D = 4, 8, 16
    enc = EncoderOutput(
        node_features=jnp.zeros((L, D)),
        edge_features=jnp.zeros((L, K, D)),
        neighbor_indices=jnp.zeros((L, K), dtype=jnp.int32),
    )
    assert enc.node_features.shape == (L, D)
    assert enc.edge_features.shape == (L, K, D)
    assert enc.neighbor_indices.shape == (L, K)


def test_encoder_output_immutable():
    """EncoderOutput cannot be mutated in-place."""
    enc = EncoderOutput(
        node_features=jnp.zeros((4, 16)),
        edge_features=jnp.zeros((4, 8, 16)),
        neighbor_indices=jnp.zeros((4, 8), dtype=jnp.int32),
    )
    with pytest.raises((AttributeError, TypeError)):
        enc.node_features = jnp.ones((4, 16))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. JAX pytree compatibility
# ---------------------------------------------------------------------------

def test_encoder_output_is_jax_pytree():
    """EncoderOutput survives jax.tree.leaves round-trip."""
    enc = EncoderOutput(
        node_features=jnp.ones((4, 16)),
        edge_features=jnp.ones((4, 8, 16)),
        neighbor_indices=jnp.zeros((4, 8), dtype=jnp.int32),
    )
    leaves = jax.tree.leaves(enc)
    assert len(leaves) >= 2  # at least node and edge features are leaves


def test_encoder_output_survives_vmap():
    """EncoderOutput as pytree works through jax.vmap."""
    S, L, K, D = 3, 4, 8, 16
    batched = EncoderOutput(
        node_features=jnp.zeros((S, L, D)),
        edge_features=jnp.zeros((S, L, K, D)),
        neighbor_indices=jnp.zeros((S, L, K), dtype=jnp.int32),
    )

    def identity(enc: EncoderOutput) -> EncoderOutput:
        return enc

    result = jax.vmap(identity)(batched)
    assert result.node_features.shape == (S, L, D)


def test_encoder_output_survives_jit():
    """EncoderOutput flows through jax.jit without retracing issues."""
    enc = EncoderOutput(
        node_features=jnp.zeros((4, 16)),
        edge_features=jnp.zeros((4, 8, 16)),
        neighbor_indices=jnp.zeros((4, 8), dtype=jnp.int32),
    )

    @jax.jit
    def add_one(e: EncoderOutput) -> EncoderOutput:
        return EncoderOutput(
            node_features=e.node_features + 1.0,
            edge_features=e.edge_features + 1.0,
            neighbor_indices=e.neighbor_indices,
        )

    out = add_one(enc)
    assert float(out.node_features[0, 0]) == pytest.approx(1.0)


def test_encoder_output_survives_scan():
    """EncoderOutput as scan carry works correctly."""
    S, L, K, D = 3, 4, 8, 16

    # Stack of per-state encodings as the scanned xs
    node_stack = jnp.zeros((S, L, D))
    edge_stack = jnp.zeros((S, L, K, D))
    nei_stack = jnp.zeros((S, L, K), dtype=jnp.int32)

    def scan_body(carry, per_state):
        nf, ef, ni = per_state
        enc = EncoderOutput(node_features=nf, edge_features=ef, neighbor_indices=ni)
        return carry, enc

    _, stacked = jax.lax.scan(scan_body, None, (node_stack, edge_stack, nei_stack))
    assert stacked.node_features.shape == (S, L, D)


# ---------------------------------------------------------------------------
# 3. Score-site integration: encode sites produce EncoderOutput
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires minimal_bundle_fixture — deferred to COMP-2")
def test_score_conditional_encode_returns_encoder_output(minimal_bundle_fixture):
    """score_conditional kernel internally produces EncoderOutput (smoke test)."""
    pytest.importorskip("prxteinmpnn.inference.score_conditional")


@pytest.mark.skip(reason="requires minimal_encode_fn_fixture — deferred to COMP-2")
def test_conditional_logits_encode_fn_returns_encoder_output(minimal_encode_fn_fixture):
    """encode_fn in conditional_logits returns EncoderOutput, not a raw tuple."""
    encode_fn, _ = minimal_encode_fn_fixture
    result = encode_fn(jnp.zeros((4, 4, 3)), jnp.ones((4,), dtype=jnp.int32),
                       jnp.arange(4), jnp.zeros(4, dtype=jnp.int32))
    assert isinstance(result, EncoderOutput), (
        f"encode_fn must return EncoderOutput, got {type(result)}"
    )
