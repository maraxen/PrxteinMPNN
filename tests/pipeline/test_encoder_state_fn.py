"""Verify EncoderStateFn carry-based scan is called during unconditional/conditional scoring."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload, EncoderOutput
from prxteinmpnn.model_inputs import BackboneGeometry
from prxteinmpnn.protocols import EncoderStateFn
from prxteinmpnn.pipeline_registry import StageSet


def _make_model():
    return eqx.tree_inference(
        PrxteinMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )


def _make_stack(S=2, L=6):
    return MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.array([0, L, S * L], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )


def _make_passthrough_encoder(m):
    """Stateless scan encoder: identical to vmap encode_one."""
    class PassthroughEncoder:
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return ()

        def __call__(self, carry, state_idx, backbone):
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf,
                inference=True, key=None,
            )
            return (), EncoderOutput(nf2, ef2, nei2, backbone.mask)

    return PassthroughEncoder(m)


def test_encoder_state_fn_carry_accumulates():
    """Carry must accumulate across S states (scan runs S times, not once)."""
    m = _make_model()
    stack = _make_stack(S=3, L=4)
    S = stack.n_states

    class CountingEncoder:
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return jnp.zeros((), dtype=jnp.int32)

        def __call__(self, carry, state_idx, backbone):
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf,
                inference=True, key=None,
            )
            new_carry = carry + jnp.ones((), dtype=jnp.int32)
            return new_carry, EncoderOutput(nf2, ef2, nei2, backbone.mask)

    encoder_fn = CountingEncoder(m)

    # Directly test carry accumulation via scan
    def scan_body(carry, per_state):
        coords_s, mask_s, ri_s, ci_s, idx_s = per_state
        backbone_s = BackboneGeometry(
            coords=coords_s, mask=mask_s,
            residue_index=ri_s, chain_index=ci_s,
        )
        return encoder_fn(carry, idx_s, backbone_s)

    init_carry = encoder_fn.init_carry()
    final_carry, enc_stacked = jax.lax.scan(
        scan_body,
        init_carry,
        (stack.coords_stack, stack.mask_stack,
         stack.residue_index_stack, stack.chain_index_stack,
         stack.state_index),
    )
    assert int(final_carry) == S, f"Expected carry={S}, got {int(final_carry)}"
    assert enc_stacked.node_features.shape[0] == S


def test_encoder_state_fn_passthrough_matches_vmap():
    """Passthrough scan path must produce logits identical to vmap path."""
    m = _make_model()
    stack = _make_stack(S=2, L=4)
    key = jax.random.PRNGKey(42)

    logits_vmap = m.score_unconditional_from_payload(
        key, stack,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=None,
    )
    logits_scan = m.score_unconditional_from_payload(
        key, stack,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=_make_passthrough_encoder(m),
    )
    assert jnp.allclose(logits_vmap, logits_scan, atol=1e-5)


def test_encoder_state_fn_in_conditional_path():
    """Passthrough scan must match vmap path for score_conditional_from_payload."""
    m = _make_model()
    stack = _make_stack(S=2, L=4)
    S, L, V = stack.n_states, stack.n_canonical, 21
    seq_oh = jnp.zeros((S, L, V))
    ar_mask = jnp.eye(L, dtype=jnp.float32)[None].repeat(S, axis=0)
    key = jax.random.PRNGKey(7)

    logits_vmap = m.score_conditional_from_payload(
        key, stack,
        seq_oh_stack=seq_oh, ar_mask_stack=ar_mask,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=None,
    )
    logits_scan = m.score_conditional_from_payload(
        key, stack,
        seq_oh_stack=seq_oh, ar_mask_stack=ar_mask,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=_make_passthrough_encoder(m),
    )
    assert logits_vmap.shape == logits_scan.shape
    assert jnp.allclose(logits_vmap, logits_scan, atol=1e-5)


def test_unconditional_pipeline_resolves_encoder_state_fn():
    """UnconditionalPipeline must resolve and thread encoder_state_fn from stage_set."""
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline

    m = _make_model()
    stage_set = StageSet.from_callables(encoder_state_fn=_make_passthrough_encoder(m))
    pipeline = UnconditionalPipeline()
    stack = _make_stack()
    logits, state_logits = pipeline(m, jax.random.PRNGKey(0), stack, stage_set=stage_set)
    assert logits.shape[-1] == 21
