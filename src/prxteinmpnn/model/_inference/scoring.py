"""Scoring inference logic using pure Pytree geometry."""

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model_inputs import ConditionalInputs, UnconditionalInputs
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.typing import Logits
from prxteinmpnn.model._shared import encoder_forward_with_int_neighbors

def score_unconditional(
    model: PrxteinMPNN,
    prng_key: PRNGKeyArray,
    inputs: UnconditionalInputs,
    *,
    inference: bool = True,
    logit_transform_fn = None,
    encoder_state_fn = None,
) -> Logits:
    """Compute unconditional logits per stacked state (S, L, V)."""
    k_enc, k_feat = jax.random.split(prng_key)
    coords = inputs.state_stack.coords
    mask = inputs.state_stack.mask
    residue_index = inputs.state_stack.residue_index
    chain_index = inputs.state_stack.chain_index

    def encode_one(c, m, r, ch):
        ef, nei, nf, _ = model.features(
            k_feat, c, m, r, ch,
            jnp.asarray(0.0, jnp.float32),
            structure_mapping=None,
            initial_node_features=None,
            rbf_features=None,
            neighbor_indices=None,
        )
        return encoder_forward_with_int_neighbors(
            model.encoder, ef, nei, m, nf, inference=inference, key=k_enc
        )

    if encoder_state_fn is not None:
        def scan_body(carry, per_state):
            coords_s, mask_s, ri_s, ci_s, idx_s = per_state
            from prxteinmpnn.model_inputs import BackboneGeometry
            backbone_s = BackboneGeometry(
                coords=coords_s, mask=mask_s,
                residue_index=ri_s, chain_index=ci_s,
            )
            new_carry, enc_out = encoder_state_fn(carry, idx_s, backbone_s)
            return new_carry, enc_out

        _, enc_stacked = jax.lax.scan(
            scan_body,
            encoder_state_fn.init_carry(),
            (coords, mask, residue_index, chain_index,
             jnp.arange(coords.shape[0], dtype=jnp.int32)),
        )
        node_b = enc_stacked.node_features
        edge_b = enc_stacked.edge_features
        nei_b = enc_stacked.neighbor_indices
    else:
        node_b, edge_b, nei_b = jax.vmap(encode_one)(
            coords, mask, residue_index, chain_index
        )

    def decode_one(nb, eb, nei, mk):
        return model.decoder(nb, eb, nei, mk, key=k_enc)

    decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask)
    logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)

    if logit_transform_fn is not None:
        # Default state weights and mapping since we removed them from core input signature
        _sw = jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        _si = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
        merged = logit_transform_fn(logits_s, _si, _sw)
        return merged

    return logits_s


def score_conditional(
    model: PrxteinMPNN,
    prng_key: PRNGKeyArray,
    inputs: ConditionalInputs,
    *,
    inference: bool = True,
    logit_transform_fn = None,
    encoder_state_fn = None,
) -> Logits:
    """Compute teacher-forced conditional logits per stacked state (S, L, V)."""
    k_enc, k_feat = jax.random.split(prng_key)
    coords = inputs.state_stack.coords
    mask = inputs.state_stack.mask
    residue_index = inputs.state_stack.residue_index
    chain_index = inputs.state_stack.chain_index
    seq_oh_stack = inputs.seq_oh_stack
    ar_mask_stack = inputs.ar_mask_stack

    def encode_one(c, m, r, ch):
        ef, nei, nf, _ = model.features(
            k_feat, c, m, r, ch,
            jnp.asarray(0.0, jnp.float32),
            structure_mapping=None,
            initial_node_features=None,
            rbf_features=None,
            neighbor_indices=None,
        )
        return encoder_forward_with_int_neighbors(
            model.encoder, ef, nei, m, nf, inference=inference, key=k_enc
        )

    if encoder_state_fn is not None:
        def scan_body(carry, per_state):
            coords_s, mask_s, ri_s, ci_s, idx_s = per_state
            from prxteinmpnn.model_inputs import BackboneGeometry
            backbone_s = BackboneGeometry(
                coords=coords_s, mask=mask_s,
                residue_index=ri_s, chain_index=ci_s,
            )
            new_carry, enc_out = encoder_state_fn(carry, idx_s, backbone_s)
            return new_carry, enc_out

        _, enc_stacked = jax.lax.scan(
            scan_body,
            encoder_state_fn.init_carry(),
            (coords, mask, residue_index, chain_index,
             jnp.arange(coords.shape[0], dtype=jnp.int32)),
        )
        node_b = enc_stacked.node_features
        edge_b = enc_stacked.edge_features
        nei_b = enc_stacked.neighbor_indices
    else:
        node_b, edge_b, nei_b = jax.vmap(encode_one)(
            coords, mask, residue_index, chain_index
        )

    def dec_one(nb, eb, nei, mk, arm, oh):
        return model.decoder.call_conditional(
            nb, eb, nei, mk, arm, oh, model.w_s_embed.weight, inference=inference, key=k_enc
        )

    decoded = jax.vmap(dec_one)(
        node_b, edge_b, nei_b, mask, ar_mask_stack, seq_oh_stack
    )

    logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
    
    if inputs.bias_stack is not None:
        logits_s = logits_s + inputs.bias_stack

    if logit_transform_fn is not None:
        _sw = jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        _si = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
        merged = logit_transform_fn(logits_s, _si, _sw)
        return merged

    return logits_s
