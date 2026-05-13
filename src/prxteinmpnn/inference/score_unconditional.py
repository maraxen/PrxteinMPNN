"""Unconditional scoring kernel using InferenceBundle."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.typing import Logits
from prxteinmpnn.model.multistate_stack import scatter_stack_to_flat
from prxteinmpnn.model.multistate_sampling import (
    arithmetic_mean_logits,
    geometric_mean_logits,
    product_of_probabilities_logits,
)


def combine_logits(logits_stack: jax.Array, strategy: int, weights: jax.Array) -> jax.Array:
    if strategy == 0:
        return arithmetic_mean_logits(logits_stack, weights)
    elif strategy == 1:
        return geometric_mean_logits(logits_stack, weights)
    else:
        return product_of_probabilities_logits(logits_stack, weights)


def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: dict[str, Any],
) -> Logits:
    """Compute unconditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand

    S = geo.n_states
    
    # We vmap over the S dimension
    def encode_one(c, m, ri, ci, y, yt, ym):
        # Depending on whether the model accepts ligand args
        return model(
            key=k_enc,
            coords=c,
            mask=m,
            residue_index=ri,
            chain_index=ci,
            y=y,
            y_t=yt,
            y_m=ym,
            inference=config.inference,
        )

    # Encode
    if config.use_rolling_state:
        # scan over states
        def scan_body(carry, per_state):
            c, m, ri, ci, y, yt, ym = per_state
            h_V, h_E, E_idx = encode_one(c, m, ri, ci, y, yt, ym)
            # Accumulate running stats if needed; for now, simple stack
            return carry, (h_V, h_E, E_idx)
            
        _, (node_b, edge_b, nei_b) = jax.lax.scan(
            scan_body,
            None,
            (geo.coords, geo.mask, geo.residue_index, geo.chain_index, lig.y, lig.y_t, lig.y_m)
        )
    else:
        node_b, edge_b, nei_b = jax.vmap(encode_one)(
            geo.coords, geo.mask, geo.residue_index, geo.chain_index,
            lig.y, lig.y_t, lig.y_m
        )

    # Decode unconditional
    def decode_one(nb, eb, nei, mk):
        # For unconditional, we don't pass sequence to the decoder
        # But wait, PrxteinMPNN decoder needs h_V, h_E, E_idx, mask
        return model.decoder(nb, eb, nei, mk, key=k_dec, inference=config.inference)

    decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, geo.mask)
    logits_stack = jax.vmap(jax.vmap(model.w_out))(decoded)

    # Add bias
    logits_stack = logits_stack + cond.bias[None, ...]

    # Scatter stack to flat representation
    # tie_group_map is (S, L) giving flat indices
    flat_logits = scatter_stack_to_flat(
        stacked_logits=logits_stack,
        state_flat_rows=geo.state_flat_rows,
        n_flat=geo.n_flat
    )

    # Combine logits across states mapped to the same tie group
    # Note: the combination requires knowing which states tie to which canonical positions.
    # In standard flat mode, this was done via scatter.
    # If S=1, flat_logits is just (L, V)
    if S > 1:
        # TODO: Implement proper combining according to logit_combine_strategy
        pass

    return flat_logits
