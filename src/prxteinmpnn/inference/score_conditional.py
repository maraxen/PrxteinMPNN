"""Conditional scoring kernel using InferenceBundle."""

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


def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: dict[str, Any],
) -> Logits:
    """Compute teacher-forced conditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand
    wave = bundle.wave

    S = geo.n_states
    
    def encode_one(c, m, ri, ci, y, yt, ym):
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

    if config.use_rolling_state:
        def scan_body(carry, per_state):
            c, m, ri, ci, y, yt, ym = per_state
            h_V, h_E, E_idx = encode_one(c, m, ri, ci, y, yt, ym)
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

    def decode_one(nb, eb, nei, mk, arm, oh):
        return model.decoder.call_conditional(
            nb, eb, nei, mk, arm, oh, model.w_s_embed.weight, inference=config.inference, key=k_dec
        )

    # For conditional scoring, wave provides the AR masks, or we assume a full mask?
    seq_oh_stack = jnp.broadcast_to(cond.sequence_oh[None, ...], (S, *cond.sequence_oh.shape))
    
    decoded = jax.vmap(decode_one)(
        node_b, edge_b, nei_b, geo.mask, cond.ar_mask, seq_oh_stack
    )

    logits_stack = jax.vmap(jax.vmap(model.w_out))(decoded)

    logits_stack = logits_stack + cond.bias[None, ...]

    flat_logits = scatter_stack_to_flat(
        stacked_logits=logits_stack,
        state_flat_rows=geo.state_flat_rows,
        n_flat=geo.n_flat
    )

    if S > 1:
        # TODO: Implement combining
        pass

    return flat_logits
