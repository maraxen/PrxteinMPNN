"""Autoregressive sampling kernel using InferenceBundle."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.typing import Logits
from prxteinmpnn.model.multistate_stack import scatter_stack_to_flat

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: dict[str, Any],
) -> jax.Array:
    """Compute autoregressive samples using wave scheduling."""
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

    # Decode AR
    # Here we would use jax.lax.scan over the wave schedule to decode autoregressively.
    # For now we create a mock to fill in the structure.
    
    # ... placeholder for AR scan ...
    
    # return the sampled sequence
    return jnp.zeros((geo.n_flat,), dtype=jnp.int32)
