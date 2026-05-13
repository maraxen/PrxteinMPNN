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



from prxteinmpnn.types.stages import StageSet

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Compute unconditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand

    S = geo.n_states
    
    # We vmap over the S dimension
    def encode_one(
        coords: jax.Array, 
        mask: jax.Array, 
        residue_index: jax.Array, 
        chain_index: jax.Array, 
        ligand_y: jax.Array, 
        ligand_y_t: jax.Array, 
        ligand_y_m: jax.Array
    ):
        return model(
            key=k_enc,
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            y=ligand_y,
            y_t=ligand_y_t,
            y_m=ligand_y_m,
            inference=config.inference,
        )

    # Encode
    if config.use_rolling_state:
        # scan over states
        def scan_body(carry: Any, per_state: tuple[jax.Array, ...]):
            coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m = per_state
            h_V, h_E, E_idx = encode_one(coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m)
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
    def decode_one(nb: jax.Array, eb: jax.Array, nei: jax.Array, mk: jax.Array):
        # For unconditional, we don't pass sequence to the decoder
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
        logit_transform = stage_set.get("logit_transform")
        if logit_transform is not None:
            # logit_transform is expected to handle combining the stack S
            # according to whatever weights/strategy were bound to it.
            logits_stack = logit_transform(logits_stack)
            # Re-scatter to flat after combining if needed, or if logit_transform 
            # returns a flat structure directly. 
            # Note: For pure stacking, if the output is just L x V, we might not need to scatter.
            # But we leave scatter for now if we still need flat graph parity.
            flat_logits = scatter_stack_to_flat(
                stacked_logits=jnp.expand_dims(logits_stack, axis=0), # S=1 now
                state_flat_rows=geo.state_flat_rows[:1],
                n_flat=geo.n_flat
            )
    return flat_logits
