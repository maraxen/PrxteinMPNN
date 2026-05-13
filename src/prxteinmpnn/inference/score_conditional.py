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


from prxteinmpnn.types.stages import StageSet

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Compute teacher-forced conditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand
    wave = bundle.wave

    S = geo.n_states
    
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

    if config.use_rolling_state:
        def scan_body(carry: Any, per_state: tuple[jax.Array, ...]):
            coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m = per_state
            h_V, h_E, E_idx = encode_one(coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m)
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

    def decode_one(nb: jax.Array, eb: jax.Array, nei: jax.Array, mk: jax.Array, arm: jax.Array, oh: jax.Array):
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
        logit_transform = stage_set.get("logit_transform")
        if logit_transform is not None:
            logits_stack = logit_transform(logits_stack)
            flat_logits = scatter_stack_to_flat(
                stacked_logits=jnp.expand_dims(logits_stack, axis=0),
                state_flat_rows=geo.state_flat_rows[:1],
                n_flat=geo.n_flat
            )

    return flat_logits
