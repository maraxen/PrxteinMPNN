"""Autoregressive sampling kernel using InferenceBundle."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.utils.types import Logits

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> jax.Array:
    """Compute autoregressive samples using wave scheduling."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand
    wave = bundle.wave

    L = geo.n_canonical
    S = geo.n_states
    
    def encode_one(c, m, ri, ci, y, yt, ym, sm):
        return model(
            key=k_enc,
            coords=c,
            mask=m,
            residue_index=ri,
            chain_index=ci,
            y=y,
            y_t=yt,
            y_m=ym,
            structure_mapping=sm,
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
            (geo.coords, geo.mask, geo.residue_index, geo.chain_index, lig.y, lig.y_t, lig.y_m, geo.structure_mapping)
        )
    else:
        node_b, edge_b, nei_b = jax.vmap(encode_one)(
            geo.coords, geo.mask, geo.residue_index, geo.chain_index,
            lig.y, lig.y_t, lig.y_m, geo.structure_mapping
        )

    def step_fn(i, sequence):
        # Current position in decoding order
        # wave.group_positions[i, 0, 0] is the position index if sequential
        pos = wave.group_positions[i, 0, 0]
        group_id = cond.tie_group_map[0, pos]

        # Check if this is the first time we encounter this group in the decoding order
        # tie_group_at_order: (L,)
        tie_group_at_order = cond.tie_group_map[0, wave.group_positions[:, 0, 0]]
        first_occurrence_idx = jnp.argmax(tie_group_at_order == group_id)
        is_first = (first_occurrence_idx == i)

        def do_sample(seq):
            # Update ConditioningBundle with current sequence
            current_cond = replace(cond,
                sequence_oh=jax.nn.one_hot(seq, 21)
            )
            current_bundle = replace(bundle, conditioning=current_cond)

            # Get logits from model
            # logits: (S, L, 21)
            logits = model(k_dec, bundle=current_bundle, config=config)

            # Combine logits across states
            # combined_logits: (L, 21)
            combined_logits = stage_set.logit_transform(logits)

            # Apply bias
            combined_logits = combined_logits + cond.bias

            # Logit averaging for the group
            mask = (cond.tie_group_map[0] == group_id)
            group_logits = jnp.where(mask[:, None], combined_logits, -jnp.inf)
            n_tied = jnp.sum(mask)
            avg_logits = jax.scipy.special.logsumexp(group_logits, axis=0) - jnp.log(jnp.maximum(n_tied, 1))

            # Sample
            subkey = jax.random.fold_in(k_dec, group_id)
            sampled = jax.random.categorical(subkey, avg_logits / config.temperature)
            
            # Update all positions in the group
            # Respect fixed positions individually within the group (though usually they match)
            group_indices = jnp.where(mask, jnp.arange(L), -1)
            
            def update_one(s, idx):
                is_valid = idx != -1
                idx_safe = jnp.where(is_valid, idx, 0)
                is_fixed = cond.fixed_mask[idx_safe]
                fixed_val = cond.fixed_tokens[idx_safe]
                token = jnp.where(is_fixed, fixed_val, sampled).astype(jnp.int32)
                return jnp.where(is_valid, s.at[idx_safe].set(token), s)

            # We can't easily loop over group_indices in JIT if size is dynamic.
            # But we can use sequence.at[group_indices_fixed].set(tokens)
            
            # If any position in the group is fixed, the whole group should use that token.
            # We take the maximum token value among fixed positions in the group 
            # (assuming they are consistent if multiple are fixed).
            is_group_fixed = jnp.any(cond.fixed_mask.astype(jnp.bool_) & mask)
            group_fixed_token = jnp.max(jnp.where(cond.fixed_mask.astype(jnp.bool_) & mask, cond.fixed_tokens, 0))
            
            final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)
            return jnp.where(mask, final_token, seq)

        # Only perform expensive sampling if this is the first occurrence of the group
        return jax.lax.cond(is_first, do_sample, lambda s: s, sequence)

    seq_init = jnp.where(cond.fixed_mask > 0.5, cond.fixed_tokens, 0).astype(jnp.int32)
    
    def scan_body(sequence, i):
        return step_fn(i, sequence), None

    final_seq, _ = jax.lax.scan(scan_body, seq_init, jnp.arange(wave.group_positions.shape[0]))
    
    # Return updated bundle
    final_cond = replace(cond, sequence_oh=jax.nn.one_hot(final_seq, 21))
    return replace(bundle, conditioning=final_cond)
