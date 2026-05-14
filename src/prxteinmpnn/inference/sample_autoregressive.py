"""Autoregressive sampling kernel for PrxteinMPNN.

This kernel implements the core sampling loop, optimized for JIT and vmap.
It consumes a unified InferenceBundle and returns a structured SampleResult.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.stages import StageSet
    from prxteinmpnn.types.arrays import PRNGKeyArray


@dataclass(frozen=True)
class SampleResult:
    """Result of an autoregressive sampling run."""
    sequence: Int[Array, "L"]
    logits: Float[Array, "L 21"]


def kernel(
    model: PrxteinMPNN,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> SampleResult:
    """Autoregressive sampling kernel.

    Optimized to encode features once and then iterate through the decoding waves.
    """
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand
    wave = bundle.wave
    
    L = geo.n_canonical
    S = geo.n_states
    
    # 1. Encode once (features are fixed for all decoding steps)
    noise_stack = jnp.broadcast_to(geo.backbone_noise, (S,))

    def encode_one(c, m, ri, ci, sm, noise):
        return model.features(
            k_enc, c, m, ri, ci, noise,
            backbone_noise_mode=geo.backbone_noise_mode,
            structure_mapping=sm
        )

    edge_f, edge_i, node_f, _ = jax.vmap(encode_one, in_axes=(0, 0, 0, 0, 0, 0))(
        geo.coords, geo.mask, geo.residue_index, geo.chain_index,
        geo.structure_mapping, noise_stack
    )
    
    node_f, edge_f = jax.vmap(model.encoder, in_axes=(0, 0, 0, 0, 0))(
        edge_f, edge_i, geo.mask, initial_node_features=node_f,
        key=jax.random.split(k_enc, S)
    )

    # 2. Decoding Loop
    def step_fn(i, sequence):
        # Current position in decoding order
        pos = wave.group_positions[i, 0, 0]
        group_id = cond.tie_group_map[0, pos]

        # Check if this is the first time we encounter this group in the decoding order
        tie_group_at_order = cond.tie_group_map[0, wave.group_positions[:, 0, 0]]
        first_occurrence_idx = jnp.argmax(tie_group_at_order == group_id)
        is_first = (first_occurrence_idx == i)

        def do_sample(seq):
            # One-hot sequence
            seq_oh = jax.nn.one_hot(seq, 21)

            # Decode (vmap over states)
            def decode_one(n, e, idx, m, arm):
                return model.decoder.call_conditional(
                    n, e, idx, m, arm, seq_oh, model.w_s_embed.weight,
                    key=k_dec, inference=config.inference
                )

            # decoded: (S, L, H)
            decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0))(node_f, edge_f, edge_i, geo.mask, cond.ar_mask)
            
            # Project to logits: (S, L, 21)
            logits = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

            # Combine logits across states: (L, 21)
            combined_logits = stage_set.logit_transform(logits)

            # Apply bias
            combined_logits = combined_logits + cond.bias

            # Logit averaging for the group
            mask = (cond.tie_group_map[0] == group_id)
            group_logits = jnp.where(mask[:, None], combined_logits, -jnp.inf)
            n_tied = jnp.sum(mask)
            # Use logsumexp for averaging in log-space
            avg_logits = jax.scipy.special.logsumexp(group_logits, axis=0) - jnp.log(jnp.maximum(n_tied, 1))

            # Sample
            subkey = jax.random.fold_in(k_dec, group_id)
            sampled = jax.random.categorical(subkey, avg_logits / cond.temperature)
            
            # Update all positions in the group
            is_group_fixed = jnp.any(cond.fixed_mask.astype(jnp.bool_) & mask)
            group_fixed_token = jnp.max(jnp.where(cond.fixed_mask.astype(jnp.bool_) & mask, cond.fixed_tokens, 0))
            final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)
            
            new_seq = jnp.where(mask, final_token, seq)
            return new_seq, combined_logits[pos]

        def no_sample(seq):
            return seq, jnp.zeros((21,))

        return jax.lax.cond(is_first, do_sample, no_sample, sequence)

    seq_init = jnp.where(cond.fixed_mask > 0.5, cond.fixed_tokens, 0).astype(jnp.int32)
    
    def scan_body(sequence, i):
        new_seq, step_logits = step_fn(i, sequence)
        return new_seq, step_logits

    # Run scan over waves
    final_seq, logits_stack = jax.lax.scan(
        scan_body,
        seq_init,
        jnp.arange(wave.n_waves)
    )

    # 3. Map logits_stack (W, 21) back to (L, 21)
    def scatter_logits(logits_final, i):
        pos = wave.group_positions[i, 0, 0]
        group_id = cond.tie_group_map[0, pos]
        mask = (cond.tie_group_map[0] == group_id)
        step_logits = logits_stack[i]
        new_logits_final = jnp.where(mask[:, None], step_logits, logits_final)
        return new_logits_final, None

    logits_init = jnp.zeros((L, 21))
    logits_final, _ = jax.lax.scan(scatter_logits, logits_init, jnp.arange(wave.n_waves))

    return SampleResult(
        sequence=final_seq,
        logits=logits_final
    )
