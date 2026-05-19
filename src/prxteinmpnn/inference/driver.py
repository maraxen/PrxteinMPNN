"""Unified StageSet-driven decode driver.

This module consolidates the three inference kernels (score_conditional,
score_unconditional, sample_autoregressive) into a single unified driver
that dispatches based on stage_set topology at call time.

Topology inference:
  TOPOLOGY_AR                 — sample_step is not None (autoregressive sampling)
  TOPOLOGY_CONDITIONAL_SCORE  — decode_step is None or ConditionalDecodeStep (teacher-forced)
  TOPOLOGY_UNCONDITIONAL      — decode_step is UnconditionalDecodeStep (unconditional scoring)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.protocols import ModelProtocol
    from prxteinmpnn.types.stages import StageSet

from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.encodings import EncoderOutput
from prxteinmpnn.types.stages import UnconditionalDecodeStep
from prxteinmpnn.inference.sample_autoregressive import SampleResult

# Topology constants (used at call time, not traced)
TOPOLOGY_AR = "ar"
TOPOLOGY_CONDITIONAL_SCORE = "conditional_score"
TOPOLOGY_UNCONDITIONAL = "unconditional"


def infer_topology(stage_set: StageSet) -> str:
    """Infer decode topology from stage_set configuration.

    Returns one of:
      - TOPOLOGY_AR: sample_step is not None (autoregressive sampling)
      - TOPOLOGY_UNCONDITIONAL: decode_step is UnconditionalDecodeStep
      - TOPOLOGY_CONDITIONAL_SCORE: all else (conditional or fallback to model.decoder)
    """
    if stage_set.sample_step is not None:
        return TOPOLOGY_AR
    if isinstance(stage_set.decode_step, UnconditionalDecodeStep):
        return TOPOLOGY_UNCONDITIONAL
    return TOPOLOGY_CONDITIONAL_SCORE


def decode(
    model: ModelProtocol,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    cond,
    wave,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Union[Logits, SampleResult]:
    """Unified decode driver.

    Dispatches to the appropriate kernel path (AR, unconditional, or conditional scoring)
    based on stage_set configuration. All kernel logic is parameterized by stages.

    Args:
        model: ModelProtocol instance
        key: PRNG key for decoding randomness
        enc: EncoderOutput from prior encode step
        cond: ConditioningBundle
        wave: WaveScheduleBundle or None (None for unconditional)
        config: InferenceConfig
        stage_set: StageSet with decode_step, logit_transform, ar_logit_transform, sample_step

    Returns:
        Logits (for scoring paths) or SampleResult (for AR path)
    """
    topology = infer_topology(stage_set)

    if topology == TOPOLOGY_AR:
        return decode_ar(model, key, enc, cond, wave, config, stage_set)
    elif topology == TOPOLOGY_UNCONDITIONAL:
        return _decode_unconditional(model, key, enc, cond, config, stage_set)
    else:
        return _decode_conditional(model, key, enc, cond, config, stage_set)


# ---------------------------------------------------------------------------
# Conditional Scoring Path (teacher-forced, vmap over states)
# ---------------------------------------------------------------------------

def _decode_conditional(
    model: ModelProtocol,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    cond,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Conditional scoring kernel.

    Performs teacher-forced decoding over S states, projects to logits,
    and fuses via stage_set.logit_transform.

    Returns:
        Logits of shape (L, V)
    """
    S = enc.node_features.shape[0]  # First dim is state dimension

    def decode_one(nb, eb, nei, mk, arm, oh):
        if stage_set.decode_step is not None:
            return stage_set.decode_step(nb, eb, nei, mk, arm, oh, key=key, inference=config.inference)
        return model.decoder.call_conditional(
            nb, eb, nei, mk, arm, oh, model.w_s_embed.weight,
            inference=config.inference, key=key
        )

    # Broadcast sequence one-hot to all states
    seq_oh_stack = jnp.broadcast_to(cond.sequence_oh[None, ...], (S, *cond.sequence_oh.shape))

    # Decode over states: (S, L, H)
    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask, cond.ar_mask, seq_oh_stack
    )

    # Project to logits: (S, L, V) -> (S, L, 21)
    logits_stack = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

    # Fuse across states
    return stage_set.logit_transform(logits_stack, bias=cond.bias)


# ---------------------------------------------------------------------------
# Unconditional Scoring Path (no sequence conditioning, vmap over states)
# ---------------------------------------------------------------------------

def _decode_unconditional(
    model: ModelProtocol,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    cond,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Unconditional scoring kernel.

    Decodes without sequence context (no ar_mask, no seq_oh).
    vmaps over S states and fuses via stage_set.logit_transform.

    Returns:
        Logits of shape (L, V)
    """
    # Unconditional path does not use wave schedule
    assert isinstance(stage_set.decode_step, UnconditionalDecodeStep), \
        f"Unconditional decoding requires UnconditionalDecodeStep, got {type(stage_set.decode_step)}"

    def decode_one(nb, eb, nei, mk):
        if stage_set.decode_step is not None:
            return stage_set.decode_step(nb, eb, nei, mk, key=key, inference=config.inference)
        # For unconditional, no sequence conditioning
        return model.decoder(nb, eb, nei, mk, key=key, inference=config.inference)

    # Decode over states: (S, L, H)
    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask
    )

    # Project to logits: (S, L, V) -> (S, L, 21)
    logits_stack = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

    # Fuse across states
    return stage_set.logit_transform(logits_stack, bias=cond.bias)


# ---------------------------------------------------------------------------
# Autoregressive Sampling Path (scan over waves)
# ---------------------------------------------------------------------------

def decode_ar(
    model: ModelProtocol,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    cond,
    wave,
    config: InferenceConfig,
    stage_set: StageSet,
) -> SampleResult:
    """Autoregressive sampling kernel.

    Encodes once and iterates through wave schedule, sampling at each position
    and updating the sequence. Returns both the sampled sequence and the logits.

    Args:
        model: ModelProtocol
        key: PRNG key for sampling
        enc: EncoderOutput from prior encode
        cond: ConditioningBundle with ar_mask, tie_group_map, fixed_mask, temperature, bias
        wave: WaveScheduleBundle with group_positions and derived n_waves from shape
        config: InferenceConfig
        stage_set: StageSet with ar_logit_transform, decode_step, logit_transform, sample_step

    Returns:
        SampleResult(sequence, logits) where sequence is (L,) and logits is (L, 21)
    """
    geo_mask = enc.mask  # Assuming enc.mask carries geometry mask
    L = enc.node_features.shape[1]  # Second dim is sequence length
    S = enc.node_features.shape[0]  # First dim is state dimension
    n_waves = wave.group_ids.shape[0]  # First dim of group_ids is W (num waves)

    # Decoding Loop
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
                if stage_set.decode_step is not None:
                    return stage_set.decode_step(n, e, idx, m, arm, seq_oh, key=key, inference=config.inference)
                return model.decoder.call_conditional(
                    n, e, idx, m, arm, seq_oh, model.w_s_embed.weight,
                    key=key, inference=config.inference
                )

            # decoded: (S, L, H)
            decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0))(
                enc.node_features, enc.edge_features, enc.neighbor_indices, geo_mask, cond.ar_mask
            )

            # Project to logits: (S, L, 21)
            logits = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

            # Combine logits across states: (L, 21)
            if stage_set.ar_logit_transform is not None:
                # Apply ar_logit_transform per position: (S, L, 21) -> (L, 21) via vmap
                ar_fused = jax.vmap(stage_set.ar_logit_transform, in_axes=1, out_axes=0)(logits)
                combined_logits = ar_fused + cond.bias if cond.bias is not None else ar_fused
            else:
                combined_logits = stage_set.logit_transform(logits, bias=cond.bias)

            # Logit averaging for the group
            mask = (cond.tie_group_map[0] == group_id)
            group_logits = jnp.where(mask[:, None], combined_logits, -jnp.inf)
            n_tied = jnp.sum(mask)
            # Use logsumexp for averaging in log-space: (L, 21) -> (21,)
            avg_logits = jax.scipy.special.logsumexp(group_logits, axis=0) - jnp.log(jnp.maximum(n_tied, 1))
            # Ensure shape is (21,) not (L, 21) by explicit reshape to avoid JAX tracing issues
            step_logits = avg_logits.reshape((21,))

            # Sample
            subkey = jax.random.fold_in(key, group_id)
            sampled = jax.random.categorical(subkey, step_logits / cond.temperature)

            # Update all positions in the group
            is_group_fixed = jnp.any(cond.fixed_mask.astype(jnp.bool_) & mask)
            group_fixed_token = jnp.max(jnp.where(cond.fixed_mask.astype(jnp.bool_) & mask, cond.fixed_tokens, 0))
            final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)

            new_seq = jnp.where(mask, final_token, seq)
            return new_seq, step_logits

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
        jnp.arange(n_waves)
    )

    # Map logits_stack (W, 21) back to (L, 21)
    def scatter_logits(logits_final, i):
        pos = wave.group_positions[i, 0, 0]
        group_id = cond.tie_group_map[0, pos]
        mask = (cond.tie_group_map[0] == group_id)
        step_logits = logits_stack[i]
        new_logits_final = jnp.where(mask[:, None], step_logits, logits_final)
        return new_logits_final, None

    logits_init = jnp.zeros((L, 21))
    logits_final, _ = jax.lax.scan(scatter_logits, logits_init, jnp.arange(n_waves))

    return SampleResult(
        sequence=final_seq,
        logits=logits_final
    )


__all__ = [
    "decode",
    "decode_ar",
    "infer_topology",
    "TOPOLOGY_AR",
    "TOPOLOGY_CONDITIONAL_SCORE",
    "TOPOLOGY_UNCONDITIONAL",
]
