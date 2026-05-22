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
    from prxteinmpnn.types.bundles import ConditioningBundle, WaveScheduleBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.protocols import ModelProtocol
    from prxteinmpnn.types.stages import StageSet

from prxteinmpnn.inference.sample_autoregressive import SampleResult
from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.encodings import EncoderOutput
from prxteinmpnn.types.stages import UnconditionalDecodeStep

# Topology constants (used at call time, not traced)
TOPOLOGY_AR = "ar"
TOPOLOGY_CONDITIONAL_SCORE = "conditional_score"
TOPOLOGY_UNCONDITIONAL = "unconditional"


def infer_topology(stage_set: StageSet) -> str:
    """Infer decode topology from StageSet slot occupancy.

    Examines stage_set fields to determine which decoding path to use:
    AR (sampling), unconditional (scoring without sequence), or conditional
    (teacher-forced scoring with sequence context).

    Parameters
    ----------
    stage_set : StageSet
        StageSet configuration with decode_step and sample_step.

    Returns
    -------
    str
        One of TOPOLOGY_AR, TOPOLOGY_UNCONDITIONAL, or TOPOLOGY_CONDITIONAL_SCORE.
        - TOPOLOGY_AR: sample_step is not None (autoregressive sampling)
        - TOPOLOGY_UNCONDITIONAL: decode_step is UnconditionalDecodeStep
        - TOPOLOGY_CONDITIONAL_SCORE: all else (conditional or fallback to model.decoder)

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
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
    cond: ConditioningBundle,
    wave: WaveScheduleBundle | None,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Union[Logits, SampleResult]:
    """Unified decode driver dispatching by StageSet topology.

    Routes to autoregressive sampling, unconditional scoring, or conditional
    (teacher-forced) scoring based on stage_set configuration. All kernel logic
    is parameterized by stages.

    Parameters
    ----------
    model : ModelProtocol
        Model instance with decoder and w_out linear layer.
    key : PRNGKeyArray
        PRNG key for decoding randomness (dropout, sampling).
    enc : EncoderOutput
        Encoder output from prior encode step.
    cond : ConditioningBundle
        Conditioning data: sequence_oh, ar_mask, bias, fixed positions, tie_group_map.
    wave : WaveScheduleBundle or None
        Wave schedule for AR sampling. None for scoring paths.
    config : InferenceConfig
        Inference configuration (temperature, inference mode flag).
    stage_set : StageSet
        StageSet with decode_step, logit_transform, ar_logit_transform, sample_step,
        tie_group_fuse.

    Returns
    -------
    Union[Logits, SampleResult]
        Logits of shape (L, 21) for scoring paths (unconditional, conditional).
        SampleResult with sequence (L,) and logits (L, 21) for AR path.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """
    topology = infer_topology(stage_set)

    if topology == TOPOLOGY_AR:
        return decode_ar(model, key, enc, cond, wave, config, stage_set)
    if topology == TOPOLOGY_UNCONDITIONAL:
        return _decode_unconditional(model, key, enc, cond, config, stage_set)
    return _decode_conditional(model, key, enc, cond, config, stage_set)


# ---------------------------------------------------------------------------
# Conditional Scoring Path (teacher-forced, vmap over states)
# ---------------------------------------------------------------------------

def _decode_conditional(
    model: ModelProtocol,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    cond: "ConditioningBundle",
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Conditional scoring kernel (teacher-forced decoding).

    Performs teacher-forced decoding with sequence context, vmapping over
    S states, projecting to logits, and fusing via stage_set.logit_transform.

    Parameters
    ----------
    model : ModelProtocol
        Model instance.
    key : PRNGKeyArray
        PRNG key.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    cond : ConditioningBundle
        ConditioningBundle with sequence_oh (1, L, 21), ar_mask, bias (L, 21).
    config : InferenceConfig
        Inference config.
    stage_set : StageSet
        StageSet with decode_step and logit_transform.

    Returns
    -------
    Logits
        Fused logits. Shape: (L, 21).

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """
    S = enc.node_features.shape[0]  # First dim is state dimension

    def decode_one(node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh):
        if stage_set.decode_step is not None:
            return stage_set.decode_step(node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh, key=key, inference=config.inference)
        return model.decoder.call_conditional(
            node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh, model.w_s_embed.weight,
            inference=config.inference, key=key,
        )

    # Broadcast sequence one-hot to all states
    seq_oh_stack = jnp.broadcast_to(cond.sequence_oh[None, ...], (S, *cond.sequence_oh.shape))

    # Decode over states: (S, L, H)
    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask, cond.ar_mask, seq_oh_stack,
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
    cond: "ConditioningBundle",
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Unconditional scoring kernel (no sequence context).

    Decodes without sequence conditioning, vmapping over S states,
    and fuses via stage_set.logit_transform.

    Parameters
    ----------
    model : ModelProtocol
        Model instance.
    key : PRNGKeyArray
        PRNG key.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    cond : ConditioningBundle
        ConditioningBundle (used for bias only; sequence/ar_mask ignored).
    config : InferenceConfig
        Inference config.
    stage_set : StageSet
        StageSet with decode_step (UnconditionalDecodeStep) and logit_transform.

    Returns
    -------
    Logits
        Fused logits. Shape: (L, 21).

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """
    # Unconditional path does not use wave schedule
    assert isinstance(stage_set.decode_step, UnconditionalDecodeStep), \
        f"Unconditional decoding requires UnconditionalDecodeStep, got {type(stage_set.decode_step)}"

    def decode_one(node_features, edge_features, neighbor_indices, mask):
        if stage_set.decode_step is not None:
            return stage_set.decode_step(node_features, edge_features, neighbor_indices, mask, key=key, inference=config.inference)
        # For unconditional, no sequence conditioning
        return model.decoder(node_features, edge_features, neighbor_indices, mask, key=key, inference=config.inference)

    # Decode over states: (S, L, H)
    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask,
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
    """Autoregressive sampling kernel scanning through wave schedule.

    Encodes once and iterates through wave schedule via lax.scan. At each wave,
    identifies the tied position group, scores all positions, fuses logits across
    the group, samples, applies fixed positions and tie_group_fuse if configured,
    and updates the sequence. Returns sampled sequence and per-position logits.

    Parameters
    ----------
    model : ModelProtocol
        Model instance.
    key : PRNGKeyArray
        PRNG key for sampling randomness.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    cond : ConditioningBundle
        ConditioningBundle with ar_mask (L,), tie_group_map (1, L),
        fixed_mask (L,), fixed_tokens (L,), temperature, bias (L, 21).
    wave : WaveScheduleBundle
        WaveScheduleBundle with group_positions (W, S_pos, 1) and group_ids (W,).
    config : InferenceConfig
        Inference config.
    stage_set : StageSet
        StageSet with ar_logit_transform, decode_step, logit_transform,
        sample_step, and tie_group_fuse.

    Returns
    -------
    SampleResult
        SampleResult(sequence, logits). Shapes: sequence (L,), logits (L, 21).

    Notes
    -----
    Scans through wave schedule, per wave step:
      1. Identify tied position group from decoding order.
      2. Check if group is first occurrence in order (avoid re-sampling).
      3. If first: vmap decode, fuse logits per position (ar_logit_transform),
         average across tied positions (tie_group_fuse), sample, apply fixed mask.
      4. If not first: skip (no update, zero logits).
    Logits stored bias-free for parity comparison; sampling logits include bias.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187

    .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
       sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
       https://doi.org/10.1038/s41592-025-02626-1
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
            def decode_one(node_features, edge_features, neighbor_indices, mask, ar_mask):
                if stage_set.decode_step is not None:
                    return stage_set.decode_step(node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh, key=key, inference=config.inference)
                return model.decoder.call_conditional(
                    node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh, model.w_s_embed.weight,
                    key=key, inference=config.inference,
                )

            # decoded: (S, L, H)
            decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0))(
                enc.node_features, enc.edge_features, enc.neighbor_indices, geo_mask, cond.ar_mask,
            )

            # Project to logits: (S, L, 21)
            logits = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

            # Fuse per-position logits across states with and without bias:
            # - stored_logits: bias-free, for log_prob parity comparison
            # - sampling_logits: bias-applied, for categorical sampling
            if stage_set.ar_logit_transform is not None:
                # ar_logit_transform signature: (logits: (S, V), bias: (V,)) -> (V,)
                # Vmap over positions to fuse each position's logits with corresponding bias
                # logits: (S, L, 21), need to vmap axis 1 (L positions)
                # cond.bias: (L, 21) per position
                # For stored logits: use zero bias
                zeros_bias = jnp.zeros_like(cond.bias)  # (L, 21)
                stored_logits = jax.vmap(stage_set.ar_logit_transform, in_axes=(1, 0), out_axes=0)(
                    logits, zeros_bias,
                )  # outputs (L, 21)
                # For sampling logits: use actual bias
                sampling_logits = jax.vmap(stage_set.ar_logit_transform, in_axes=(1, 0), out_axes=0)(
                    logits, cond.bias,
                )  # outputs (L, 21)
            else:
                # Fallback to logit_transform with explicit bias handling
                stored_logits = stage_set.logit_transform(logits, bias=jnp.zeros_like(cond.bias))
                sampling_logits = stage_set.logit_transform(logits, bias=cond.bias)

            # Logit averaging for the group (tied positions)
            mask = (cond.tie_group_map[0] == group_id)

            # Fuse stored logits (bias-free) across tied positions
            if stage_set.tie_group_fuse is not None:
                step_logits = stage_set.tie_group_fuse(stored_logits, mask).reshape((21,))
            else:
                stored_group = jnp.where(mask[:, None], stored_logits, -jnp.inf)
                n_tied = jnp.sum(mask)
                avg_stored = jax.scipy.special.logsumexp(stored_group, axis=0) - jnp.log(jnp.maximum(n_tied, 1))
                step_logits = avg_stored.reshape((21,))

            # Fuse sampling logits (bias-applied) across tied positions
            if stage_set.tie_group_fuse is not None:
                avg_sampling = stage_set.tie_group_fuse(sampling_logits, mask).reshape((21,))
            else:
                sampling_group = jnp.where(mask[:, None], sampling_logits, -jnp.inf)
                n_tied = jnp.sum(mask)
                avg_sampling = jax.scipy.special.logsumexp(sampling_group, axis=0) - jnp.log(jnp.maximum(n_tied, 1))
                avg_sampling = avg_sampling.reshape((21,))

            # Sample from bias-applied logits
            subkey = jax.random.fold_in(key, group_id)
            sampled = jax.random.categorical(subkey, avg_sampling / cond.temperature)

            # Update all positions in the group
            is_group_fixed = jnp.any(cond.fixed_mask.astype(jnp.bool_) & mask)
            group_fixed_token = jnp.max(jnp.where(cond.fixed_mask.astype(jnp.bool_) & mask, cond.fixed_tokens, 0))
            final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)

            new_seq = jnp.where(mask, final_token, seq)
            return new_seq, step_logits

        def no_sample(seq):
            # No update for tied positions that have already been sampled
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
        jnp.arange(n_waves),
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
        logits=logits_final,
    )


__all__ = [
    "TOPOLOGY_AR",
    "TOPOLOGY_CONDITIONAL_SCORE",
    "TOPOLOGY_UNCONDITIONAL",
    "decode",
    "decode_ar",
    "infer_topology",
]
