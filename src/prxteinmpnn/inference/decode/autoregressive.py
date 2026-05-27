"""AutoregressiveDecode mode class (Task 9).

Autoregressive sampling kernel that iterates through a wave schedule (sequence of
tied position groups), carrying the evolving sequence through the waves via a
ScanIterator. The wave-axis carry is reified as a CarryShape metadata struct
(Risk D-10 mitigation).

After the wave scan completes, a post-hoc scatter scan maps per-wave logits back
to per-position logits (Risk D-11 mitigation) — this stays outside the iterator.

Key invariants:
- The wave_iterator field is always JaxScanIterator (structural invariant; no
  user-facing W-axis strategy knob per Risk D-3).
- The state_iterator is injected at factory time, controlling S-axis parallelism.
- CarryShape is metadata-only; the actual init-array is materialized inside
  __call__ from the metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.inference.decode._kernel import (
  _decode_one_step,
  _project_logits,
)
from prxteinmpnn.inference.sample_autoregressive import SampleResult
from prxteinmpnn.tiling.carry_shape import CarryShape
from prxteinmpnn.tiling.iterator import MapIterator, ScanIterator
from prxteinmpnn.types.bundles import EncoderOutput, InferenceBundle, WaveScheduleBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.stages import StageSet

# Type alias for decoding order function
DecodingOrderFn = Callable[[WaveScheduleBundle], Any]


class AutoregressiveDecode(eqx.Module):
  """Autoregressive decode mode: carry-bearing wave iteration with post-hoc scatter.

  Combines a MapIterator (for state-axis stateless iteration) and a ScanIterator
  (for wave-axis carry-bearing iteration). The wave carry is the evolving sequence,
  reified as a CarryShape metadata struct to defer materialization to __call__.

  After the main wave scan, a post-hoc scatter scan (outside the iterator) maps
  per-wave logits back to per-position logits, preserving the two-scan structure
  from driver.py:decode_ar (Risk D-11).

  Parameters
  ----------
  model : Any
      Model instance with decoder and w_out attributes.
  decoding_order_fn : Callable
      Function (wave_schedule) -> decoding_order. Determines the order in which
      positions are decoded within the wave schedule.
  state_iterator : MapIterator
      Iterator for the S axis (VmapIterator, SafeMapIterator, etc.).
      Injected at factory time; determines parallelism strategy.
  wave_iterator : ScanIterator
      Iterator for the W axis (always JaxScanIterator; field exists for
      type-symmetry only; structural invariant per Risk D-3).
  wave_carry : CarryShape
      Metadata for the wave-axis carry (name="sequence", shape=(L,),
      dtype=jnp.int32). The actual init-array is materialized inside __call__.

  Attributes
  ----------
  model : Any = eqx.field(static=True)
      Model is static (always traced same way).
  decoding_order_fn : Callable = eqx.field(static=True)
      Decoding order function is static.
  state_iterator : MapIterator
      State axis iterator (injected at factory time).
  wave_iterator : ScanIterator
      Wave axis iterator (always JaxScanIterator; structural invariant).
  wave_carry : CarryShape = eqx.field(static=True)
      Metadata for carry shape (name, shape, dtype; no value).

  Notes
  -----
  This class implements Pattern 5 (injection): both state_iterator and
  wave_iterator are injected dependencies. Different combinations allow
  flexible composition of S-axis and W-axis iteration strategies without
  code duplication.

  The post-hoc scatter scan is performed inside __call__, not inside the
  wave_iterator. This preserves the two-scan structure from driver.py:decode_ar:
  1. Wave scan (inside iterator): carries sequence, outputs per-wave logits.
  2. Scatter scan (post-hoc, outside iterator): maps per-wave to per-position logits.
  """

  model: Any = eqx.field(static=True)
  decoding_order_fn: DecodingOrderFn = eqx.field(static=True)
  state_iterator: MapIterator
  wave_iterator: ScanIterator
  wave_carry: CarryShape = eqx.field(static=True)

  def __call__(
    self,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
  ) -> SampleResult:
    """Autoregressive decode: carry sequence through wave scan, then scatter logits.

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key for sampling randomness.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    bundle : InferenceBundle
        Inference bundle with conditioning (ar_mask, tie_group_map, fixed_mask,
        fixed_tokens, temperature, bias) and wave schedule.
    config : InferenceConfig
        Inference configuration.
    stage_set : StageSet
        Pipeline stages with ar_logit_transform, decode_step,
        logit_transform, tie_group_fuse.

    Returns
    -------
    SampleResult
        Result with sequence (L,) and logits (L, 21).

    Notes
    -----
    Workflow:
    1. Materialize init sequence from wave_carry metadata.
    2. Compute decoding order via decoding_order_fn.
    3. Build scan_body that:
       a. For each state, call _decode_one_step via state_iterator.
       b. Project to logits.
       c. Fuse per-position logits via ar_logit_transform.
       d. Average across tied positions via tie_group_fuse.
       e. Sample from averaged logits.
       f. Update sequence for sampled positions.
       g. Return (new_sequence, per_wave_logits).
    4. Call wave_iterator(scan_body, init, jnp.arange(n_waves)).
    5. Post-hoc scatter: map (n_waves, V) logits to (L, V) via second scan.
    6. Return SampleResult(final_sequence, logits).
    """
    L = enc.node_features.shape[1]
    S = enc.node_features.shape[0]
    cond = bundle.conditioning
    wave = bundle.wave
    n_waves = wave.group_ids.shape[0]

    # 1. Materialize init sequence from metadata
    init_sequence = self.wave_carry.materialize()

    # Initialize with fixed positions
    init_sequence = jnp.where(
      cond.fixed_mask > 0.5,
      cond.fixed_tokens,
      init_sequence,
    ).astype(jnp.int32)

    # 2. Build step function that processes one wave at a time
    def step_fn(sequence: jnp.ndarray, wave_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
      """Process one wave: decode, fuse, sample, update sequence.

      Parameters
      ----------
      sequence : array (L,)
          Current sequence (evolving carry).
      wave_idx : scalar int
          Index into the wave schedule.

      Returns
      -------
      (new_sequence, step_logits) : (array (L,), array (21,))
          Updated sequence and per-wave logits.
      """
      # Identify tied position group for this wave
      pos = wave.group_positions[wave_idx, 0, 0]
      group_id = cond.tie_group_map[0, pos]

      # Check if this is the first time we encounter this group
      # Note: we only check the first position in the group (group_positions[wave_idx, 0, 0])
      # to determine if this group should be sampled
      tie_group_at_order = cond.tie_group_map[0, wave.group_positions[:, 0, 0]]
      first_occurrence_idx = jnp.argmax(tie_group_at_order == group_id)
      is_first = first_occurrence_idx == wave_idx

      def do_sample(seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Decode, fuse, sample, and update sequence."""
        # One-hot encode sequence for all S states
        seq_oh = jax.nn.one_hot(seq, 21)  # (L, 21)
        seq_oh_stack = jnp.broadcast_to(
          seq_oh[None, ...],
          (S, L, 21),
        )  # (S, L, 21)

        # Per-state inputs
        per_state_inputs = (
          enc.node_features,
          enc.edge_features,
          enc.neighbor_indices,
          enc.mask,
          cond.ar_mask,
          seq_oh_stack,
        )

        # Per-state decode closure
        def per_state_fn(inputs):
          node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh = inputs
          return _decode_one_step(
            model=self.model,
            node_features=node_features,
            edge_features=edge_features,
            neighbor_indices=neighbor_indices,
            mask=mask,
            ar_mask=ar_mask,
            sequence_oh=seq_oh,
            key=key,
            inference=config.inference,
            decode_step=stage_set.decode_step,
          )

        # Apply state_iterator to iterate over S axis
        decoded = self.state_iterator(per_state_fn, per_state_inputs, in_axes=0)

        # Project to logits: (S, L, H) -> (S, L, 21)
        logits = _project_logits(self.model, decoded)

        # Fuse per-position logits across states with and without bias
        # For stored logits: bias-free
        zeros_bias = jnp.zeros_like(cond.bias)  # (L, 21)
        if stage_set.ar_logit_transform is not None:
          stored_logits = jax.vmap(
            stage_set.ar_logit_transform,
            in_axes=(1, 0),
            out_axes=0,
          )(logits, zeros_bias)  # (L, 21)
          sampling_logits = jax.vmap(
            stage_set.ar_logit_transform,
            in_axes=(1, 0),
            out_axes=0,
          )(logits, cond.bias)  # (L, 21)
        else:
          stored_logits = stage_set.logit_transform(
            logits,
            bias=zeros_bias,
          )
          sampling_logits = stage_set.logit_transform(
            logits,
            bias=cond.bias,
          )

        # Logit averaging for the group (tied positions)
        mask_group = cond.tie_group_map[0] == group_id

        # Fuse stored logits (bias-free) across tied positions
        if stage_set.tie_group_fuse is not None:
          step_logits = stage_set.tie_group_fuse(
            stored_logits,
            mask_group,
          ).reshape((21,))
        else:
          stored_group = jnp.where(
            mask_group[:, None],
            stored_logits,
            -jnp.inf,
          )
          n_tied = jnp.sum(mask_group)
          avg_stored = jax.scipy.special.logsumexp(
            stored_group,
            axis=0,
          ) - jnp.log(jnp.maximum(n_tied, 1))
          step_logits = avg_stored.reshape((21,))

        # Fuse sampling logits (bias-applied) across tied positions
        if stage_set.tie_group_fuse is not None:
          avg_sampling = stage_set.tie_group_fuse(
            sampling_logits,
            mask_group,
          ).reshape((21,))
        else:
          sampling_group = jnp.where(
            mask_group[:, None],
            sampling_logits,
            -jnp.inf,
          )
          n_tied = jnp.sum(mask_group)
          avg_sampling = jax.scipy.special.logsumexp(
            sampling_group,
            axis=0,
          ) - jnp.log(jnp.maximum(n_tied, 1))
          avg_sampling = avg_sampling.reshape((21,))

        # Sample from bias-applied logits
        subkey = jax.random.fold_in(key, group_id)
        sampled = jax.random.categorical(
          subkey,
          avg_sampling / cond.temperature,
        )

        # Update all positions in the group
        is_group_fixed = jnp.any(cond.fixed_mask.astype(jnp.bool_) & mask_group)
        group_fixed_token = jnp.max(
          jnp.where(
            cond.fixed_mask.astype(jnp.bool_) & mask_group,
            cond.fixed_tokens,
            0,
          ),
        )
        final_token = jnp.where(
          is_group_fixed,
          group_fixed_token,
          sampled,
        ).astype(jnp.int32)

        new_seq = jnp.where(mask_group, final_token, seq)
        return new_seq, step_logits

      def no_sample(seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Skip update for positions already sampled."""
        return seq, jnp.zeros((21,))

      # Conditionally sample or skip based on first_occurrence check
      return jax.lax.cond(is_first, do_sample, no_sample, sequence)

    # 3. Run wave scan
    final_seq, logits_stack = self.wave_iterator(
      step_fn,
      init_sequence,
      jnp.arange(n_waves),
    )

    # 4. Post-hoc scatter: map per-wave logits to per-position logits
    def scatter_logits(
      logits_final: jnp.ndarray,
      wave_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
      """Scatter one wave's logits to its positions."""
      pos = wave.group_positions[wave_idx, 0, 0]
      group_id = cond.tie_group_map[0, pos]
      mask_group = cond.tie_group_map[0] == group_id
      step_logits = logits_stack[wave_idx]  # (21,)
      new_logits_final = jnp.where(
        mask_group[:, None],
        step_logits,
        logits_final,
      )
      return new_logits_final, None

    logits_init = jnp.zeros((L, 21))
    logits_final, _ = jax.lax.scan(
      scatter_logits,
      logits_init,
      jnp.arange(n_waves),
    )

    # 5. Return SampleResult
    return SampleResult(
      sequence=final_seq,
      logits=logits_final,
    )


__all__ = ["AutoregressiveDecode"]
