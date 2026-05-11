"""Host-level dispatcher for multi-structure SamplingInputs scoring."""

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from prxteinmpnn.model_inputs import SamplingInputs
    from prxteinmpnn.model.mpnn import PrxteinMPNN


@dataclasses.dataclass(frozen=True)
class PayloadDispatcher:
    """Host-level dispatcher for per-structure SamplingInputs iteration.

    Iterates over a list[SamplingInputs] (n_structures axis).
    Within each structure, dispatches to the model method directly.
    prng_key splitting is pre-computed before the host loop to guarantee
    plan-independence (identical keys whether caller uses vmap or safe_map externally).

    Does NOT use safe_map internally — the structure loop is a plain Python for-loop
    since SamplingInputs is a heterogeneous type (variable n_states per structure).
    The model method (score_unconditional_from_payload etc.) handles internal vmap.
    """

    def score_unconditional(
        self,
        model: Any,  # PrxteinMPNN, but using Any to avoid circular imports
        prng_key,  # JAX PRNG key
        stack_list,  # list[MultistateStackPayload]
        *,
        tie_group_map,
        multi_state_strategy_idx: int,
        state_weights,
        state_mapping,
        inference: bool = True,
        logit_transform_fn: Callable | None = None,
        encoder_state_fn: Callable | None = None,
    ):
        """Score each MultistateStackPayload in stack_list unconditionally.

        Args:
            model: PrxteinMPNN instance.
            prng_key: JAX PRNG key for the entire structure batch.
            stack_list: list of MultistateStackPayload, one per structure.
            tie_group_map: forwarded to model.score_unconditional_from_payload.
            multi_state_strategy_idx: forwarded to model method.
            state_weights: forwarded to model method.
            state_mapping: forwarded to model method.
            inference: forwarded to model method.
            logit_transform_fn: forwarded to model method.
            encoder_state_fn: forwarded to model method.

        Returns:
            list of Logits, one per structure (list of arrays).
        """
        # Guard for empty list
        if not stack_list:
            return []

        # Pre-split PRNG keys for determinism
        n = len(stack_list)
        structure_keys = jax.random.split(prng_key, n)  # shape (n, 2)

        results = []
        for i, stack in enumerate(stack_list):
            logits = model.score_unconditional_from_payload(
                structure_keys[i],
                stack,
                tie_group_map=tie_group_map,
                multi_state_strategy_idx=multi_state_strategy_idx,
                state_weights=state_weights,
                state_mapping=state_mapping,
                inference=inference,
                logit_transform_fn=logit_transform_fn,
                encoder_state_fn=encoder_state_fn,
            )
            results.append(logits)

        return results

    def score_conditional(
        self,
        model: Any,  # PrxteinMPNN
        prng_key,  # JAX PRNG key
        stack_list,  # list[MultistateStackPayload]
        seq_oh_stack_list,  # list of one-hot sequence arrays
        ar_mask_stack_list,  # list of AR mask arrays
        *,
        tie_group_map,
        multi_state_strategy_idx: int,
        state_weights,
        state_mapping,
        bias_flat_stack_list=None,  # optional list of bias arrays, one per structure
        inference: bool = True,
        logit_transform_fn: Callable | None = None,
        encoder_state_fn: Callable | None = None,
    ):
        """Score each MultistateStackPayload in stack_list conditionally.

        Args:
            model: PrxteinMPNN instance.
            prng_key: JAX PRNG key for the entire structure batch.
            stack_list: list of MultistateStackPayload, one per structure.
            seq_oh_stack_list: list of one-hot sequence arrays, aligned with stack_list.
            ar_mask_stack_list: list of AR mask arrays, aligned with stack_list.
            tie_group_map: forwarded to model.score_conditional_from_payload.
            multi_state_strategy_idx: forwarded to model method.
            state_weights: forwarded to model method.
            state_mapping: forwarded to model method.
            bias_flat_stack_list: optional list of bias arrays, one per structure. If None, bias_flat=None is passed to model method.
            inference: forwarded to model method.
            logit_transform_fn: forwarded to model method.
            encoder_state_fn: forwarded to model method.

        Returns:
            list of Logits, one per structure (list of arrays).
        """
        # Guard for empty list
        if not stack_list:
            return []

        # Validate aligned list lengths
        assert len(stack_list) == len(seq_oh_stack_list) == len(ar_mask_stack_list), \
            f"List lengths must match: {len(stack_list)}, {len(seq_oh_stack_list)}, {len(ar_mask_stack_list)}"

        # Validate bias_flat_stack_list length if provided
        if bias_flat_stack_list is not None:
            assert len(bias_flat_stack_list) == len(stack_list), \
                f"bias_flat_stack_list length {len(bias_flat_stack_list)} != stack_list length {len(stack_list)}"

        # Pre-split PRNG keys for determinism
        n = len(stack_list)
        structure_keys = jax.random.split(prng_key, n)  # shape (n, 2)

        results = []
        for i, stack in enumerate(stack_list):
            bias_flat = None if bias_flat_stack_list is None else bias_flat_stack_list[i]
            logits = model.score_conditional_from_payload(
                structure_keys[i],
                stack,
                seq_oh_stack_list[i],
                ar_mask_stack_list[i],
                tie_group_map=tie_group_map,
                multi_state_strategy_idx=multi_state_strategy_idx,
                state_weights=state_weights,
                state_mapping=state_mapping,
                bias_flat=bias_flat,
                inference=inference,
                logit_transform_fn=logit_transform_fn,
                encoder_state_fn=encoder_state_fn,
            )
            results.append(logits)

        return results


__all__ = ["PayloadDispatcher"]
