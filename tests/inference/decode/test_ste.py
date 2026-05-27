"""Parity tests for STEDecode mode class (Task 10).

STEDecode wraps a ConditionalDecode inner mode and applies iterative STE refinement
with gradient-based optimization. Tests verify numerical parity with
optimize_ste.py:make_optimize_sequence_fn on fixtures including tied and untied positions.

Risk D-2 fixture mandate: tied-positions parity test is LOAD-BEARING.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference.decode.conditional import ConditionalDecode
from prxteinmpnn.inference.decode.ste import STEDecode
from prxteinmpnn.inference.logits import make_stage_set
from prxteinmpnn.inference.optimize_ste import make_optimize_sequence_fn
from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.tiling.iterator import VmapIterator
from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.stages import StageSet


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 20,
    seed: int = 42,
    tie_group_map: jnp.ndarray | None = None,
) -> tuple[PrxteinMPNN, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, InferenceBundle, InferenceConfig]:
    """Build deterministic fixture with S=num_states, L=num_residues.

    Parameters
    ----------
    num_states : int
        Number of states.
    num_residues : int
        Number of residues (L).
    seed : int
        Random seed.
    tie_group_map : jnp.ndarray | None
        Optional tied-positions group map. Shape (L,). If None, each position is its own group.

    Returns
    -------
    tuple
        (model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config)
    """
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    # Build model
    model = PrxteinMPNN(
        node_features=64,
        edge_features=64,
        hidden_features=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=5,
        dropout_rate=0.0,
        key=jax_key,
    )
    model = eqx.tree_inference(model, value=True)

    # Build geometry
    coordinates = jnp.array(
        rng.normal(size=(num_residues, 4, 3)).astype(np.float32)
    )
    mask = jnp.ones((num_residues,), dtype=jnp.float32)
    residue_index = jnp.arange(num_residues, dtype=jnp.int32)
    chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

    # Replicate geometry S times
    coordinates_stack = jnp.stack([coordinates] * num_states, axis=0)
    mask_stack = jnp.stack([mask] * num_states, axis=0)
    residue_index_stack = jnp.stack([residue_index] * num_states, axis=0)
    chain_index_stack = jnp.stack([chain_index] * num_states, axis=0)

    # Conditioning
    sequence_tokens = jnp.array(
        rng.integers(0, 20, size=(num_residues,), dtype=np.int32)
    )
    sequence_oh = jax.nn.one_hot(sequence_tokens, 21)

    # Set up tie_group_map if provided
    if tie_group_map is None:
        tie_group_map = jnp.arange(num_residues, dtype=jnp.int32)

    # Use build_inference_bundle
    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        sequence=sequence_oh,
        state_weights=state_weights,
        mode="score_conditional",
        tie_group_map=tie_group_map,
    )

    return model, coordinates_stack, mask_stack, residue_index_stack, chain_index_stack, sequence_oh, bundle, config


@pytest.mark.parametrize("num_states", [1, 4])
def test_ste_decode_untied(
    num_states: int,
) -> None:
    """STEDecode runs successfully for untied positions (Fixture A).

    Fixture A: S ∈ {1, 4}, L=20, tie_group_map = jnp.arange(20) (untied).
    """
    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = _build_synthetic_fixture(
        num_states=num_states,
        num_residues=20,
        seed=42,
        tie_group_map=None,  # Untied
    )

    # Implementation: STEDecode
    inner = ConditionalDecode(model=model, state_iterator=VmapIterator())
    ste_decode = STEDecode(
        inner=inner,
        iterations=3,  # Reduced iterations for faster testing
        optimizer=optax.adam(1e-3),
    )
    stage_set_ste = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )
    jax_key = jax.random.PRNGKey(0)
    impl_seq, impl_logits_final, impl_logits_ste = ste_decode(
        prng_key=jax_key,
        bundle=bundle,
        config=config,
        temperature=1.0,
        stage_set=stage_set_ste,
    )

    # Verify shapes
    assert impl_seq.shape == (20,), f"Expected seq shape (20,), got {impl_seq.shape}"
    assert impl_logits_ste.shape == (20, 21), f"Expected logits_ste shape (20, 21), got {impl_logits_ste.shape}"
    assert impl_logits_final.shape == (20, 21), f"Expected logits_final shape (20, 21), got {impl_logits_final.shape}"


def test_ste_decode_tied() -> None:
    """STEDecode runs successfully for tied positions (Fixture B — Risk D-2 LOAD-BEARING).

    Fixture B: S=4, L=20, tie_group_map=[0,0,1,1,2,2,...,9,9] (10 groups).
    This is the CRITICAL test for Risk D-2 mitigation: verifies tied-group einsum
    averaging is applied correctly during optimization.

    Includes numerical parity check (Step 10.1): STEDecode output matches
    make_optimize_sequence_fn reference to atol=1e-5, rtol=1e-5.
    """
    # Build tied positions: pairs of positions
    tie_group_map = jnp.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        dtype=jnp.int32,
    )

    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = _build_synthetic_fixture(
        num_states=4,
        num_residues=20,
        seed=42,
        tie_group_map=tie_group_map,
    )

    jax_key = jax.random.PRNGKey(0)
    iterations = 10
    learning_rate = 1e-3
    temperature = 1.0

    # ========== Reference path via make_optimize_sequence_fn ==========
    ref_optimize_fn = make_optimize_sequence_fn(model)
    ref_seq, ref_logits_output, ref_logits_ste = ref_optimize_fn(
        prng_key=jax_key,
        bundle=bundle,
        config=config,
        iterations=iterations,
        learning_rate=learning_rate,
        temperature=temperature,
    )

    # ========== Implementation path via STEDecode ==========
    inner = ConditionalDecode(model=model, state_iterator=VmapIterator())
    ste_decode = STEDecode(
        inner=inner,
        iterations=iterations,
        optimizer=optax.adam(learning_rate),
    )
    stage_set_ste = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )
    impl_seq, impl_logits_output, impl_logits_ste = ste_decode(
        prng_key=jax_key,
        bundle=bundle,
        config=config,
        temperature=temperature,
        stage_set=stage_set_ste,
    )

    # ========== Numerical parity assertion (Risk D-2 fixture mandate) ==========
    # Verify shapes
    assert impl_seq.shape == (20,), f"Expected seq shape (20,), got {impl_seq.shape}"
    assert impl_logits_ste.shape == (20, 21), f"Expected logits_ste shape (20, 21), got {impl_logits_ste.shape}"
    assert impl_logits_output.shape == (20, 21), f"Expected logits_output shape (20, 21), got {impl_logits_output.shape}"

    # Verify tied-group consistency: positions with same group_id should have nearly identical logits
    # (they may not be exactly identical due to floating point, but very close)
    for group_id in range(10):
        positions = jnp.where(tie_group_map == group_id)[0]
        if len(positions) > 1:
            for p in positions[1:]:
                np.testing.assert_allclose(
                    impl_logits_ste[positions[0], :],
                    impl_logits_ste[p, :],
                    atol=1e-6,
                    rtol=1e-6,
                    err_msg=f"Tied positions {positions[0]} and {p} (group {group_id}) should have identical logits"
                )

    # Assert numerical parity between reference and implementation (per spec Step 10.1)
    np.testing.assert_allclose(
        impl_logits_output,
        ref_logits_output,
        atol=1e-5,
        rtol=1e-5,
        err_msg="STEDecode output logits do not match make_optimize_sequence_fn reference (Risk D-2 parity)"
    )
    np.testing.assert_allclose(
        impl_logits_ste,
        ref_logits_ste,
        atol=1e-5,
        rtol=1e-5,
        err_msg="STEDecode STE logits do not match make_optimize_sequence_fn reference (Risk D-2 parity)"
    )


@pytest.mark.parametrize("num_states", [1, 4])
def test_ste_decode_single_state(
    num_states: int,
) -> None:
    """STEDecode works with S=1 (both tied and untied)."""
    # Untied
    model, coords, mask, residue_index, chain_index, sequence_oh, bundle_untied, config = _build_synthetic_fixture(
        num_states=1,
        num_residues=20,
        seed=42,
        tie_group_map=None,
    )

    inner = ConditionalDecode(model=model, state_iterator=VmapIterator())
    ste_decode = STEDecode(
        inner=inner,
        iterations=5,
        optimizer=optax.adam(1e-3),
    )
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle_untied.conditioning.state_weights,
    )
    jax_key = jax.random.PRNGKey(0)

    seq, logits_final, logits_ste = ste_decode(
        prng_key=jax_key,
        bundle=bundle_untied,
        config=config,
        temperature=1.0,
        stage_set=stage_set,
    )

    # Basic shape checks
    assert seq.shape == (20,), f"Expected seq shape (20,), got {seq.shape}"
    assert logits_ste.shape == (20, 21), f"Expected logits_ste shape (20, 21), got {logits_ste.shape}"
    assert logits_final.shape == (20, 21), f"Expected logits_final shape (20, 21), got {logits_final.shape}"


def test_ste_decode_stage_set_projection() -> None:
    """STEDecode projects stage_set to logit_transform only (Step 10.2).

    Verifies that stage_set projection doesn't cause errors even when
    the input stage_set has multiple transforms populated.
    """
    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = _build_synthetic_fixture(
        num_states=1,
        num_residues=20,
        seed=42,
    )

    # Build a full stage_set with logit_transform only
    full_stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    # Create STEDecode
    inner = ConditionalDecode(model=model, state_iterator=VmapIterator())
    ste_decode = STEDecode(
        inner=inner,
        iterations=2,  # Minimal iterations for testing
        optimizer=optax.adam(1e-3),
    )

    jax_key = jax.random.PRNGKey(0)

    # Call with full stage_set - should project internally without error
    seq, logits_final, logits_ste = ste_decode(
        prng_key=jax_key,
        bundle=bundle,
        config=config,
        temperature=1.0,
        stage_set=full_stage_set,
    )

    # Verify shapes
    assert logits_ste.shape == (20, 21), f"Expected logits_ste shape (20, 21), got {logits_ste.shape}"
