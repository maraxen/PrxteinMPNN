"""Parity tests for ConditionalDecode mode class (Task 7).

ConditionalDecode wraps conditional scoring with state-axis iteration via
a MapIterator (Vmap, SafeMap). Tests verify numerical parity with
driver.py:_decode_conditional over fixture sizes S ∈ {1, 4, 8} and
two iterator strategies (Vmap, SafeMap).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference.decode.conditional import ConditionalDecode
from prxteinmpnn.inference.driver import _decode_conditional
from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.inference.logits import make_stage_set
from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.tiling.iterator import VmapIterator, SafeMapIterator
from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 8,
    seed: int = 42,
) -> tuple[PrxteinMPNN, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, InferenceBundle, InferenceConfig]:
    """Build deterministic fixture with S=num_states, L=num_residues.

    Returns: (model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config)
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

    # Use build_inference_bundle
    # Set state_weights to uniform across states
    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        sequence=sequence_oh,
        state_weights=state_weights,
        mode="score_conditional",
    )

    return model, coordinates_stack, mask_stack, residue_index_stack, chain_index_stack, sequence_oh, bundle, config


@pytest.mark.parametrize("num_states", [1, 4, 8])
@pytest.mark.parametrize("iterator_factory", [VmapIterator, SafeMapIterator])
def test_conditional_decode_parity_with_driver(
    num_states: int,
    iterator_factory,
) -> None:
    """ConditionalDecode matches _decode_conditional for various S and iterator types."""
    if iterator_factory == SafeMapIterator:
        iterator = iterator_factory(tile=2)
    else:
        iterator = iterator_factory()

    model, coords, mask, residue_index, chain_index, sequence_oh, bundle, config = _build_synthetic_fixture(
        num_states=num_states, seed=42
    )

    # Get encoder output
    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Oracle: _decode_conditional from driver.py
    # Pass state_weights matching the bundle's conditioning
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )
    oracle_logits = _decode_conditional(
        model, k_dec, enc, bundle.conditioning, config, stage_set
    )

    # Implementation: ConditionalDecode
    cond_decode = ConditionalDecode(model=model, state_iterator=iterator)
    impl_logits = cond_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
    )

    # Verify parity
    np.testing.assert_allclose(oracle_logits, impl_logits, atol=1e-6, rtol=1e-6)
