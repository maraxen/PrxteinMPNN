"""Parity tests for AutoregressiveDecode mode class (Task 9).

AutoregressiveDecode wraps autoregressive sampling with two iterators:
- state_iterator: MapIterator for the S axis (state dimension)
- wave_iterator: ScanIterator for the W axis (wave schedule)

The wave_iterator carries the sequence through the scan; after the scan returns,
a post-hoc scatter scan maps per-wave logits back to per-position logits (Risk D-11).

Tests verify numerical parity with driver.py:decode_ar over fixture sizes and
two iterator strategies. Also verify CarryShape round-trip and S-axis parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference.decode.autoregressive import AutoregressiveDecode
from prxteinmpnn.inference.driver import decode_ar
from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.inference.logits import make_stage_set
from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.tiling.carry_shape import CarryShape
from prxteinmpnn.tiling.iterator import VmapIterator, SafeMapIterator, JaxScanIterator
from prxteinmpnn.types.bundles import InferenceBundle, WaveScheduleBundle
from prxteinmpnn.types.configs import InferenceConfig


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 8,
    seed: int = 42,
) -> tuple[PrxteinMPNN, InferenceBundle, InferenceConfig]:
    """Build deterministic AR fixture with S=num_states, L=num_residues.

    Returns: (model, bundle, config)
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

    # Build bundle
    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        state_weights=state_weights,
        sequence=None,  # Will be sampled
        mode="sample_autoregressive",
    )

    return model, bundle, config


def _dummy_decoding_order_fn(wave):
    """Dummy decoding order function."""
    return jnp.arange(wave.group_ids.shape[0])  # Return number of waves as dummy order


def test_autoregressive_parity_untied_s1():
    """Test AR decode parity with driver.decode_ar, S=1 untied positions."""
    model, bundle, config = _build_synthetic_fixture(num_states=1, num_residues=8, seed=42)
    key = jax.random.PRNGKey(999)

    # Build AR mode
    L = int(bundle.geometry.residue_index.shape[1])
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        decoding_order_fn=_dummy_decoding_order_fn,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    # Get stage set
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    # Encode
    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Call AR decode
    result_ar = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    # Call driver.decode_ar
    result_driver = decode_ar(
        model=model,
        key=k_dec,
        enc=enc,
        cond=bundle.conditioning,
        wave=bundle.wave,
        config=config,
        stage_set=stage_set,
    )

    # Assert parity
    np.testing.assert_allclose(result_ar.sequence, result_driver.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_ar.logits, result_driver.logits, atol=1e-6)


def test_autoregressive_parity_untied_s4():
    """Test AR decode parity with driver.decode_ar, S=4 untied positions."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=12, seed=43)
    key = jax.random.PRNGKey(999)

    # Build AR mode
    L = bundle.geometry.residue_index.shape[1]
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        decoding_order_fn=_dummy_decoding_order_fn,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    result_ar = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    result_driver = decode_ar(
        model=model,
        key=k_dec,
        enc=enc,
        cond=bundle.conditioning,
        wave=bundle.wave,
        config=config,
        stage_set=stage_set,
    )

    np.testing.assert_allclose(result_ar.sequence, result_driver.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_ar.logits, result_driver.logits, atol=1e-6)


def test_carry_shape_round_trip():
    """Test CarryShape materialization and round-trip."""
    L = 12
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)

    # Materialize init
    init = wave_carry.materialize()

    # Check shape and dtype
    assert init.shape == (L,), f"Expected shape ({L},), got {init.shape}"
    assert init.dtype == jnp.int32, f"Expected dtype int32, got {init.dtype}"

    # Check values are zeros
    np.testing.assert_array_equal(init, jnp.zeros((L,), dtype=jnp.int32))


def test_state_iterator_parity_vmap_vs_safemap():
    """Test S-axis parity: Vmap vs SafeMap inside wave scan."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=8, seed=44)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # AR decode with Vmap
    ar_vmap = AutoregressiveDecode(
        model=model,
        decoding_order_fn=lambda w: jnp.arange(L),
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    result_vmap = ar_vmap(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    # AR decode with SafeMap
    ar_safemap = AutoregressiveDecode(
        model=model,
        decoding_order_fn=lambda w: jnp.arange(L),
        state_iterator=SafeMapIterator(tile=2),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    result_safemap = ar_safemap(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    # Results should be identical
    np.testing.assert_allclose(result_vmap.sequence, result_safemap.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_vmap.logits, result_safemap.logits, atol=1e-6)


def test_autoregressive_with_tied_positions():
    """Test AR decode with tied positions (S=1, tied groups)."""
    model, bundle, config = _build_synthetic_fixture(num_states=1, num_residues=8, seed=45)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]

    # Override tie_group_map: [0,0,1,1,2,2,3,3]
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    bundle = eqx.tree_at(
        lambda b: b.conditioning.tie_group_map,
        bundle,
        jnp.expand_dims(tie_group_map, 0),
    )

    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        decoding_order_fn=_dummy_decoding_order_fn,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Should not raise
    result_ar = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    # Compare with driver
    result_driver = decode_ar(
        model=model,
        key=k_dec,
        enc=enc,
        cond=bundle.conditioning,
        wave=bundle.wave,
        config=config,
        stage_set=stage_set,
    )

    np.testing.assert_allclose(result_ar.sequence, result_driver.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_ar.logits, result_driver.logits, atol=1e-6)


def test_autoregressive_with_tied_positions_s4():
    """Test AR decode with tied positions (S=4, tied groups)."""
    model, bundle, config = _build_synthetic_fixture(num_states=4, num_residues=12, seed=46)
    key = jax.random.PRNGKey(999)

    L = bundle.geometry.residue_index.shape[1]

    # Override tie_group_map: [0,0,1,1,2,2,3,3,4,4,5,5]
    tie_group_map = jnp.arange(L) // 2
    bundle = eqx.tree_at(
        lambda b: b.conditioning.tie_group_map,
        bundle,
        jnp.expand_dims(tie_group_map, 0),
    )

    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    ar_decode = AutoregressiveDecode(
        model=model,
        decoding_order_fn=_dummy_decoding_order_fn,
        state_iterator=VmapIterator(),
        wave_iterator=JaxScanIterator(),
        wave_carry=wave_carry,
    )

    stage_set = make_stage_set(
        strategy="arithmetic_mean",
        state_weights=bundle.conditioning.state_weights,
    )

    k_enc, k_dec = jax.random.split(key)
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    result_ar = ar_decode(
        key=k_dec,
        enc=enc,
        bundle=bundle.conditioning,
        config=config,
        stage_set=stage_set,
        wave=bundle.wave,
    )

    result_driver = decode_ar(
        model=model,
        key=k_dec,
        enc=enc,
        cond=bundle.conditioning,
        wave=bundle.wave,
        config=config,
        stage_set=stage_set,
    )

    np.testing.assert_allclose(result_ar.sequence, result_driver.sequence, rtol=1e-6)
    np.testing.assert_allclose(result_ar.logits, result_driver.logits, atol=1e-6)
