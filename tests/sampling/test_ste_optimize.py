"""Tests for the ste_optimize module."""
from functools import partial
from typing import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int, PRNGKeyArray

from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    Logits,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
)


class MockPrxteinMPNN(eqx.Module):
    """A mock PrxteinMPNN model that returns logits of the correct shape."""

    num_residues: int = eqx.static_field()
    num_classes: int = eqx.static_field()

    def __call__(
        self,
        structure_coordinates: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_index: ResidueIndex,
        chain_index: ChainIndex,
        decoding_approach: str,
        one_hot_sequence: Array,
        ar_mask: AutoRegressiveMask,
        backbone_noise: BackboneNoise | None = None,
    ) -> tuple[None, Logits]:
        """Return mock logits."""
        chex.assert_shape(
            structure_coordinates, (self.num_residues, 4, 3)
        )
        chex.assert_shape(mask, (self.num_residues,))
        chex.assert_shape(residue_index, (self.num_residues,))
        chex.assert_shape(chain_index, (self.num_residues,))
        chex.assert_shape(
            one_hot_sequence, (self.num_residues, self.num_classes)
        )
        chex.assert_shape(
            ar_mask, (self.num_residues, self.num_residues)
        )

        return None, jax.random.normal(
            jax.random.key(0),
            (self.num_residues, self.num_classes),
            dtype=jnp.float32,
        )


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    num_residues = 10
    return {
        "structure_coordinates": jnp.ones((num_residues, 4, 3), dtype=jnp.float32),
        "mask": jnp.ones((num_residues,), dtype=jnp.bool_),
        "residue_index": jnp.arange(num_residues, dtype=jnp.int32),
        "chain_index": jnp.zeros((num_residues,), dtype=jnp.int32),
    }


def test_make_optimize_sequence_fn():
    """Test that `make_optimize_sequence_fn` returns a callable."""
    model = MockPrxteinMPNN(num_residues=10, num_classes=21)
    optimize_fn = make_optimize_sequence_fn(model)
    assert isinstance(optimize_fn, Callable)


def test_optimize_sequence_shapes(dummy_data):
    """Test that `optimize_sequence` runs and returns correct shapes."""
    key = jax.random.key(0)
    num_residues = 10
    num_classes = 21

    model = MockPrxteinMPNN(num_residues=num_residues, num_classes=num_classes)
    optimize_fn = make_optimize_sequence_fn(model)

    final_sequence, final_output_logits, final_logits = optimize_fn(
        prng_key=key,
        iterations=2,
        learning_rate=0.01,
        temperature=1.0,
        **dummy_data,
    )

    chex.assert_shape(final_sequence, (num_residues,))
    chex.assert_shape(final_output_logits, (num_residues, num_classes))
    chex.assert_shape(final_logits, (num_residues, num_classes))


def test_optimize_sequence_with_backbone_noise(dummy_data):
    """Test that `optimize_sequence` runs with backbone noise."""
    key = jax.random.key(0)
    num_residues = 10
    num_classes = 21

    model = MockPrxteinMPNN(num_residues=num_residues, num_classes=num_classes)
    optimize_fn = make_optimize_sequence_fn(model)

    final_sequence, final_output_logits, final_logits = optimize_fn(
        prng_key=key,
        iterations=2,
        learning_rate=0.01,
        temperature=1.0,
        backbone_noise=jnp.ones((num_residues, 4, 3), dtype=jnp.float32),
        **dummy_data,
    )

    chex.assert_shape(final_sequence, (num_residues,))
    chex.assert_shape(final_output_logits, (num_residues, num_classes))
    chex.assert_shape(final_logits, (num_residues, num_classes))


def test_optimize_sequence_logic(dummy_data):
    """Test that `optimize_sequence` updates logits."""
    key = jax.random.key(0)
    num_residues = 10
    num_classes = 21

    model = MockPrxteinMPNN(num_residues=num_residues, num_classes=num_classes)
    optimize_fn = make_optimize_sequence_fn(model)

    _, _, final_logits = optimize_fn(
        prng_key=key,
        iterations=2,
        learning_rate=0.01,
        temperature=1.0,
        **dummy_data,
    )

    initial_logits = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)
    assert not jnp.allclose(initial_logits, final_logits)


def test_optimize_sequence_jit(dummy_data):
    """Test that `optimize_sequence` can be JIT-compiled."""
    key = jax.random.key(0)
    num_residues = 10
    num_classes = 21

    model = MockPrxteinMPNN(num_residues=num_residues, num_classes=num_classes)
    optimize_fn = make_optimize_sequence_fn(model)

    jit_optimize_fn = jax.jit(optimize_fn)

    final_sequence, final_output_logits, final_logits = jit_optimize_fn(
        prng_key=key,
        iterations=2,
        learning_rate=0.01,
        temperature=1.0,
        **dummy_data,
    )

    chex.assert_shape(final_sequence, (num_residues,))
    chex.assert_shape(final_output_logits, (num_residues, num_classes))
    chex.assert_shape(final_logits, (num_residues, num_classes))
