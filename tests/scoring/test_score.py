"""Tests for scoring functions."""
from unittest.mock import MagicMock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import jaxtyped

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture
def mock_model() -> PrxteinMPNN:
    """Fixture for a mock PrxteinMPNN model."""
    model_mock = MagicMock(spec=PrxteinMPNN)

    def mock_call_impl(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach,
        prng_key,
        ar_mask,
        one_hot_sequence,
        temperature=None,
        bias=None,
        backbone_noise=None,
        structure_mapping=None,
    ):
        # Using a dynamic shape from the input tracer (structure_coordinates.shape[0])
        # inside the mock's side_effect causes a ConcretizationTypeError during JIT.
        # We hardcode the residue count because the '1ubq.pdb' fixture is static.
        n_residues = 76
        logits = jax.random.uniform(
            prng_key, shape=(n_residues, 21), dtype=jnp.float32
        )
        return None, logits

    model_mock.side_effect = mock_call_impl
    return model_mock


@pytest.mark.parametrize("jit_compile", [True, False])
@jaxtyped
def test_make_score_sequence_output_shape_and_type(
    mock_model: PrxteinMPNN, protein_structure: Protein, jit_compile: bool
):
    """Test the output shape and type of the scoring function."""

    def get_score_fn():
        """Returns the scoring function, respecting the JIT context."""
        return make_score_sequence(mock_model)

    if jit_compile:
        score_fn = get_score_fn()
    else:
        # To test the non-JIT version, we patch jax.jit to be an identity function.
        # This prevents the @partial(jax.jit, ...) decorator inside make_score_sequence
        # from compiling the function.
        with patch(
            "prxteinmpnn.scoring.score.jax.jit", new=lambda fn, *args, **kwargs: fn
        ):
            score_fn = get_score_fn()

    prng_key = jax.random.key(0)

    coords = protein_structure.coordinates
    mask = protein_structure.mask
    residue_index = protein_structure.residue_index
    chain_index = protein_structure.chain_index
    one_hot_sequence = protein_structure.one_hot_sequence
    sequence = protein_structure.aatype
    length = sequence.shape[0]

    # Test with integer sequence
    score, logits, order = score_fn(
        prng_key,
        sequence,
        coords,
        mask,
        residue_index,
        chain_index,
        _k_neighbors=48,
    )
    chex.assert_shape(score, ())
    chex.assert_type(score, float)
    chex.assert_shape(logits, (length, 21))
    chex.assert_shape(order, (length,))
    chex.assert_tree_all_finite((score, logits))

    # Test with one-hot sequence
    score, logits, order = score_fn(
        prng_key,
        one_hot_sequence,
        coords,
        mask,
        residue_index,
        chain_index,
        _k_neighbors=48,
    )
    chex.assert_shape(score, ())
    chex.assert_type(score, float)
    chex.assert_tree_all_finite((score, logits))

    # Test with backbone noise
    noise = jax.random.normal(prng_key, coords.shape)
    score, logits, order = score_fn(
        prng_key,
        sequence,
        coords,
        mask,
        residue_index,
        chain_index,
        _k_neighbors=48,
        backbone_noise=noise,
    )
    chex.assert_shape(score, ())
    chex.assert_type(score, float)
    chex.assert_tree_all_finite((score, logits))

    # Test with custom autoregressive mask
    ar_mask = jnp.zeros((length, length))
    score, logits, order = score_fn(
        prng_key,
        sequence,
        coords,
        mask,
        residue_index,
        chain_index,
        _k_neighbors=48,
        ar_mask=ar_mask,
    )
    chex.assert_shape(score, ())
    chex.assert_type(score, float)
    chex.assert_tree_all_finite((score, logits))
