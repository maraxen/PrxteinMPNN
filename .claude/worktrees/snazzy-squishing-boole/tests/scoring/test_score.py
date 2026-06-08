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

    def mock_features(
        key,
        coords,
        mask,
        residue_index,
        chain_index,
        backbone_noise,
        backbone_noise_mode=None,
        structure_mapping=None,
        **kwargs
    ):
        del (
            key,
            coords,
            mask,
            residue_index,
            chain_index,
            backbone_noise,
            backbone_noise_mode,
            structure_mapping,
            kwargs
        )
        n_residues = 76
        edge_f = jnp.zeros((n_residues, 48, 128))
        edge_i = jnp.zeros((n_residues, 48), dtype=jnp.int32)
        node_f = jnp.zeros((n_residues, 128))
        padding = jnp.zeros((n_residues,))
        return edge_f, edge_i, node_f, padding

    model_mock.features = MagicMock(side_effect=mock_features)

    # Mock encoder
    model_mock.encoder = MagicMock()
    model_mock.encoder.side_effect = lambda ef, ei, m, initial_node_features=None, key=None: (
        initial_node_features if initial_node_features is not None else jnp.zeros((76, 128)),
        ef
    )

    # Mock __call__ for per-state array interface (composability contract)
    def mock_call(coords, mask, residue_index, chain_index, **kwargs):
        """Return (node_features, edge_features, neighbor_indices)"""
        n_residues = 76
        node_f = jnp.zeros((n_residues, 128))
        edge_f = jnp.zeros((n_residues, 48, 128))
        edge_i = jnp.zeros((n_residues, 48), dtype=jnp.int32)
        return node_f, edge_f, edge_i

    model_mock.side_effect = mock_call

    # Mock decoder
    model_mock.decoder = MagicMock()
    model_mock.decoder.call_conditional.side_effect = lambda nb, eb, i, m, a, oh, w, **kw: jnp.zeros((nb.shape[0], 128))

    # Mock projections
    model_mock.w_out = MagicMock()
    model_mock.w_out.side_effect = lambda x: jnp.zeros(21)

    model_mock.w_s_embed = MagicMock()
    model_mock.w_s_embed.weight = jnp.zeros((21, 128))

    return model_mock


@pytest.mark.parametrize("jit_compile", [True, False])
@jaxtyped
def test_make_score_sequence_output_shape_and_type(
    mock_model: PrxteinMPNN, protein_structure: Protein, jit_compile: bool,
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
            "prxteinmpnn.scoring.score.jax.jit", new=lambda fn, *args, **kwargs: fn,
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
    score, logits, order = score_fn(
        prng_key,
        sequence,
        coords,
        mask,
        residue_index,
        chain_index,
        _k_neighbors=48,
        backbone_noise=0.1,
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
