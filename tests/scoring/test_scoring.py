"""Tests for the scoring module."""
import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from prxteinmpnn.scoring.score import make_score_sequence, ScoringFn
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.types import Model

@pytest.fixture
def scoring_fn(mock_model_parameters: Model) -> ScoringFn:
    """Create a scoring function from a mock model."""
    return make_score_sequence(
        mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

@pytest.mark.parametrize("seq_len", [50, 76])
def test_make_score_sequence_from_mock_model(
    scoring_fn: ScoringFn,
    protein_structure: Protein,
    rng_key: PRNGKeyArray,
    seq_len: int,
):
    """Test `make_score_sequence` from a mock model with different sequence lengths."""
    original_len = protein_structure.aatype.shape[0]
    protein = jax.tree_util.tree_map(
        lambda x: x[:seq_len] if isinstance(x, jnp.ndarray) and x.shape and x.shape[0] == original_len else x,
        protein_structure
    )

    assert callable(scoring_fn)

    score, logits, decoding_order = scoring_fn(
        prng_key=rng_key,
        sequence=protein.aatype,
        structure_coordinates=protein.coordinates,
        mask=protein.mask,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        k_neighbors=48
    )

    chex.assert_shape(score, ())
    chex.assert_type(score, jnp.floating)
    chex.assert_shape(logits, (seq_len, 21))
    chex.assert_type(logits, jnp.floating)
    chex.assert_shape(decoding_order, (seq_len,))
    chex.assert_type(decoding_order, jnp.integer)

def test_score_sequence_is_deterministic(
    scoring_fn: ScoringFn,
    protein_structure: Protein,
    rng_key: PRNGKeyArray,
):
    """Test that `score_sequence` is deterministic."""
    score1, logits1, do1 = scoring_fn(
        prng_key=rng_key,
        sequence=protein_structure.aatype,
        structure_coordinates=protein_structure.coordinates,
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
        k_neighbors=48
    )
    score2, logits2, do2 = scoring_fn(
        prng_key=rng_key,
        sequence=protein_structure.aatype,
        structure_coordinates=protein_structure.coordinates,
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
        k_neighbors=48
    )
    chex.assert_trees_all_close(
        (score1, logits1, do1), (score2, logits2, do2), atol=1e-5, rtol=1e-5
    )

def test_perplexity_calculation(
    scoring_fn: ScoringFn,
    protein_structure: Protein,
    rng_key: PRNGKeyArray,
):
    """Test that the perplexity is calculated correctly."""
    score, _, _ = scoring_fn(
        prng_key=rng_key,
        sequence=protein_structure.aatype,
        structure_coordinates=protein_structure.coordinates,
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
        k_neighbors=48
    )
    perplexity = jnp.exp(score)
    chex.assert_shape(perplexity, ())
    chex.assert_type(perplexity, jnp.floating)

def test_jit_compilation(
    scoring_fn: ScoringFn,
    protein_structure: Protein,
    rng_key: PRNGKeyArray,
):
    """Test that the scoring function can be JIT compiled."""
    compiled_fn = jax.jit(scoring_fn, static_argnames=("k_neighbors",))

    score, logits, decoding_order = compiled_fn(
        prng_key=rng_key,
        sequence=protein_structure.aatype,
        structure_coordinates=protein_structure.coordinates,
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
        k_neighbors=48
    )

    seq_len = protein_structure.aatype.shape[0]
    chex.assert_shape(score, ())
    chex.assert_type(score, jnp.floating)
    chex.assert_shape(logits, (seq_len, 21))
    chex.assert_type(logits, jnp.floating)
    chex.assert_shape(decoding_order, (seq_len,))
    chex.assert_type(decoding_order, jnp.integer)

def test_batch_processing(
    scoring_fn: ScoringFn,
    protein_structure: Protein,
    rng_key: PRNGKeyArray,
):
    """Test that the scoring function can be batched with vmap."""
    batch_size = 4

    # Create a batch of inputs
    batched_protein = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * batch_size), protein_structure
    )

    # vmap the scoring function
    vmapped_scoring_fn = jax.vmap(
        scoring_fn, in_axes=(0, 0, 0, 0, 0, 0, None, None, None)
    )

    batch_rng_key = jax.random.split(rng_key, batch_size)

    scores, logits, decoding_orders = vmapped_scoring_fn(
        batch_rng_key,
        batched_protein.aatype,
        batched_protein.coordinates,
        batched_protein.mask,
        batched_protein.residue_index,
        batched_protein.chain_index,
        48, # k_neighbors
        None, # backbone_noise
        None, # ar_mask
    )

    seq_len = protein_structure.aatype.shape[0]
    chex.assert_shape(scores, (batch_size,))
    chex.assert_shape(logits, (batch_size, seq_len, 21))
    chex.assert_shape(decoding_orders, (batch_size, seq_len))
