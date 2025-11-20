"""Tests for unconditional_logits."""
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
    from prxteinmpnn.model import PrxteinMPNN

from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture
def mock_model(protein) -> "PrxteinMPNN":
    """Create a mock PrxteinMPNN model."""
    mock = MagicMock()
    mock.return_value = (
        None,
        jnp.ones((protein.coordinates.shape[0], 21)),
    )
    return mock


@pytest.fixture
def protein(protein_structure: Protein) -> Protein:
    """Create a mock protein."""
    return protein_structure


def test_make_unconditional_logits_fn(mock_model: "PrxteinMPNN"):
    """Test the make_unconditional_logits_fn function."""
    logits_fn = make_unconditional_logits_fn(mock_model)
    assert callable(logits_fn)


def test_unconditional_logits(
    mock_model: "PrxteinMPNN",
    protein: Protein,
    rng_key: PRNGKeyArray,
):
    """Test the unconditional_logits function."""
    with patch("prxteinmpnn.sampling.unconditional_logits.jax.jit", new=lambda fn, *args, **kwargs: fn):
        logits_fn = make_unconditional_logits_fn(mock_model)
        logits = logits_fn(
            rng_key,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
        )
        chex.assert_shape(logits, (protein.coordinates.shape[0], 21))
        mock_model.assert_called_once_with(
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            decoding_approach="unconditional",
            ar_mask=None,
            backbone_noise=None,
        )


def test_unconditional_logits_with_ar_mask(
    mock_model: "PrxteinMPNN",
    protein: Protein,
    rng_key: PRNGKeyArray,
):
    """Test the unconditional_logits function with an autoregressive mask."""
    with patch("prxteinmpnn.sampling.unconditional_logits.jax.jit", new=lambda fn, *args, **kwargs: fn):
        logits_fn = make_unconditional_logits_fn(mock_model)
        ar_mask = jnp.ones((protein.coordinates.shape[0], protein.coordinates.shape[0]))
        logits = logits_fn(
            rng_key,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            ar_mask=ar_mask,
        )
        chex.assert_shape(logits, (protein.coordinates.shape[0], 21))
        mock_model.assert_called_once_with(
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            decoding_approach="unconditional",
            ar_mask=ar_mask,
            backbone_noise=None,
        )


def test_unconditional_logits_with_backbone_noise(
    mock_model: "PrxteinMPNN",
    protein: Protein,
    rng_key: PRNGKeyArray,
):
    """Test the unconditional_logits function with backbone noise."""
    with patch("prxteinmpnn.sampling.unconditional_logits.jax.jit", new=lambda fn, *args, **kwargs: fn):
        logits_fn = make_unconditional_logits_fn(mock_model)
        backbone_noise = jnp.ones((protein.coordinates.shape[0], 4, 3))
        logits = logits_fn(
            rng_key,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            backbone_noise=backbone_noise,
        )
        chex.assert_shape(logits, (protein.coordinates.shape[0], 21))
        mock_model.assert_called_once_with(
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            decoding_approach="unconditional",
            ar_mask=None,
            backbone_noise=backbone_noise,
        )
