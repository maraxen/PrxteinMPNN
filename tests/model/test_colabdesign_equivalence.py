"""Tests for equivalence between PrxteinMPNN and ColabDesign implementations.

This test suite validates that the PrxteinMPNN implementation produces
equivalent outputs to the original ColabDesign ProteinMPNN for all three
decoding modes: unconditional, conditional, and autoregressive.

Target correlation: >0.95 for all paths
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from colabdesign.mpnn import mk_mpnn_model

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.utils.data_structures import Protein

# Alphabet conversion between AlphaFold and MPNN orderings
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"


def af_logits_to_mpnn(logits_af):
    """Convert logits from AlphaFold alphabet order to MPNN alphabet order."""
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]


@pytest.fixture
def test_structure_path():
    """Path to test PDB structure."""
    return "tests/data/1ubq.pdb"


@pytest.fixture
def colabdesign_model(test_structure_path):
    """Initialize ColabDesign MPNN model."""
    model = mk_mpnn_model()
    model.prep_inputs(pdb_filename=test_structure_path)
    return model


@pytest.fixture
def prxteinmpnn_model():
    """Initialize PrxteinMPNN with ColabDesign weights."""
    key = jax.random.key(42)
    return load_model("v_48_020", key=key)


@pytest.fixture
def protein_data(test_structure_path):
    """Load protein structure data."""
    protein_tuple = next(parse_input(test_structure_path))
    return Protein.from_tuple(protein_tuple)


class TestColabDesignEquivalence:
    """Test suite for ColabDesign equivalence."""

    def test_unconditional_logits(self, colabdesign_model, prxteinmpnn_model, protein_data):
        """Test unconditional logits match between implementations.

        Unconditional logits are structure-based predictions without any
        sequence input. This is the most basic test of equivalence.

        Target: correlation > 0.95
        """
        key = jax.random.key(42)

        # Get ColabDesign unconditional logits
        colab_logits = colabdesign_model.get_unconditional_logits(key=key)
        colab_logits_mpnn = af_logits_to_mpnn(colab_logits)

        # Get PrxteinMPNN unconditional logits
        _, prx_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "unconditional",
            prng_key=key,
        )

        # Calculate correlation
        correlation = np.corrcoef(
            colab_logits_mpnn.flatten(),
            prx_logits.flatten(),
        )[0, 1]

        # Assert high correlation
        assert correlation > 0.95, (
            f"Unconditional logits correlation {correlation:.4f} < 0.95. "
            "PrxteinMPNN implementation does not match ColabDesign."
        )

    def test_conditional_logits(self, colabdesign_model, prxteinmpnn_model, protein_data):
        """Test conditional logits match between implementations.

        Conditional logits score a fixed sequence. This tests the conditional
        decoder with ar_mask=0 (all positions scored in parallel).

        Target: correlation > 0.95
        """
        key = jax.random.key(42)
        L = len(colabdesign_model._inputs["S"])

        # Use native sequence from structure
        native_seq_af_list = [int(x) for x in colabdesign_model._inputs["S"]]
        native_seq_mpnn = jnp.array([
            MPNN_ALPHABET.index(AF_ALPHABET[idx]) for idx in native_seq_af_list
        ])

        # Get ColabDesign conditional logits
        # Note: ColabDesign expects a Python list, not a JAX array
        ar_mask_zero = jnp.zeros((L, L), dtype=jnp.int32)
        colab_result = colabdesign_model.score(
            seq=native_seq_af_list,
            ar_mask=ar_mask_zero,
            key=key,
        )
        colab_logits_mpnn = af_logits_to_mpnn(colab_result["logits"])

        # Get PrxteinMPNN conditional logits
        one_hot_seq = jax.nn.one_hot(native_seq_mpnn, 21)
        _, prx_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "conditional",
            ar_mask=ar_mask_zero,
            one_hot_sequence=one_hot_seq,
            prng_key=key,
        )

        # Calculate correlation
        correlation = np.corrcoef(
            colab_logits_mpnn.flatten(),
            prx_logits.flatten(),
        )[0, 1]

        # Assert high correlation
        assert correlation > 0.95, (
            f"Conditional logits correlation {correlation:.4f} < 0.95. "
            "PrxteinMPNN conditional decoder does not match ColabDesign."
        )

    def test_autoregressive_sampling(self, colabdesign_model, prxteinmpnn_model, protein_data):
        """Test autoregressive sampling matches between implementations.

        Autoregressive sampling generates sequences by sampling one position
        at a time. This tests the full autoregressive decoder path.

        Target: correlation > 0.95
        """
        key = jax.random.key(42)
        L = len(colabdesign_model._inputs["S"])

        # Use fixed decoding order for reproducibility
        fixed_order = np.arange(L)

        # Construct ar_mask for sequential decoding
        ar_mask = jnp.zeros((L, L), dtype=jnp.int32)
        for i, pos in enumerate(fixed_order):
            ar_mask = ar_mask.at[pos, fixed_order[:i]].set(1)

        # Get ColabDesign autoregressive logits
        temperature = 0.1
        colab_sample = colabdesign_model.sample(
            num=1,
            batch=1,
            temperature=temperature,
            decoding_order=fixed_order,
            key=key,
        )
        colab_logits_mpnn = af_logits_to_mpnn(colab_sample["logits"][0])

        # Get PrxteinMPNN autoregressive logits
        _, prx_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "autoregressive",
            ar_mask=ar_mask,
            temperature=temperature,
            prng_key=key,
        )

        # Calculate correlation
        correlation = np.corrcoef(
            colab_logits_mpnn.flatten(),
            prx_logits.flatten(),
        )[0, 1]

        # Assert high correlation
        assert correlation > 0.95, (
            f"Autoregressive logits correlation {correlation:.4f} < 0.95. "
            "PrxteinMPNN autoregressive decoder does not match ColabDesign."
        )

    def test_ar_first_step_matches_unconditional(self, prxteinmpnn_model, protein_data):
        """Test that first autoregressive step matches unconditional logits.

        When the first position is decoded with no prior context (ar_mask=0
        for that position), it should produce identical logits to the
        unconditional path. This is a sanity check for the autoregressive
        implementation.
        """
        key = jax.random.key(42)
        L = protein_data.coordinates.shape[0]

        # Get unconditional logits
        _, unconditional_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "unconditional",
            prng_key=key,
        )

        # Get autoregressive logits
        fixed_order = np.arange(L)
        ar_mask = jnp.zeros((L, L), dtype=jnp.int32)
        for i, pos in enumerate(fixed_order):
            ar_mask = ar_mask.at[pos, fixed_order[:i]].set(1)

        _, ar_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "autoregressive",
            ar_mask=ar_mask,
            temperature=0.1,
            prng_key=key,
        )

        # First position should be identical (no context)
        max_diff = np.abs(unconditional_logits[0] - ar_logits[0]).max()

        assert max_diff < 0.001, (
            f"First AR step differs from unconditional by {max_diff:.6f}. "
            "Expected near-identical outputs for position with no context."
        )

    def test_conditional_with_zero_mask_matches_unconditional(
        self, prxteinmpnn_model, protein_data,
    ):
        """Test that conditional with ar_mask=0 matches unconditional.

        When ar_mask is all zeros (no autoregressive masking), the conditional
        decoder should produce the same results as unconditional, since no
        sequence information is being used.
        """
        key = jax.random.key(42)
        L = protein_data.coordinates.shape[0]

        # Get unconditional logits
        _, unconditional_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "unconditional",
            prng_key=key,
        )

        # Get conditional logits with ar_mask=0
        ar_mask_zero = jnp.zeros((L, L), dtype=jnp.int32)
        dummy_seq = jax.nn.one_hot(jnp.zeros(L, dtype=jnp.int32), 21)

        _, conditional_logits = prxteinmpnn_model(
            protein_data.coordinates,
            protein_data.mask,
            protein_data.residue_index,
            protein_data.chain_index,
            "conditional",
            ar_mask=ar_mask_zero,
            one_hot_sequence=dummy_seq,
            prng_key=key,
        )

        # Should be nearly identical
        correlation = np.corrcoef(
            unconditional_logits.flatten(),
            conditional_logits.flatten(),
        )[0, 1]

        assert correlation > 0.999, (
            f"Conditional with ar_mask=0 correlation {correlation:.6f} < 0.999. "
            "Expected near-identical outputs when no sequence context is used."
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
