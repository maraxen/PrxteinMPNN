"""Test sampling diversity at different temperatures.

This test validates that the sampling process produces diverse sequences
at high temperature, which is important for design applications.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling import make_sample_sequences
from prxteinmpnn.utils.data_structures import Protein


@pytest.mark.slow
def test_diversity():
    """Test that high temperature produces diverse sequences.

    At high temperature (T=2.0), the model should produce sequences
    that are quite different from each other, validating that the
    sampling process is working correctly and exploring sequence space.
    """
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"

    try:
        model = load_model(local_path=str(weights_path))
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    sample_fn = make_sample_sequences(model)

    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    key = jax.random.PRNGKey(42)
    sequences = []

    # Sample 20 sequences at high temperature
    for i in range(20):
        key, subkey = jax.random.split(key)
        sampled_seq, _, _ = sample_fn(
            subkey,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            temperature=jnp.array(2.0),  # High temperature for diversity
        )
        sequences.append(sampled_seq)

    sequences = jnp.stack(sequences)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = (sequences[i] == sequences[j]).sum() / protein.mask.sum()
            similarities.append(float(sim))

    mean_sim = jnp.mean(jnp.array(similarities))
    print(f"\nMean pairwise similarity (T=2.0): {mean_sim:.1%}")
    print(f"Number of unique sequences: {len(jnp.unique(sequences, axis=0))}/20")

    # At high temperature, sequences should be quite diverse
    assert mean_sim < 0.40, (
        f"Similarity {mean_sim:.1%} too high at T=2.0. "
        f"Expected < 40% for diverse sampling."
    )

    print("✅ Diversity test passed!")


@pytest.mark.slow
def test_temperature_effect():
    """Test that temperature affects diversity as expected.

    Low temperature should produce more similar sequences (higher similarity),
    while high temperature should produce more diverse sequences (lower similarity).
    """
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"

    try:
        model = load_model(local_path=str(weights_path))
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    sample_fn = make_sample_sequences(model)

    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    def measure_diversity(temperature, num_samples=10):
        """Measure diversity at a given temperature."""
        key = jax.random.PRNGKey(123)
        sequences = []

        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            sampled_seq, _, _ = sample_fn(
                subkey,
                protein.coordinates,
                protein.mask,
                protein.residue_index,
                protein.chain_index,
                temperature=jnp.array(temperature),
            )
            sequences.append(sampled_seq)

        sequences = jnp.stack(sequences)

        # Calculate mean pairwise similarity
        similarities = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                sim = (sequences[i] == sequences[j]).sum() / protein.mask.sum()
                similarities.append(float(sim))

        return jnp.mean(jnp.array(similarities))

    # Test at different temperatures
    low_temp_sim = measure_diversity(0.1, num_samples=8)
    high_temp_sim = measure_diversity(2.0, num_samples=8)

    print(f"\nSimilarity at T=0.1: {low_temp_sim:.1%}")
    print(f"Similarity at T=2.0: {high_temp_sim:.1%}")

    # Higher temperature should give lower similarity (more diversity)
    assert high_temp_sim < low_temp_sim, (
        f"High temp similarity ({high_temp_sim:.1%}) should be lower than "
        f"low temp similarity ({low_temp_sim:.1%})"
    )

    print("✅ Temperature effect validated!")


@pytest.mark.slow
def test_low_temperature_consistency():
    """Test that low temperature produces consistent sequences.

    At very low temperature (T=0.01), the model should produce
    nearly identical sequences, showing that sampling converges
    to the mode of the distribution.
    """
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"

    try:
        model = load_model(local_path=str(weights_path))
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    sample_fn = make_sample_sequences(model)

    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    key = jax.random.PRNGKey(42)
    sequences = []

    # Sample 10 sequences at very low temperature
    for i in range(10):
        key, subkey = jax.random.split(key)
        sampled_seq, _, _ = sample_fn(
            subkey,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            temperature=jnp.array(0.01),  # Very low temperature
        )
        sequences.append(sampled_seq)

    sequences = jnp.stack(sequences)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = (sequences[i] == sequences[j]).sum() / protein.mask.sum()
            similarities.append(float(sim))

    mean_sim = jnp.mean(jnp.array(similarities))
    print(f"\nMean pairwise similarity (T=0.01): {mean_sim:.1%}")

    # At very low temperature, sequences should be very similar
    assert mean_sim > 0.90, (
        f"Similarity {mean_sim:.1%} too low at T=0.01. "
        f"Expected > 90% for low-temperature sampling."
    )

    print("✅ Low temperature consistency test passed!")
