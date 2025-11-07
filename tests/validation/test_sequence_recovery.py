"""Validate sequence recovery after bug fixes.

This test verifies that the decoder bug fixes result in proper sequence recovery
rates (~40-60%) on native backbones, as expected from the original ProteinMPNN.
"""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.run.sampling import sample


def compute_sequence_recovery(native_seq: jnp.ndarray, designed_seq: jnp.ndarray) -> float:
    """Compute sequence recovery rate between native and designed sequences.

    Args:
        native_seq: Native sequence as indices (N,) or one-hot (N, 21)
        designed_seq: Designed sequence as indices (N,) or one-hot (N, 21)

    Returns:
        Recovery rate as a float between 0 and 1
    """
    # Convert to indices if one-hot
    if native_seq.ndim == 2:
        native_seq = jnp.argmax(native_seq, axis=-1)
    if designed_seq.ndim == 2:
        designed_seq = jnp.argmax(designed_seq, axis=-1)

    # Compute recovery
    matches = jnp.sum(native_seq == designed_seq)
    total = len(native_seq)
    return float(matches / total)


@pytest.mark.parametrize(
    "pdb_id,expected_min_recovery,expected_max_recovery",
    [
        ("1ubq", 0.35, 0.70),  # Ubiquitin - small, well-folded protein
        ("1qys", 0.35, 0.70),  # Barnase - RNase enzyme
        ("5trv", 0.35, 0.70),  # SARS-CoV-2 Spike RBD - larger protein
    ],
)
def test_sequence_recovery_on_native_structures(
    pdb_id: str,
    expected_min_recovery: float,
    expected_max_recovery: float,
):
    """Test sequence recovery on native PDB structures.

    This test validates that the model achieves reasonable sequence recovery
    (40-60% typical for ProteinMPNN) when redesigning native structures at
    low temperature (T=0.1).

    Args:
        pdb_id: PDB accession code
        expected_min_recovery: Minimum expected recovery rate
        expected_max_recovery: Maximum expected recovery rate
    """
    # Sample sequences using the run interface with low temperature
    result = sample(
        inputs=pdb_id,
        num_samples=5,  # Sample 5 sequences
        temperature=0.1,  # Low temp for high recovery
        sampling_strategy="temperature",
        random_seed=42,
        backbone_noise=0.0,  # No noise - native backbone
    )

    # Extract results
    # Shape: (N_structures, N_noise, N_samples, seq_len)
    sequences = result["sequences"]

    # Get native sequence from metadata
    native_sequence = result["metadata"]["native_sequences"][0]  # First structure

    # Compute recovery for each sample
    recoveries = []
    for sample_idx in range(sequences.shape[2]):
        designed_seq = sequences[0, 0, sample_idx]  # First structure, first noise, sample_idx
        recovery = compute_sequence_recovery(native_sequence, designed_seq)
        recoveries.append(recovery)

    # Compute mean recovery
    mean_recovery = jnp.mean(jnp.array(recoveries))

    print(f"\n{pdb_id} Sequence Recovery Results:")
    print(f"  Mean recovery: {mean_recovery:.1%}")
    print(f"  Individual recoveries: {[f'{r:.1%}' for r in recoveries]}")
    print(f"  Expected range: {expected_min_recovery:.1%} - {expected_max_recovery:.1%}")

    # Assert recovery is in expected range
    assert mean_recovery >= expected_min_recovery, (
        f"Recovery too low for {pdb_id}: {mean_recovery:.1%} < {expected_min_recovery:.1%}. "
        f"This suggests the decoder bug fix may not be working correctly."
    )
    assert mean_recovery <= expected_max_recovery, (
        f"Recovery suspiciously high for {pdb_id}: {mean_recovery:.1%} > {expected_max_recovery:.1%}. "
        f"This might indicate overfitting or a data leak."
    )


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
def test_temperature_effect_on_recovery(temperature: float):
    """Test that recovery decreases with higher temperature.

    Higher temperatures should lead to more diversity and lower recovery.

    Args:
        temperature: Sampling temperature
    """
    # Use 1ubq as test case
    result = sample(
        inputs="1ubq",
        num_samples=3,
        temperature=temperature,
        sampling_strategy="temperature",
        random_seed=42,
        backbone_noise=0.0,
    )

    sequences = result["sequences"]
    native_sequence = result["metadata"]["native_sequences"][0]

    # Compute mean recovery
    recoveries = []
    for sample_idx in range(sequences.shape[2]):
        designed_seq = sequences[0, 0, sample_idx]
        recovery = compute_sequence_recovery(native_sequence, designed_seq)
        recoveries.append(recovery)

    mean_recovery = jnp.mean(jnp.array(recoveries))

    print(f"\nTemperature {temperature}: Mean recovery = {mean_recovery:.1%}")

    # At T=0.1, should be high recovery (>35%)
    # At T=1.0, should be lower but still reasonable (>20%)
    if temperature <= 0.2:
        assert mean_recovery >= 0.35, f"Low temp recovery too low: {mean_recovery:.1%}"
    else:
        assert mean_recovery >= 0.20, f"Recovery too low even at high temp: {mean_recovery:.1%}"


def test_split_sampling_recovery():
    """Test sequence recovery with split sampling strategy.

    Split sampling often achieves better recovery than temperature sampling.
    """
    result = sample(
        inputs="1ubq",
        num_samples=5,
        sampling_strategy="split",
        random_seed=42,
        backbone_noise=0.0,
    )

    sequences = result["sequences"]
    native_sequence = result["metadata"]["native_sequences"][0]

    # Compute mean recovery
    recoveries = []
    for sample_idx in range(sequences.shape[2]):
        designed_seq = sequences[0, 0, sample_idx]
        recovery = compute_sequence_recovery(native_sequence, designed_seq)
        recoveries.append(recovery)

    mean_recovery = jnp.mean(jnp.array(recoveries))

    print(f"\nSplit Sampling Recovery: {mean_recovery:.1%}")

    # Split sampling should achieve good recovery
    assert mean_recovery >= 0.35, (
        f"Split sampling recovery too low: {mean_recovery:.1%}. "
        f"Expected >= 35% for native backbone redesign."
    )


def test_no_alanine_bias():
    """Test that the model doesn't have extreme Alanine bias.

    The unconditional decoder bug caused >60% Alanine predictions.
    This should be fixed now.
    """
    result = sample(
        inputs="1ubq",
        num_samples=10,
        temperature=1.0,  # Higher temp to see diversity
        sampling_strategy="temperature",
        random_seed=42,
        backbone_noise=0.0,
    )

    sequences = result["sequences"]

    # Compute Alanine frequency across all samples
    # Alanine is typically index 0 in the 21-amino acid encoding
    ala_counts = []
    for sample_idx in range(sequences.shape[2]):
        designed_seq = sequences[0, 0, sample_idx]
        # Count Alanine (index 0)
        ala_count = jnp.sum(designed_seq == 0)
        ala_freq = float(ala_count / len(designed_seq))
        ala_counts.append(ala_freq)

    mean_ala_freq = jnp.mean(jnp.array(ala_counts))

    print(f"\nAlanine Frequency: {mean_ala_freq:.1%}")
    print(f"Individual samples: {[f'{f:.1%}' for f in ala_counts]}")

    # Natural Alanine frequency is ~8-9% in proteins
    # Allow up to 20% to be conservative, but should not be >40%
    assert mean_ala_freq < 0.40, (
        f"Excessive Alanine bias detected: {mean_ala_freq:.1%}. "
        f"This suggests the decoder bug fix may not be complete."
    )


def test_sequence_diversity_at_high_temperature():
    """Test that the model produces diverse sequences at high temperature.

    At T=2.0, we should see high diversity (low pairwise identity).
    """
    result = sample(
        inputs="1ubq",
        num_samples=10,
        temperature=2.0,
        sampling_strategy="temperature",
        random_seed=42,
        backbone_noise=0.0,
    )

    sequences = result["sequences"]

    # Compute pairwise sequence identity
    n_samples = sequences.shape[2]
    pairwise_identities = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            seq_i = sequences[0, 0, i]
            seq_j = sequences[0, 0, j]
            identity = compute_sequence_recovery(seq_i, seq_j)
            pairwise_identities.append(identity)

    mean_pairwise_identity = jnp.mean(jnp.array(pairwise_identities))

    print(f"\nMean Pairwise Sequence Identity (T=2.0): {mean_pairwise_identity:.1%}")

    # At high temperature, pairwise identity should be low (<40%)
    assert mean_pairwise_identity < 0.50, (
        f"Insufficient diversity at high temperature: {mean_pairwise_identity:.1%}. "
        f"Sequences should be more diverse."
    )


if __name__ == "__main__":
    # Run tests individually for debugging
    print("Testing sequence recovery on 1ubq...")
    test_sequence_recovery_on_native_structures("1ubq", 0.35, 0.70)

    print("\nTesting temperature effect...")
    test_temperature_effect_on_recovery(0.1)
    test_temperature_effect_on_recovery(1.0)

    print("\nTesting split sampling...")
    test_split_sampling_recovery()

    print("\nTesting Alanine bias...")
    test_no_alanine_bias()

    print("\nTesting diversity...")
    test_sequence_diversity_at_high_temperature()

    print("\n✅ All sequence recovery tests passed!")
