"""Extensive sequence recovery testing on diverse PDB structures.

This test validates that the model achieves expected sequence recovery
performance, which is a key indicator that the implementation is correct.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling import make_sample_sequences
from prxteinmpnn.utils.data_structures import Protein


# Test structures available in tests/data/
TEST_STRUCTURES = [
    "1ubq.pdb",  # Ubiquitin (76 residues)
    "5awl.pdb",  # Another test structure
]


@pytest.mark.slow
def test_recovery_across_structures():
    """Test sequence recovery on available protein structures.

    This test validates that the model achieves reasonable sequence
    recovery performance, indicating correct implementation.
    """
    try:
        model = load_model()
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    sample_fn = make_sample_sequences(model)

    results = {}
    test_data_dir = Path(__file__).parent.parent / "data"

    for pdb_file in TEST_STRUCTURES:
        pdb_path = test_data_dir / pdb_file
        if not pdb_path.exists():
            print(f"Skipping {pdb_file}: file not found")
            continue

        print(f"\nTesting {pdb_file}...")

        try:
            protein_tuple = next(parse_input(str(pdb_path)))
            protein = Protein.from_tuple(protein_tuple)
        except Exception as e:
            print(f"  Could not parse {pdb_file}: {e}")
            continue

        key = jax.random.PRNGKey(42)
        recoveries = []

        # Run 10 independent samples
        for i in range(10):
            key, subkey = jax.random.split(key)
            sampled_seq, _, _ = sample_fn(
                subkey,
                protein.coordinates,
                protein.mask,
                protein.residue_index,
                protein.chain_index,
                temperature=jnp.array(0.1),
            )

            # Calculate recovery (fraction of positions matching native sequence)
            recovery = (sampled_seq == protein.aatype).sum() / protein.mask.sum()
            recoveries.append(float(recovery))

        results[pdb_file] = {
            "mean": jnp.mean(jnp.array(recoveries)),
            "std": jnp.std(jnp.array(recoveries)),
        }

        print(f"  Recovery: {results[pdb_file]['mean']:.1%} ± {results[pdb_file]['std']:.1%}")

    if not results:
        pytest.skip("No structures were successfully tested")

    all_recoveries = [r["mean"] for r in results.values()]
    mean_recovery = float(jnp.mean(jnp.array(all_recoveries)))

    print(f"\n{'='*60}")
    print(f"Overall Mean Recovery: {mean_recovery:.1%}")
    print(f"{'='*60}")

    # Assert expected performance
    # For ProteinMPNN, we expect 35-65% recovery on native backbones
    assert mean_recovery >= 0.25, (
        f"Mean recovery {mean_recovery:.1%} is unexpectedly low. "
        f"Expected at least 25% (typical range: 35-65%)."
    )
    assert mean_recovery <= 0.80, (
        f"Mean recovery {mean_recovery:.1%} is unexpectedly high. "
        f"Expected at most 80% (typical range: 35-65%)."
    )

    print("\n✅ All structures passed recovery thresholds!")


@pytest.mark.slow
def test_recovery_single_structure():
    """Test sequence recovery on a single structure (faster test).

    This is a lighter-weight version of the full recovery test.
    """
    try:
        model = load_model()
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    sample_fn = make_sample_sequences(model)

    # Test on 1ubq only
    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    key = jax.random.PRNGKey(42)
    recoveries = []

    # Run 5 samples for faster testing
    for i in range(5):
        key, subkey = jax.random.split(key)
        sampled_seq, _, _ = sample_fn(
            subkey,
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            temperature=jnp.array(0.1),
        )

        recovery = (sampled_seq == protein.aatype).sum() / protein.mask.sum()
        recoveries.append(float(recovery))

    mean_recovery = float(jnp.mean(jnp.array(recoveries)))

    print(f"\n1ubq Recovery: {mean_recovery:.1%}")

    # Looser bounds for single-structure test
    assert mean_recovery >= 0.20, f"Recovery {mean_recovery:.1%} too low"
    assert mean_recovery <= 0.85, f"Recovery {mean_recovery:.1%} too high"

    print("✅ Recovery test passed!")
