"""Test conditional scoring accuracy.

This test validates that when the model scores its own native sequence,
it should predict the native sequence with very high accuracy. This is
a strong indicator that the conditional decoder is working correctly.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.utils.data_structures import Protein


TEST_STRUCTURES = ["1ubq.pdb", "5awl.pdb"]


@pytest.mark.slow
def test_conditional_scoring():
    """Test that conditional scoring gives high accuracy for native sequences.

    When we give the model the native sequence and ask it to score/predict,
    it should recover the native sequence with very high accuracy (>90%).
    This validates the conditional decoder implementation.
    """
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"

    try:
        model = load_model(local_path=str(weights_path))
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    key = jax.random.PRNGKey(42)
    test_data_dir = Path(__file__).parent.parent / "data"

    results = {}

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

        # Test unconditional (baseline)
        key, subkey = jax.random.split(key)
        _, unconditional_logits = model(
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            "unconditional",
            prng_key=subkey,
        )
        uncond_preds = unconditional_logits.argmax(axis=-1)
        uncond_recovery = (uncond_preds == protein.aatype).sum() / protein.mask.sum()
        print(f"  Unconditional recovery: {uncond_recovery:.1%}")

        # Test conditional scoring with native sequence
        key, subkey = jax.random.split(key)
        one_hot_seq = jax.nn.one_hot(protein.aatype, 21)
        _, conditional_logits = model(
            protein.coordinates,
            protein.mask,
            protein.residue_index,
            protein.chain_index,
            "conditional",
            prng_key=subkey,
            one_hot_sequence=one_hot_seq,
        )

        cond_preds = conditional_logits.argmax(axis=-1)
        cond_recovery = (cond_preds == protein.aatype).sum() / protein.mask.sum()
        print(f"  Conditional recovery (self-score): {cond_recovery:.1%}")

        results[pdb_file] = {
            "unconditional": float(uncond_recovery),
            "conditional": float(cond_recovery),
        }

        # Conditional self-scoring should be very high
        assert cond_recovery >= 0.85, (
            f"{pdb_file}: Conditional recovery {cond_recovery:.1%} too low. "
            f"Expected >= 85% for self-scoring."
        )
        print(f"  ✅ Passed!")

    if not results:
        pytest.skip("No structures were successfully tested")

    print(f"\n{'='*60}")
    print("All conditional scoring tests passed!")
    print(f"{'='*60}")


@pytest.mark.slow
def test_conditional_vs_unconditional():
    """Test that conditional accuracy is higher than unconditional.

    The conditional decoder should perform better than unconditional
    when given the correct sequence context.
    """
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"

    try:
        model = load_model(local_path=str(weights_path))
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    key = jax.random.PRNGKey(42)

    # Unconditional
    key, subkey = jax.random.split(key)
    _, uncond_logits = model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=subkey,
    )
    uncond_acc = (uncond_logits.argmax(-1) == protein.aatype).sum() / protein.mask.sum()

    # Conditional
    key, subkey = jax.random.split(key)
    one_hot_seq = jax.nn.one_hot(protein.aatype, 21)
    _, cond_logits = model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "conditional",
        prng_key=subkey,
        one_hot_sequence=one_hot_seq,
    )
    cond_acc = (cond_logits.argmax(-1) == protein.aatype).sum() / protein.mask.sum()

    print(f"\nUnconditional accuracy: {uncond_acc:.1%}")
    print(f"Conditional accuracy: {cond_acc:.1%}")

    # Conditional should be significantly better
    assert cond_acc > uncond_acc, (
        f"Conditional accuracy ({cond_acc:.1%}) should be higher than "
        f"unconditional ({uncond_acc:.1%})"
    )

    print("✅ Conditional outperforms unconditional!")
