"""Direct comparison tests between PrxteinMPNN and ColabDesign.

This test directly compares the outputs of PrxteinMPNN against the reference
ColabDesign implementation to identify any discrepancies.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add ColabDesign to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ColabDesign"))

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn_model
from prxteinmpnn.utils.data_structures import Protein


@pytest.mark.slow
def test_comparison_with_colabdesign():
    """Compare PrxteinMPNN unconditional logits with ColabDesign reference."""
    # Load ColabDesign model
    try:
        from colabdesign.mpnn.model import mk_mpnn_model
    except ImportError as e:
        pytest.skip(f"ColabDesign not available: {e}")

    # Load models
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"
    prxtein_model = load_prxteinmpnn_model(local_path=str(weights_path))

    colab_model = mk_mpnn_model(model_name="v_48_020", weights="original")

    # Load test structure
    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    # Prepare inputs for PrxteinMPNN
    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    # Prepare inputs for ColabDesign
    colab_model.prep_inputs(pdb_filename=str(pdb_path))

    # Get unconditional logits from PrxteinMPNN
    key = jax.random.PRNGKey(0)
    _, prxtein_logits = prxtein_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=key,
    )

    # Get unconditional logits from ColabDesign
    colab_logits = colab_model.get_unconditional_logits()

    print(f"\nPrxteinMPNN logits shape: {prxtein_logits.shape}")
    print(f"ColabDesign logits shape: {colab_logits.shape}")

    # Compare predictions
    prxtein_preds = prxtein_logits.argmax(axis=-1)
    colab_preds = colab_logits.argmax(axis=-1)

    # Calculate agreement
    agreement = (prxtein_preds == colab_preds).sum() / protein.mask.sum()
    print(f"\nPrediction agreement: {agreement:.1%}")

    # Compare to native sequence
    prxtein_recovery = (prxtein_preds == protein.aatype).sum() / protein.mask.sum()
    colab_recovery = (colab_preds == protein.aatype).sum() / protein.mask.sum()

    print(f"PrxteinMPNN recovery: {prxtein_recovery:.1%}")
    print(f"ColabDesign recovery: {colab_recovery:.1%}")

    # Check logits correlation
    prxtein_flat = prxtein_logits.reshape(-1)
    colab_flat = colab_logits.reshape(-1)
    correlation = np.corrcoef(np.array(prxtein_flat), np.array(colab_flat))[0, 1]
    print(f"Logits correlation: {correlation:.4f}")

    # Compare first position logits in detail
    print(f"\nFirst position logits comparison:")
    print(f"PrxteinMPNN: {prxtein_logits[0, :10]}")
    print(f"ColabDesign: {colab_logits[0, :10]}")

    # Assert high agreement expected
    assert agreement >= 0.80, (
        f"Prediction agreement {agreement:.1%} too low. "
        f"Expected >= 80% agreement with ColabDesign."
    )

    print("\n✅ Comparison test passed!")


@pytest.mark.slow
def test_conditional_comparison_with_colabdesign():
    """Compare PrxteinMPNN conditional scoring with ColabDesign."""
    try:
        from colabdesign.mpnn.model import mk_mpnn_model
    except ImportError as e:
        pytest.skip(f"ColabDesign not available: {e}")

    # Load models
    weights_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/io/weights/original_v_48_020.eqx"
    prxtein_model = load_prxteinmpnn_model(local_path=str(weights_path))

    colab_model = mk_mpnn_model(model_name="v_48_020", weights="original")

    # Load test structure
    test_data_dir = Path(__file__).parent.parent / "data"
    pdb_path = test_data_dir / "1ubq.pdb"

    if not pdb_path.exists():
        pytest.skip("Test data file 1ubq.pdb not found")

    # Prepare inputs
    protein_tuple = next(parse_input(str(pdb_path)))
    protein = Protein.from_tuple(protein_tuple)

    colab_model.prep_inputs(pdb_filename=str(pdb_path))

    # Get conditional logits from PrxteinMPNN (score native sequence)
    key = jax.random.PRNGKey(0)
    one_hot_seq = jax.nn.one_hot(protein.aatype, 21)
    _, prxtein_cond_logits = prxtein_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "conditional",
        prng_key=key,
        one_hot_sequence=one_hot_seq,
    )

    # Get conditional logits from ColabDesign (score native sequence)
    colab_output = colab_model.score(seq=None)
    colab_cond_logits = colab_output["logits"]

    # Compare predictions
    prxtein_preds = prxtein_cond_logits.argmax(axis=-1)
    colab_preds = colab_cond_logits.argmax(axis=-1)

    # Recovery from native
    prxtein_recovery = (prxtein_preds == protein.aatype).sum() / protein.mask.sum()
    colab_recovery = (colab_preds == protein.aatype).sum() / protein.mask.sum()

    print(f"\nConditional self-scoring recovery:")
    print(f"PrxteinMPNN: {prxtein_recovery:.1%}")
    print(f"ColabDesign: {colab_recovery:.1%}")

    # Agreement between implementations
    agreement = (prxtein_preds == colab_preds).sum() / protein.mask.sum()
    print(f"Prediction agreement: {agreement:.1%}")

    # Both should have high recovery for self-scoring
    print(f"\nExpected: Both should have >85% recovery for conditional self-scoring")

    if colab_recovery >= 0.85:
        print(f"✅ ColabDesign achieves expected performance: {colab_recovery:.1%}")
    else:
        print(f"❌ ColabDesign below expected: {colab_recovery:.1%}")

    if prxtein_recovery >= 0.85:
        print(f"✅ PrxteinMPNN achieves expected performance: {prxtein_recovery:.1%}")
    else:
        print(f"❌ PrxteinMPNN below expected: {prxtein_recovery:.1%}")

    # The agreement should be high
    assert agreement >= 0.80, (
        f"Agreement {agreement:.1%} too low between implementations"
    )
