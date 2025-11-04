"""Tests for the updated prep.py with architecture adapter."""

from prxteinmpnn.eqx_new import PrxteinMPNN
from prxteinmpnn.run.prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import SamplingSpecification


class TestPrepWithAdapter:
  """Test that prep.py works with the new architecture adapter.

  Note: Legacy architecture tests are skipped because .pkl files no longer exist
  on HuggingFace. Only .eqx format is available now.
  """

  def test_prep_new_architecture(self):
    """Test that prep returns PrxteinMPNN with use_new_architecture=True."""
    # Create a minimal spec for testing
    spec = SamplingSpecification(
      inputs=["1ubq"],  # Use a simple PDB ID
      model_version="v_48_020",
      model_weights="original",
      batch_size=1,
    )

    protein_iterator, model = prep_protein_stream_and_model(spec, use_new_architecture=True)

    # Verify it's an Equinox PrxteinMPNN instance
    assert isinstance(model, PrxteinMPNN), f"Expected PrxteinMPNN, got {type(model)}"
    print(f"✓ New architecture: {type(model)}")

  def test_prep_new_architecture_all_versions(self):
    """Test that all model versions work through prep."""
    versions = ["v_48_002", "v_48_020"]  # Test a couple versions

    for version in versions:
      spec = SamplingSpecification(
        inputs=["1ubq"],
        model_version=version,
        model_weights="original",
        batch_size=1,
      )

      protein_iterator, model = prep_protein_stream_and_model(
        spec, use_new_architecture=True
      )

      assert isinstance(model, PrxteinMPNN), f"Failed for {version}: got {type(model)}"
      print(f"✓ Loaded {version} through prep.py")
