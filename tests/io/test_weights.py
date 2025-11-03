import equinox as eqx
import os
import pytest
from prxteinmpnn.io.weights import load_weights

class TinyModel(eqx.Module):
    a: float

def test_load_weights_as_pytree(tmp_path):
    """Test loading weights as a raw PyTree (inference mode)."""
    # Create a tiny .eqx file
    weights = {"a": 1.0}
    eqx.tree_serialise_leaves(tmp_path / "tiny_model.eqx", weights)
    result = load_weights(local_path=str(tmp_path / "tiny_model.eqx"), skeleton=None)
    assert isinstance(result, dict)
    assert result["a"] == 1.0

def test_load_weights_into_skeleton(tmp_path):
    """Test loading weights into an eqx.Module skeleton (training mode)."""
    weights = {"a": 1.0}
    eqx.tree_serialise_leaves(tmp_path / "tiny_model.eqx", weights)
    skeleton = TinyModel(a=0.0)
    model = load_weights(local_path=str(tmp_path / "tiny_model.eqx"), skeleton=skeleton)
    assert isinstance(model, TinyModel)
    assert model.a == 1.0

@pytest.mark.slow
def test_huggingface_download(monkeypatch):
    """Integration test: download weights from Hugging Face Hub."""
    # Patch hf_hub_download to avoid real network call
    called = {}
    def fake_hf_hub_download(repo_id, filename):
        called["repo_id"] = repo_id
        called["filename"] = filename
        # Use a local file for the test
        eqx.tree_serialise_leaves("/tmp/fake_model.eqx", {"a": 42.0})
        return "/tmp/fake_model.eqx"
    monkeypatch.setattr("prxteinmpnn.io.weights.hf_hub_download", fake_hf_hub_download)
    result = load_weights(model_version="v_48_020", model_weights="original", skeleton=None)
    assert isinstance(result, dict)
    assert result["a"] == 42.0
    assert called["repo_id"] == "maraxen/prxteinmpnn"
    assert called["filename"] == "original/v_48_020.eqx"
