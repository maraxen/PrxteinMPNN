"""Tests for PottsMPNN checkpoint conversion to Equinox format."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import zstandard as zstd


def test_etab_to_dense_h_j_w_synthetic_2residue():
  """Test etab_to_dense_h_j_w with synthetic 2-residue case.

  Hand-crafted h, J with known log_unnormalized — verify x2 scale is preserved.
  """
  from scripts.recapture.pottsmpnn_to_eqx import etab_to_dense_h_j_w

  # Synthetic 2-residue case: q=22 (standard Potts vocab)
  # etab shape: (1, 2, K, 22, 22) where K=2 (two neighbors per residue)

  # h_ref and j_ref are what we expect AFTER 2x scaling and symmetrization
  h_ref = np.array([
    [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  ], dtype=np.float32)

  # Build j_ref: AFTER etab[0,1] contribution is symmetrized with etab[1,0]
  # Both slots point to the same interaction, so they get averaged, then doubled
  j_raw = np.zeros((2, 2, 22, 22), dtype=np.float64)
  j_raw[0, 1, 0, 0] = 0.1
  j_raw[0, 1, 0, 1] = 0.05
  j_raw[0, 1, 1, 0] = 0.02
  j_raw[0, 1, 1, 1] = 0.03

  j_raw[1, 0, 0, 0] = 0.06  # Different pre-symmetry
  j_raw[1, 0, 0, 1] = 0.02
  j_raw[1, 0, 1, 0] = 0.04
  j_raw[1, 0, 1, 1] = 0.01

  # Apply symmetrization: j = 0.5 * (j + transpose)
  j_sym = 0.5 * (j_raw + np.transpose(j_raw, (1, 0, 3, 2)))
  # Then 2x scale
  j_ref = (2.0 * j_sym).astype(np.float32)

  w_ref = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

  # Build etab (pre-scaled by 2) and e_idx
  # Each residue has K=2 slots: one for self, one for neighbor
  etab = np.zeros((1, 2, 2, 22, 22), dtype=np.float64)
  e_idx = np.zeros((1, 2, 2), dtype=np.int64)

  # Residue 0:
  #   Slot 0: self-loop (neighbor = residue 0)
  etab[0, 0, 0, :, :] = np.diag(h_ref[0] / 2.0)  # Pre-scale
  e_idx[0, 0, 0] = 0

  #   Slot 1: interaction with residue 1
  etab[0, 0, 1, :, :] = j_raw[0, 1]  # Raw, will be symmetrized
  e_idx[0, 0, 1] = 1

  # Residue 1:
  #   Slot 0: interaction with residue 0
  etab[0, 1, 0, :, :] = j_raw[1, 0]  # Raw, will be symmetrized
  e_idx[0, 1, 0] = 0

  #   Slot 1: self-loop
  etab[0, 1, 1, :, :] = np.diag(h_ref[1] / 2.0)  # Pre-scale
  e_idx[0, 1, 1] = 1

  mask = np.ones((1, 2), dtype=np.float64)

  h_out, j_out, w_out = etab_to_dense_h_j_w(etab, e_idx, mask)

  # Verify shape
  assert h_out.shape == (2, 22), f"Expected (2, 22), got {h_out.shape}"
  assert j_out.shape == (2, 2, 22, 22), f"Expected (2, 2, 22, 22), got {j_out.shape}"
  assert w_out.shape == (2, 2), f"Expected (2, 2), got {w_out.shape}"

  # Verify values within tolerance (1e-6 as spec'd)
  np.testing.assert_allclose(h_out, h_ref, atol=1e-6, rtol=1e-6,
                             err_msg="h values do not match reference")
  np.testing.assert_allclose(j_out, j_ref, atol=1e-6, rtol=1e-6,
                             err_msg="j values do not match reference")
  np.testing.assert_allclose(w_out, w_ref, atol=1e-6, rtol=1e-6,
                             err_msg="w values do not match reference")


def test_save_potts_checkpoint_format():
  """Test that potts checkpoint is saved in .eqx.zst format with metadata."""
  from scripts.recapture.pottsmpnn_to_eqx import save_potts_checkpoint

  with tempfile.TemporaryDirectory() as tmpdir:
    outpath = Path(tmpdir) / "test.eqx.zst"

    # Create a minimal test pytree
    @dataclass
    class SimplePottsModel(eqx.Module):
      h: jnp.ndarray
      j: jnp.ndarray
      k_neighbors: int

    model = SimplePottsModel(
      h=jnp.ones((2, 22), dtype=jnp.float32),
      j=jnp.ones((2, 2, 22, 22), dtype=jnp.float32),
      k_neighbors=48,
    )

    metadata = {
      "k_neighbors": 48,
      "n_residues": 2,
      "vocab_size": 22,
    }

    save_potts_checkpoint(model, outpath, metadata)

    # Verify file exists and is zstd compressed
    assert outpath.exists(), f"Checkpoint not created at {outpath}"
    with open(outpath, "rb") as f:
      header = f.read(4)
      assert header == b"\x28\xb5\x2f\xfd", "File is not zstd compressed (wrong magic number)"

    # Verify can decompress
    with open(outpath, "rb") as f:
      dctx = zstd.ZstdDecompressor()
      content = dctx.decompress(f.read())
      assert len(content) > 0, "Decompressed content is empty"


def test_k_neighbors_from_model_config():
  """Test that k_neighbors is read from loaded model config, not CLI args."""
  # This will be a unit test for the parameter extraction logic
  from scripts.recapture.pottsmpnn_to_eqx import extract_k_neighbors_from_config

  config = {"k_neighbors": 48}
  k = extract_k_neighbors_from_config(config)
  assert k == 48, f"Expected k_neighbors=48, got {k}"

  # Missing k_neighbors should raise
  with pytest.raises(ValueError, match="k_neighbors"):
    extract_k_neighbors_from_config({})
