"""Tests for side-chain context conditioning via build_inference_bundle + score_unconditional.

Covers the bundle-builder → GeometryBundle → encode.py → ProteinFeaturesLigand path
that score_unconditional.kernel uses (distinct from the score_conditional.kernel
kwargs-injection path tested in test_ligand_sidechain_context.py).

Three test cases:
  1. bundle_builder unit: atom_37/atom_37_mask shape-normalize correctly.
  2. integration (slow): logits differ when atom_37 changes with use_side_chains=True.
  3. guard: ValueError raised when use_side_chains=True model receives no atom_37 in bundle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference import score_unconditional
from aminx.model.ligand_mpnn import PrxteinLigandMPNN


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _synthetic_inputs(*, seq_len: int = 8, ligand_atoms: int = 6) -> dict[str, jax.Array]:
  rng = np.random.default_rng(42)
  return {
    "coords": jnp.asarray(rng.normal(size=(seq_len, 4, 3)).astype(np.float32)),
    "mask": jnp.ones((seq_len,), dtype=jnp.float32),
    "residue_index": jnp.arange(seq_len, dtype=jnp.int32),
    "chain_index": jnp.zeros((seq_len,), dtype=jnp.int32),
    "y": jnp.asarray(rng.normal(size=(seq_len, ligand_atoms, 3)).astype(np.float32)),
    "y_t": jnp.asarray(rng.integers(1, 30, size=(seq_len, ligand_atoms), dtype=np.int32)),
    "y_m": jnp.ones((seq_len, ligand_atoms), dtype=jnp.float32),
    "atom_37_a": jnp.asarray(rng.normal(size=(seq_len, 37, 3)).astype(np.float32)),
    "atom_37_b": jnp.asarray(rng.normal(size=(seq_len, 37, 3)).astype(np.float32) * 5.0),
    "atom_37_mask": jnp.ones((seq_len, 37), dtype=jnp.float32),
  }


def _small_ligand_model(*, use_side_chains: bool, key: jax.Array) -> PrxteinLigandMPNN:
  return PrxteinLigandMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=6,
    num_context_layers=2,
    dropout_rate=0.0,
    ligand_mpnn_use_side_chain_context=use_side_chains,
    key=key,
  )


# ---------------------------------------------------------------------------
# 1. Bundle builder unit test — shape normalization
# ---------------------------------------------------------------------------

def test_build_inference_bundle_atom_37_3d_normalizes_to_4d() -> None:
  """(L, 37, 3) atom_37 is expanded to (1, L, 37, 3) in GeometryBundle."""
  L = 10
  bundle, _ = build_inference_bundle(
    coords=jnp.zeros((L, 4, 3)),
    mask=jnp.ones(L),
    residue_index=jnp.arange(L, dtype=jnp.int32),
    chain_index=jnp.zeros(L, dtype=jnp.int32),
    atom_37=jnp.zeros((L, 37, 3)),
    atom_37_mask=jnp.ones((L, 37)),
  )
  assert bundle.geometry.atom_37 is not None
  assert bundle.geometry.atom_37.shape == (1, L, 37, 3)
  assert bundle.geometry.atom_37_mask is not None
  assert bundle.geometry.atom_37_mask.shape == (1, L, 37)


def test_build_inference_bundle_atom_37_4d_passthrough() -> None:
  """Already-(S, L, 37, 3) atom_37 is stored without adding a leading axis."""
  S, L = 2, 10
  bundle, _ = build_inference_bundle(
    coords=jnp.zeros((S, L, 4, 3)),
    mask=jnp.ones((S, L)),
    residue_index=jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None], (S, L)),
    chain_index=jnp.zeros((S, L), dtype=jnp.int32),
    atom_37=jnp.zeros((S, L, 37, 3)),
    atom_37_mask=jnp.ones((S, L, 37)),
  )
  assert bundle.geometry.atom_37.shape == (S, L, 37, 3)
  assert bundle.geometry.atom_37_mask.shape == (S, L, 37)


def test_build_inference_bundle_no_atom_37_leaves_none() -> None:
  """Omitting atom_37 leaves GeometryBundle.atom_37 as None."""
  L = 8
  bundle, _ = build_inference_bundle(
    coords=jnp.zeros((L, 4, 3)),
    mask=jnp.ones(L),
    residue_index=jnp.arange(L, dtype=jnp.int32),
    chain_index=jnp.zeros(L, dtype=jnp.int32),
  )
  assert bundle.geometry.atom_37 is None
  assert bundle.geometry.atom_37_mask is None


# ---------------------------------------------------------------------------
# 2. Integration: logits change with different atom_37 (use_side_chains=True)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_score_unconditional_side_chain_context_changes_logits() -> None:
  """Logits from score_unconditional differ when atom_37 changes, given use_side_chains=True.

  The encode.py path computes chain_mask = 1 - fixed_mask, then gates atom_37_mask by
  (1 - chain_mask_in).  Residues that are designable (fixed_mask=0 → chain_mask=1) have
  their side-chain context zeroed out — only fixed residues contribute.  We therefore set
  fixed_mask=ones so all residues are fixed and side-chain coordinates flow through.
  """
  inp = _synthetic_inputs()
  seq_len = inp["coords"].shape[0]
  model = _small_ligand_model(use_side_chains=True, key=jax.random.PRNGKey(0))
  stage_set = make_stage_set()
  key = jax.random.PRNGKey(1)

  def _run(atom_37: jax.Array) -> jax.Array:
    bundle, config = build_inference_bundle(
      coords=inp["coords"],
      mask=inp["mask"],
      residue_index=inp["residue_index"],
      chain_index=inp["chain_index"],
      ligand_coords=inp["y"],
      ligand_atom_types=inp["y_t"],
      ligand_mask=inp["y_m"],
      atom_37=atom_37,
      atom_37_mask=inp["atom_37_mask"],
      # All residues fixed so chain_mask=0 and atom_37_mask is not zeroed.
      fixed_mask=jnp.ones(seq_len, dtype=jnp.float32),
      mode="score_conditional",
      inference=True,
    )
    return score_unconditional.kernel(model, key, bundle, config, stage_set)

  logits_a = _run(inp["atom_37_a"])
  logits_b = _run(inp["atom_37_b"])

  assert logits_a.shape == logits_b.shape
  assert bool(jnp.all(jnp.isfinite(logits_a)))
  assert bool(jnp.all(jnp.isfinite(logits_b)))
  # Side-chain coordinates differ substantially; logits must differ
  assert not np.allclose(np.asarray(logits_a), np.asarray(logits_b), atol=1e-4), (
    "Expected logits to differ when atom_37 changes with use_side_chains=True"
  )


@pytest.mark.slow
def test_score_unconditional_side_chain_context_no_effect_when_gate_off() -> None:
  """Logits are unchanged when atom_37 changes but use_side_chains=False.

  Uses fixed_mask=ones (same as the positive test above) so conditions differ only in
  whether the model was constructed with ligand_mpnn_use_side_chain_context.
  """
  inp = _synthetic_inputs()
  seq_len = inp["coords"].shape[0]
  model = _small_ligand_model(use_side_chains=False, key=jax.random.PRNGKey(2))
  stage_set = make_stage_set()
  key = jax.random.PRNGKey(3)

  def _run(atom_37: jax.Array) -> jax.Array:
    bundle, config = build_inference_bundle(
      coords=inp["coords"],
      mask=inp["mask"],
      residue_index=inp["residue_index"],
      chain_index=inp["chain_index"],
      ligand_coords=inp["y"],
      ligand_atom_types=inp["y_t"],
      ligand_mask=inp["y_m"],
      atom_37=atom_37,
      atom_37_mask=inp["atom_37_mask"],
      fixed_mask=jnp.ones(seq_len, dtype=jnp.float32),
      mode="score_conditional",
      inference=True,
    )
    return score_unconditional.kernel(model, key, bundle, config, stage_set)

  logits_a = _run(inp["atom_37_a"])
  logits_b = _run(inp["atom_37_b"])

  np.testing.assert_allclose(
    np.asarray(logits_a), np.asarray(logits_b), atol=1e-5,
    err_msg="Logits should be identical when use_side_chains=False regardless of atom_37",
  )


# ---------------------------------------------------------------------------
# 3. Guard: missing atom_37 raises when use_side_chains=True
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_score_unconditional_missing_atom_37_raises_when_use_side_chains() -> None:
  """score_unconditional raises ValueError when model needs atom_37 but bundle has none."""
  inp = _synthetic_inputs()
  model = _small_ligand_model(use_side_chains=True, key=jax.random.PRNGKey(4))
  stage_set = make_stage_set()

  bundle, config = build_inference_bundle(
    coords=inp["coords"],
    mask=inp["mask"],
    residue_index=inp["residue_index"],
    chain_index=inp["chain_index"],
    ligand_coords=inp["y"],
    ligand_atom_types=inp["y_t"],
    ligand_mask=inp["y_m"],
    fixed_mask=jnp.ones(inp["coords"].shape[0], dtype=jnp.float32),
    mode="score_conditional",
    inference=True,
    # No atom_37 / atom_37_mask
  )

  with pytest.raises(ValueError, match="atom_37"):
    score_unconditional.kernel(model, jax.random.PRNGKey(5), bundle, config, stage_set)
