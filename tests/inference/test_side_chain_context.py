"""Acceptance tests for side-chain context fix (bug fix + load_model flag).

Spec: .praxia/docs/specs/260622_sc-encode-vmap-and-load-model-flag.md
Task: 260622_sc-fix

Five acceptance criteria:
  1. load_model with use_side_chain_context=True succeeds; flag is True on model.
     Default (no kwarg) stays False.
  2. score_unconditional.kernel (vmap path) and ScanEncode path run with
     atom_37/atom_37_mask and S>=2 WITHOUT the broadcast error.
  3. Differential — SC changes output: logits with reveal mask (fixed_mask=1 at a
     subset) differ from fixed_mask=0 everywhere (max|Delta|>1e-3), and
     SC-on-with-no-reveal ≈ SC-off (max|Delta|<1e-5).
  4. Ligand differential pinned: ligand-on vs ligand-off logits differ max|Delta|>1e-3.
  5. No regressions (covered by uv run pytest).
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference import score_unconditional
from aminx.inference.encode import make_encode_fn
from aminx.model.ligand_mpnn import PrxteinLigandMPNN
from aminx.types.configs import InferenceConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKPOINT = "ligandmpnn_v_32_020_25"


def _small_ligand_model(*, use_sc: bool, key: jax.Array) -> PrxteinLigandMPNN:
    """Construct a tiny ligand model (no weights needed — random init)."""
    return PrxteinLigandMPNN(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=6,
        num_context_layers=2,
        dropout_rate=0.0,
        ligand_mpnn_use_side_chain_context=use_sc,
        key=key,
    )


def _synthetic_inputs(*, seq_len: int = 8, ligand_atoms: int = 4, n_states: int = 2):
    """Return minimal synthetic inputs for multistate (S>=2) tests."""
    rng = np.random.default_rng(31415)
    coords_1 = rng.normal(size=(seq_len, 4, 3)).astype(np.float32)
    # Second state: slight perturbation
    coords_2 = coords_1 + rng.normal(size=(seq_len, 4, 3)).astype(np.float32) * 0.1
    coords = np.stack([coords_1] + [coords_2] * (n_states - 1), axis=0)  # (S, L, 4, 3)

    atom_37 = rng.normal(size=(n_states, seq_len, 37, 3)).astype(np.float32)
    atom_37_mask = np.ones((n_states, seq_len, 37), dtype=np.float32)

    lig_coords = rng.normal(size=(n_states, seq_len, ligand_atoms, 3)).astype(np.float32)
    lig_types = rng.integers(1, 30, size=(n_states, seq_len, ligand_atoms)).astype(np.int32)
    lig_mask = np.ones((n_states, seq_len, ligand_atoms), dtype=np.float32)

    return {
        "coords": jnp.asarray(coords),
        "mask": jnp.ones((n_states, seq_len), dtype=jnp.float32),
        "residue_index": jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :], (n_states, seq_len)
        ),
        "chain_index": jnp.zeros((n_states, seq_len), dtype=jnp.int32),
        "atom_37": jnp.asarray(atom_37),
        "atom_37_mask": jnp.asarray(atom_37_mask),
        "lig_coords": jnp.asarray(lig_coords),
        "lig_types": jnp.asarray(lig_types),
        "lig_mask": jnp.asarray(lig_mask),
        "seq_len": seq_len,
        "n_states": n_states,
    }


# ---------------------------------------------------------------------------
# Criterion 1: load_model flag
# ---------------------------------------------------------------------------

@pytest.mark.requires_weights
def test_load_model_sc_flag_true() -> None:
    """load_model(use_side_chain_context=True) succeeds; model flag is True."""
    from aminx.io.weights import load_model

    model = load_model(_CHECKPOINT, use_side_chain_context=True)
    assert model.ligand_mpnn_use_side_chain_context is True


@pytest.mark.requires_weights
def test_load_model_sc_flag_default_false() -> None:
    """load_model() without use_side_chain_context defaults to False."""
    from aminx.io.weights import load_model

    model = load_model(_CHECKPOINT)
    assert model.ligand_mpnn_use_side_chain_context is False


# ---------------------------------------------------------------------------
# Criterion 2: multistate SC kernel runs without broadcast error
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vmap_encode_sc_multistate_no_crash() -> None:
    """_VmapEncode (score_unconditional.kernel) runs with SC and S>=2 without broadcast error."""
    inp = _synthetic_inputs(seq_len=8, ligand_atoms=4, n_states=2)
    model = _small_ligand_model(use_sc=True, key=jax.random.PRNGKey(0))
    stage_set = make_stage_set()

    bundle, config = build_inference_bundle(
        coords=inp["coords"],
        mask=inp["mask"],
        residue_index=inp["residue_index"],
        chain_index=inp["chain_index"],
        ligand_coords=inp["lig_coords"],
        ligand_atom_types=inp["lig_types"],
        ligand_mask=inp["lig_mask"],
        atom_37=inp["atom_37"],
        atom_37_mask=inp["atom_37_mask"],
        fixed_mask=jnp.ones(inp["seq_len"], dtype=jnp.float32),
        mode="score_conditional",
        inference=True,
    )
    logits = score_unconditional.kernel(model, jax.random.PRNGKey(1), bundle, config, stage_set)
    assert logits is not None
    assert bool(jnp.all(jnp.isfinite(logits)))


@pytest.mark.slow
def test_scan_encode_sc_multistate_no_crash() -> None:
    """_ScanEncode (use_rolling_state=True) runs with SC and S>=2 without broadcast error."""
    inp = _synthetic_inputs(seq_len=8, ligand_atoms=4, n_states=2)
    model = _small_ligand_model(use_sc=True, key=jax.random.PRNGKey(2))
    stage_set = make_stage_set()

    bundle, config = build_inference_bundle(
        coords=inp["coords"],
        mask=inp["mask"],
        residue_index=inp["residue_index"],
        chain_index=inp["chain_index"],
        ligand_coords=inp["lig_coords"],
        ligand_atom_types=inp["lig_types"],
        ligand_mask=inp["lig_mask"],
        atom_37=inp["atom_37"],
        atom_37_mask=inp["atom_37_mask"],
        fixed_mask=jnp.ones(inp["seq_len"], dtype=jnp.float32),
        mode="score_conditional",
        inference=True,
    )
    # Exercise _ScanEncode directly
    encode_fn = make_encode_fn(model, use_rolling_state=True)
    enc = encode_fn(bundle, jax.random.PRNGKey(3), config)
    assert enc is not None
    assert bool(jnp.all(jnp.isfinite(enc.node_features)))


# ---------------------------------------------------------------------------
# Criterion 3: differential — SC changes output when reveal mask is set
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sc_differential_reveal_vs_no_reveal() -> None:
    """Core gate: logits with SC reveal differ from no-reveal; SC-on+no-reveal ≈ SC-off.

    reveal = fixed_mask=1 at half of residues (those side chains are visible).
    no-reveal = fixed_mask=0 everywhere (all residues are designable; chain_mask=1;
                atom_37_mask zeroed out by (1-chain_mask_in), so SC has no effect).
    SC-off = model built without use_sc; atom_37 provided but model ignores it.

    Expected:
      max|logits_reveal - logits_no_reveal| > 1e-3  (SC is live when fixed)
      max|logits_sc_no_reveal - logits_off_no_reveal| < 1e-5  (SC-on+no-reveal ≈ SC-off)
    """
    inp = _synthetic_inputs(seq_len=8, ligand_atoms=4, n_states=2)
    seq_len = inp["seq_len"]
    stage_set = make_stage_set()
    key = jax.random.PRNGKey(42)

    model_sc_on = _small_ligand_model(use_sc=True, key=jax.random.PRNGKey(10))
    model_sc_off = _small_ligand_model(use_sc=False, key=jax.random.PRNGKey(10))

    # Reveal mask: first half of residues are fixed (fixed_mask=1 → chain_mask=0 → SC flows)
    reveal_mask = jnp.zeros(seq_len, dtype=jnp.float32).at[: seq_len // 2].set(1.0)
    # No-reveal mask: all residues designable (fixed_mask=0 → chain_mask=1 → SC zeroed)
    no_reveal_mask = jnp.zeros(seq_len, dtype=jnp.float32)

    def _run(model: PrxteinLigandMPNN, fixed_mask: jax.Array) -> jax.Array:
        bundle, config = build_inference_bundle(
            coords=inp["coords"],
            mask=inp["mask"],
            residue_index=inp["residue_index"],
            chain_index=inp["chain_index"],
            ligand_coords=inp["lig_coords"],
            ligand_atom_types=inp["lig_types"],
            ligand_mask=inp["lig_mask"],
            atom_37=inp["atom_37"],
            atom_37_mask=inp["atom_37_mask"],
            fixed_mask=fixed_mask,
            mode="score_conditional",
            inference=True,
        )
        return score_unconditional.kernel(model, key, bundle, config, stage_set)

    logits_reveal = _run(model_sc_on, reveal_mask)
    logits_no_reveal = _run(model_sc_on, no_reveal_mask)
    logits_off_no_reveal = _run(model_sc_off, no_reveal_mask)

    delta_reveal = float(np.abs(np.asarray(logits_reveal) - np.asarray(logits_no_reveal)).max())
    delta_sc_vs_off = float(np.abs(np.asarray(logits_no_reveal) - np.asarray(logits_off_no_reveal)).max())

    # Log the values so the caller can report them
    print(f"[SC differential] reveal vs no-reveal max|Δ| = {delta_reveal:.6f}")
    print(f"[SC differential] sc-on+no-reveal vs sc-off max|Δ| = {delta_sc_vs_off:.2e}")

    assert delta_reveal > 1e-3, (
        f"SC reveal should change logits by >1e-3; got max|Δ|={delta_reveal}"
    )
    assert delta_sc_vs_off < 1e-5, (
        f"SC-on with no reveal should ≈ SC-off; got max|Δ|={delta_sc_vs_off}"
    )


# ---------------------------------------------------------------------------
# Criterion 4: ligand differential pinned
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ligand_on_vs_off_logits_differ() -> None:
    """Ligand-on vs ligand-off (empty ligand mask) logits differ max|Δ|>1e-3."""
    inp = _synthetic_inputs(seq_len=8, ligand_atoms=4, n_states=2)
    seq_len = inp["seq_len"]
    n_states = inp["n_states"]
    stage_set = make_stage_set()
    key = jax.random.PRNGKey(7)

    model = _small_ligand_model(use_sc=False, key=jax.random.PRNGKey(20))

    def _run(lig_mask: jax.Array) -> jax.Array:
        bundle, config = build_inference_bundle(
            coords=inp["coords"],
            mask=inp["mask"],
            residue_index=inp["residue_index"],
            chain_index=inp["chain_index"],
            ligand_coords=inp["lig_coords"],
            ligand_atom_types=inp["lig_types"],
            ligand_mask=lig_mask,
            mode="score_conditional",
            inference=True,
        )
        return score_unconditional.kernel(model, key, bundle, config, stage_set)

    logits_ligand_on = _run(inp["lig_mask"])
    logits_ligand_off = _run(jnp.zeros_like(inp["lig_mask"]))

    delta = float(np.abs(np.asarray(logits_ligand_on) - np.asarray(logits_ligand_off)).max())
    assert delta > 1e-3, (
        f"Ligand-on vs ligand-off should differ by >1e-3; got max|Δ|={delta}"
    )
