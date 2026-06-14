"""Tests: atom_37 / atom_37_mask forwarding from ligand_context to build_inference_bundle.

The kernel_dispatch call sites were previously dropping atom_37 and atom_37_mask
from ligand_context silently. This test suite verifies the wiring is present at
both the helper and bundle-builder layers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.host._sampling_helper import _prepare_ligand_context
from aminx.run.specs import SamplingSpecification


L = 8  # sequence length
BATCH = 1


class _FakeProtein:
    """Minimal Protein stand-in for _prepare_ligand_context."""

    def __init__(self, *, seq_len: int, batch_size: int) -> None:
        rng = np.random.default_rng(0)
        self.coordinates = rng.normal(size=(batch_size, seq_len, 37, 3)).astype(np.float32)
        self.Y = None
        self.Y_t = None
        self.Y_m = None
        self.atom_mask = np.ones((batch_size, seq_len, 37), dtype=np.float32)
        self.full_atom_mask = None


def _base_spec(**kwargs):
    return SamplingSpecification(inputs="test.pdb", model_family="ligandmpnn", **kwargs)


# ── _prepare_ligand_context: sidechain_conditioning populates atom_37 ──────────

def test_ligand_context_populates_atom37_when_sidechain_conditioning():
    """_prepare_ligand_context must set atom_37/atom_37_mask when sidechain_conditioning=True."""
    protein = _FakeProtein(seq_len=L, batch_size=BATCH)
    spec = _base_spec(sidechain_conditioning=True)
    ctx = _prepare_ligand_context(spec, batched_ensemble=protein, batch_size=BATCH, seq_len=L)

    assert ctx["atom_37"] is not None, "atom_37 must be set when sidechain_conditioning=True"
    assert ctx["atom_37_mask"] is not None, "atom_37_mask must be set when sidechain_conditioning=True"
    assert ctx["atom_37"].shape == (BATCH, L, 37, 3), f"Unexpected shape: {ctx['atom_37'].shape}"
    assert ctx["atom_37_mask"].shape == (BATCH, L, 37), f"Unexpected shape: {ctx['atom_37_mask'].shape}"


def test_ligand_context_atom37_none_without_sidechain_conditioning():
    """_prepare_ligand_context must leave atom_37 as None when sidechain_conditioning=False."""
    protein = _FakeProtein(seq_len=L, batch_size=BATCH)
    spec = _base_spec(sidechain_conditioning=False)
    ctx = _prepare_ligand_context(spec, batched_ensemble=protein, batch_size=BATCH, seq_len=L)

    assert ctx["atom_37"] is None
    assert ctx["atom_37_mask"] is None


# ── build_inference_bundle: atom_37 forwarding persists in GeometryBundle ──────

def _make_inputs(seq_len: int):
    rng = np.random.default_rng(1)
    coords = jnp.asarray(rng.normal(size=(seq_len, 4, 3)).astype(np.float32))
    mask = jnp.ones(seq_len, dtype=jnp.float32)
    res_idx = jnp.arange(seq_len, dtype=jnp.int32)
    chain_idx = jnp.zeros(seq_len, dtype=jnp.int32)
    return coords, mask, res_idx, chain_idx


def test_build_inference_bundle_stores_atom37():
    """atom_37 passed to build_inference_bundle must appear in bundle.geometry."""
    coords, mask, res_idx, chain_idx = _make_inputs(L)
    rng = np.random.default_rng(2)
    atom_37 = jnp.asarray(rng.normal(size=(L, 37, 3)).astype(np.float32))
    atom_37_mask = jnp.ones((L, 37), dtype=jnp.float32)

    bundle, _ = build_inference_bundle(
        coords, mask, res_idx, chain_idx,
        atom_37=atom_37,
        atom_37_mask=atom_37_mask,
    )

    assert bundle.geometry.atom_37 is not None, "atom_37 missing from geometry bundle"
    assert bundle.geometry.atom_37_mask is not None, "atom_37_mask missing from geometry bundle"
    # Shape is (S, L, 37, 3) after the rank-3→4 expansion in build_inference_bundle
    assert bundle.geometry.atom_37.shape == (1, L, 37, 3)
    assert bundle.geometry.atom_37_mask.shape == (1, L, 37)


def test_build_inference_bundle_atom37_none_when_not_passed():
    """Without atom_37, geometry.atom_37 must be None."""
    coords, mask, res_idx, chain_idx = _make_inputs(L)
    bundle, _ = build_inference_bundle(coords, mask, res_idx, chain_idx)
    assert bundle.geometry.atom_37 is None
    assert bundle.geometry.atom_37_mask is None


# ── chain_mask: not hardcoded to all-zeros when sidechain_conditioning ──────────

def test_ligand_context_chain_mask_not_all_zeros_with_sidechain_conditioning():
    """chain_mask must NOT be all-zeros when sidechain_conditioning=True.

    chain_mask semantics: 1=designable, 0=fixed.
    An all-zeros chain_mask disables all sidechain design (semantically meaningless).
    This test verifies that the computed chain_mask is semantically correct.
    """
    protein = _FakeProtein(seq_len=L, batch_size=BATCH)
    spec = _base_spec(sidechain_conditioning=True)
    ctx = _prepare_ligand_context(spec, batched_ensemble=protein, batch_size=BATCH, seq_len=L)

    assert ctx["chain_mask"] is not None, "chain_mask must be set when sidechain_conditioning=True"
    assert ctx["chain_mask"].shape == (BATCH, L), f"Unexpected shape: {ctx['chain_mask'].shape}"

    # Check that chain_mask is NOT all-zeros (the hardcoded bug)
    assert not jnp.allclose(ctx["chain_mask"], 0.0), (
        "chain_mask must not be all-zeros; this is semantically meaningless "
        "(all-zeros = all residues fixed, disabling sidechain design)"
    )


def test_ligand_context_chain_mask_none_without_sidechain_conditioning():
    """chain_mask must remain None when sidechain_conditioning=False."""
    protein = _FakeProtein(seq_len=L, batch_size=BATCH)
    spec = _base_spec(sidechain_conditioning=False)
    ctx = _prepare_ligand_context(spec, batched_ensemble=protein, batch_size=BATCH, seq_len=L)

    assert ctx["chain_mask"] is None


# ── build_inference_bundle: chain_mask parameter and storage ──────────

def test_build_inference_bundle_stores_chain_mask():
    """chain_mask passed to build_inference_bundle must appear in geometry bundle."""
    coords, mask, res_idx, chain_idx = _make_inputs(L)
    rng = np.random.default_rng(3)
    atom_37 = jnp.asarray(rng.normal(size=(L, 37, 3)).astype(np.float32))
    atom_37_mask = jnp.ones((L, 37), dtype=jnp.float32)
    chain_mask = jnp.ones((L,), dtype=jnp.float32)  # All designable

    bundle, _ = build_inference_bundle(
        coords, mask, res_idx, chain_idx,
        atom_37=atom_37,
        atom_37_mask=atom_37_mask,
        chain_mask=chain_mask,
    )

    assert bundle.geometry.chain_mask is not None, "chain_mask missing from geometry bundle"
    assert bundle.geometry.chain_mask.shape == (1, L), f"Unexpected shape: {bundle.geometry.chain_mask.shape}"
    assert jnp.allclose(bundle.geometry.chain_mask[0], chain_mask)


def test_build_inference_bundle_chain_mask_none_when_not_passed():
    """Without chain_mask, geometry.chain_mask must be None."""
    coords, mask, res_idx, chain_idx = _make_inputs(L)
    bundle, _ = build_inference_bundle(coords, mask, res_idx, chain_idx)
    assert bundle.geometry.chain_mask is None
