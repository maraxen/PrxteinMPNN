"""Phase 3 PR1/PR2: `replace` harness (dummy) + real Equinox payloads (`payloads.py`)."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.payloads import (
  EncodedFeatures,
  GridLineage,
  LigandStack,
  MultistateStackPayload,
  SampleResult,
  SamplingControls,
)


class DummyPayload(eqx.Module):
    """Minimal stand-in for roadmap §3.2 payloads — dynamic array + `static=True` int.

    `eqx.tree_at` only addresses **pytree leaves**; static fields are not leaves, so
    `replace` uses `tree_at` for `data` and reconstructs the module when `tag`
    changes (same split we will document on real payloads in PR2).
    """

    data: jax.Array
    tag: int = eqx.field(static=True)

    def replace(self, **kw: Any) -> DummyPayload:
        """Keyword-only updates."""
        out: DummyPayload = self
        for key, value in kw.items():
            if key not in ("data", "tag"):
                msg = f"DummyPayload.replace: unknown field {key!r}"
                raise TypeError(msg)
            if key == "data":
                out = eqx.tree_at(lambda s: s.data, out, value)
            else:
                out = DummyPayload(data=out.data, tag=int(value))
        return out


def test_replace_data_roundtrip() -> None:
    p = DummyPayload(data=jnp.zeros((2,)), tag=0)
    q = p.replace(data=jnp.array([1.0, 2.0]))
    assert p.tag == q.tag == 0
    np.testing.assert_array_equal(np.asarray(q.data), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(p.data), np.zeros((2,)))


def test_replace_static_tag() -> None:
    p = DummyPayload(data=jnp.ones((1,)), tag=1)
    q = p.replace(tag=42)
    np.testing.assert_array_equal(np.asarray(q.data), np.asarray(p.data))
    assert q.tag == 42


def test_replace_multiple_fields() -> None:
    p = DummyPayload(data=jnp.arange(3.0), tag=0)
    q = p.replace(data=jnp.array([7.0, 8.0, 9.0]), tag=3)
    assert q.tag == 3
    np.testing.assert_array_equal(np.asarray(q.data), np.array([7.0, 8.0, 9.0]))


def test_replace_unknown_field_raises() -> None:
    p = DummyPayload(data=jnp.zeros((1,)), tag=0)
    with pytest.raises(TypeError, match="unknown field 'nope'"):
        p.replace(nope=jnp.ones((1,)))


def test_replace_under_jit() -> None:
    p = DummyPayload(data=jnp.array([0.0, 1.0]), tag=7)

    @jax.jit
    def bump(x: DummyPayload) -> DummyPayload:
        return x.replace(data=x.data + 1.0)

    q = bump(p)
    np.testing.assert_array_equal(np.asarray(q.data), np.array([1.0, 2.0]))
    assert q.tag == 7


def test_tree_structure_preserved_after_data_replace() -> None:
    """Dynamic-only `replace` keeps the same PyTreeDef; static metadata lives in treedef."""
    p = DummyPayload(data=jnp.array([3.0]), tag=9)
    q = p.replace(data=jnp.array([4.0]))
    leaves_p, treedef_p = jax.tree_util.tree_flatten(p)
    leaves_q, treedef_q = jax.tree_util.tree_flatten(q)
    assert treedef_p == treedef_q
    assert len(leaves_p) == len(leaves_q) == 1
    np.testing.assert_array_equal(np.asarray(leaves_p[0]), np.array([3.0]))
    np.testing.assert_array_equal(np.asarray(leaves_q[0]), np.array([4.0]))
    np.testing.assert_array_equal(np.asarray(q.data), np.array([4.0]))


def test_static_change_updates_treedef_metadata() -> None:
    """Replacing `static=True` fields rebuilds the module; treedef may differ (expected)."""
    p = DummyPayload(data=jnp.array([3.0]), tag=9)
    q = p.replace(tag=1)
    _, treedef_p = jax.tree_util.tree_flatten(p)
    _, treedef_q = jax.tree_util.tree_flatten(q)
    assert treedef_p != treedef_q
    assert q.tag == 1


def _tiny_multistate_stack() -> MultistateStackPayload:
    s, l_, n_flat = 2, 8, 32
    coords = jnp.zeros((s, l_, 4, 3), dtype=jnp.float32)
    mask = jnp.ones((s, l_), dtype=jnp.float32)
    res_i = jnp.arange(l_, dtype=jnp.int32)[None, :].repeat(s, axis=0)
    chain = jnp.zeros((s, l_), dtype=jnp.int32)
    tie = jnp.arange(l_, dtype=jnp.int32)[None, :].repeat(s, axis=0)
    fixed_m = jnp.ones((s, l_), dtype=jnp.float32)
    fixed_t = jnp.zeros((s, l_), dtype=jnp.int32)
    rows = jnp.full((s, l_), -1, dtype=jnp.int32)
    offs = jnp.arange(s + 1, dtype=jnp.int32)
    return MultistateStackPayload(
        coords_stack=coords,
        mask_stack=mask,
        residue_index_stack=res_i,
        chain_index_stack=chain,
        tie_group_map_stack=tie,
        fixed_mask_stack=fixed_m,
        fixed_tokens_stack=fixed_t,
        state_flat_rows=rows,
        flat_row_offsets=offs,
        n_states=s,
        n_canonical=l_,
        n_flat=n_flat,
    )


def test_multistate_stack_dynamic_replace() -> None:
    p = _tiny_multistate_stack()
    new_mask = jnp.zeros_like(p.mask_stack)
    q = p.replace(mask_stack=new_mask)
    assert q.n_states == p.n_states
    np.testing.assert_array_equal(np.asarray(q.mask_stack), np.asarray(new_mask))
    np.testing.assert_array_equal(np.asarray(q.coords_stack), np.asarray(p.coords_stack))


def test_multistate_stack_static_replace_rebuilds() -> None:
    p = _tiny_multistate_stack()
    q = p.replace(n_flat=64)
    assert q.n_flat == 64
    assert q.n_states == p.n_states
    np.testing.assert_array_equal(np.asarray(q.coords_stack), np.asarray(p.coords_stack))


def test_multistate_stack_unknown_field() -> None:
    p = _tiny_multistate_stack()
    with pytest.raises(TypeError, match="unknown field"):
        p.replace(not_a_field=jnp.array(1))


def test_ligand_stack_replace() -> None:
    s, a, d = 2, 5, 3
    y = jnp.zeros((s, a, d))
    p = LigandStack(y_stack=y, y_t_stack=y + 1.0, y_m_stack=y + 2.0)
    y2 = jnp.ones((s, a, d))
    q = p.replace(y_stack=y2)
    np.testing.assert_array_equal(np.asarray(q.y_stack), np.ones((s, a, d)))
    np.testing.assert_array_equal(np.asarray(q.y_t_stack), np.asarray(p.y_t_stack))


def test_grid_lineage_replace_static() -> None:
    g = GridLineage(job_id="j0", chunk_id=0, sample_start=0, sample_count=4)
    h = g.replace(chunk_id=3, sample_count=8)
    assert h.job_id == "j0"
    assert h.chunk_id == 3
    assert h.sample_count == 8


def test_sampling_controls_under_jit() -> None:
    ctrl = SamplingControls(
        temperature=jnp.array(0.5, dtype=jnp.float32),
        multi_state_strategy_idx=jnp.array(1, dtype=jnp.int32),
        multi_state_temperature=jnp.array(1.0, dtype=jnp.float32),
        state_weights=jnp.ones((2,), dtype=jnp.float32) / 2.0,
        bias_stack=jnp.zeros((2, 4, 21), dtype=jnp.float32),
    )

    @jax.jit
    def scale_temperature(c: SamplingControls) -> SamplingControls:
        return c.replace(temperature=c.temperature * 2.0)

    out = scale_temperature(ctrl)
    assert float(np.asarray(out.temperature)) == pytest.approx(1.0)


def test_encoded_features_and_sample_result_replace() -> None:
    nf = jnp.zeros((3, 5))
    ef = jnp.zeros((3, 4, 2))
    nei = jnp.zeros((3, 4), dtype=jnp.int32)
    enc = EncodedFeatures(node_features=nf, edge_features=ef, neighbor_indices=nei)
    enc2 = enc.replace(node_features=nf + 1.0)
    np.testing.assert_array_equal(np.asarray(enc2.node_features), np.ones((3, 5)))
    seq = jnp.zeros((5,), dtype=jnp.int32)
    logits = jnp.zeros((5, 21), dtype=jnp.float32)
    sr = SampleResult(sequence=seq, logits=logits)
    sr2 = sr.replace(logits=logits + 2.0)
    np.testing.assert_array_equal(np.asarray(sr2.logits), np.full((5, 21), 2.0, dtype=np.float32))
