"""Parity checks for chunked ligand pairwise projections (jaxbeans-style inlined tiling)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Internal imports: not public API
from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi


def test_map_chunks_matches_dense_y_edges() -> None:
  key = jax.random.PRNGKey(42)
  feat = ProteinFeaturesLigand(
    node_features=128,
    edge_features=128,
    k_neighbors=8,
    ligand_l_chunk=-1,
    key=key,
  )
  rng = np.random.default_rng(0)
  L, M = 47, 12
  y = jnp.asarray(rng.standard_normal(size=(L, M, 3)), dtype=jnp.float32)

  dense = feat._y_edges_coords_to_embed(y)
  tiled = map_chunks_axis0(
    y,
    chunk_size=5,
    fn=feat._y_edges_coords_to_embed,
  )

  np.testing.assert_allclose(dense, tiled, rtol=1e-5, atol=1e-5)


def test_map_chunks_axis0_multi_matches_reference() -> None:
  rng = np.random.default_rng(2)
  L, cs = 11, 4
  A = jnp.asarray(rng.standard_normal((L, 2)), dtype=jnp.float32)
  B = jnp.asarray(rng.standard_normal((L, 5)), dtype=jnp.float32)

  def fn(a_slab: jax.Array, b_slab: jax.Array) -> tuple[jax.Array, jax.Array]:
    return a_slab + jnp.float32(0.25), jnp.sin(b_slab)

  out_a, out_b = map_chunks_axis0_multi(fn, cs, (A, B))
  np.testing.assert_allclose(out_a, A + jnp.float32(0.25), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(out_b, jnp.sin(B), rtol=1e-5, atol=1e-5)

  jit_multi = jax.jit(map_chunks_axis0_multi, static_argnums=(0, 1))
  out_j = jit_multi(fn, cs, (A, B))
  np.testing.assert_allclose(out_j[0], out_a, rtol=1e-5, atol=1e-5)


def test_map_chunks_matches_dense_y_nodes() -> None:
  key = jax.random.PRNGKey(7)
  feat = ProteinFeaturesLigand(
    node_features=128,
    edge_features=128,
    k_neighbors=8,
    ligand_l_chunk=-1,
    key=key,
  )
  rng = np.random.default_rng(1)
  L, M = 31, 9
  h = jnp.asarray(rng.standard_normal(size=(L, M, 147)), dtype=jnp.float32)

  dense = feat._y_nodes_proj(h)
  tiled = map_chunks_axis0(
    h,
    chunk_size=4,
    fn=feat._y_nodes_proj,
  )

  np.testing.assert_allclose(dense, tiled, rtol=1e-5, atol=1e-5)