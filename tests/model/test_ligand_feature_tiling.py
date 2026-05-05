"""Parity checks for chunked ligand pairwise projections (jaxbeans-style inlined tiling)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
from prxteinmpnn.model.ligand_tiling import map_chunks_axis0


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