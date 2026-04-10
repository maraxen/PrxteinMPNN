"""Ligand-aware feature extraction module for PrxteinMPNN.

This module contains the ProteinFeaturesLigand class that extracts features
from both protein coordinates and ligand/atomic context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from prxteinmpnn.utils.types import (
        AlphaCarbonMask,
        ChainIndex,
        ResidueIndex,
        StructureAtomicCoordinates,
    )

PRNGKeyArray = jax.Array
LayerNorm = eqx.nn.LayerNorm


class PositionalEncodings(eqx.Module):
    """Positional encodings for residues and chains."""

    w_pos: eqx.nn.Linear
    num_embeddings: int = eqx.field(static=True)

    def __init__(self, num_embeddings: int, *, key: PRNGKeyArray) -> None:
        # num_embeddings here refers to the number of relative features (e.g. 16 or 32)
        # The output dimension is ALWAYS 16 in the reference models
        self.num_embeddings = num_embeddings
        # Input to linear is [offset_one_hot(2*num_pos + 1), chain_one_hot(1)]
        self.w_pos = eqx.nn.Linear(2 * num_embeddings + 2, 16, use_bias=False, key=key)

    def __call__(self, offset: jax.Array, same_chain: jax.Array) -> jax.Array:
        # offset: (N, K) relative residue indices
        # same_chain: (N, K) boolean for same-chain status

        # Clamp offset to [-num_embeddings, num_embeddings]
        d_offset = jnp.clip(offset + self.num_embeddings, 0, 2 * self.num_embeddings).astype(jnp.int32)
        offset_one_hot = jax.nn.one_hot(d_offset, 2 * self.num_embeddings + 1)

        # Combine with chain info
        # same_chain=1 for same chain

        chain_info = (1.0 - same_chain.astype(jnp.float32))[..., None]

        # Combined input: (N, K, 2*num_embeddings + 1 + 1)
        d_combined = jnp.concatenate([offset_one_hot, chain_info], axis=-1)

        return jax.vmap(jax.vmap(self.w_pos))(d_combined)


class ProteinFeaturesLigand(eqx.Module):
    """Extracts features from protein coordinates and ligand context."""

    embeddings: PositionalEncodings
    edge_embedding: eqx.nn.Linear
    norm_edges: LayerNorm

    node_project_down: eqx.nn.Linear
    norm_nodes: LayerNorm

    type_linear: eqx.nn.Linear

    y_nodes: eqx.nn.Linear
    y_edges: eqx.nn.Linear

    norm_y_edges: LayerNorm
    norm_y_nodes: LayerNorm

    w_e_proj: eqx.nn.Linear  # Added to match W_e in ProteinMPNN

    periodic_table_features: jnp.ndarray = eqx.field(static=True)
    side_chain_atom_types: jnp.ndarray = eqx.field(static=True)

    k_neighbors: int = eqx.field(static=True)
    atom_context_num: int = eqx.field(static=True)
    use_side_chains: bool = eqx.field(static=True)

    def _make_angle_features(self, A: jax.Array, B: jax.Array, C: jax.Array, Y: jax.Array) -> jax.Array:
        """Port of _make_angle_features from PyTorch."""
        v1 = A - B
        v2 = C - B
        e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
        e1_v2_dot = jnp.sum(e1 * v2, axis=-1, keepdims=True)
        u2 = v2 - e1 * e1_v2_dot
        e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
        e3 = jnp.cross(e1, e2, axis=-1)

        # R_residue: (L, 3, 3) relative rotation matrix
        R_residue = jnp.stack([e1, e2, e3], axis=-1)

        # local_vectors: relative position of Y in residue frame
        # (L, M, 3)
        diff = Y - B[:, None, :]
        local_vectors = jnp.einsum("lqp, lym -> lyp", R_residue, diff)

        rxy = jnp.sqrt(local_vectors[..., 0]**2 + local_vectors[..., 1]**2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        return jnp.stack([f1, f2, f3, f4], axis=-1)

    def _rbf(self, D: jax.Array) -> jax.Array:
        """Standard RBF bins [2.0, 22.0] for LigandMPNN."""
        D_min, D_max, D_count = 2.0, 22.0, 16
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        return jnp.exp(-(((D[..., None] - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A: jax.Array, B: jax.Array, E_idx: jax.Array) -> jax.Array:
        D_A_B = jnp.sqrt(jnp.sum((A[:, None, :] - B[None, :, :])**2, axis=-1) + 1e-6)
        D_neighbors = jnp.take_along_axis(D_A_B, E_idx, axis=1)
        return self._rbf(D_neighbors)

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        k_neighbors: int,
        atom_context_num: int = 16,
        num_positional_embeddings: int = 16,
        use_side_chains: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jax.random.split(key, 8)

        self.k_neighbors = k_neighbors
        self.atom_context_num = atom_context_num
        self.use_side_chains = use_side_chains

        self.embeddings = PositionalEncodings(num_positional_embeddings, key=keys[0])
        # edge_in = 16 + 16 * 25 = 416
        self.edge_embedding = eqx.nn.Linear(416, edge_features, use_bias=False, key=keys[1])
        self.norm_edges = LayerNorm(edge_features)

        # node input: 5 * 16 (rbf) + 64 (type) + 4 (angle) = 148
        self.node_project_down = eqx.nn.Linear(148, node_features, key=keys[2])
        self.norm_nodes = LayerNorm(node_features)

        self.type_linear = eqx.nn.Linear(147, 64, key=keys[3])
        self.y_nodes = eqx.nn.Linear(147, node_features, use_bias=False, key=keys[4])
        self.y_edges = eqx.nn.Linear(16, node_features, use_bias=False, key=keys[5])

        self.norm_y_edges = LayerNorm(node_features)
        self.norm_y_nodes = LayerNorm(node_features)

        self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[7])

        # Static periodic table features: (3, 120)
        # 0: Atomic Number (0-119)
        # 1: Group (19 categories)
        # 2: Period (8 categories)
        import numpy as np
        self.periodic_table_features = np.array([
            np.arange(119),
            np.array([0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
        ])
        self.side_chain_atom_types = np.array([
            6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6,
            6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8,
        ])

    def __call__(
        self,
        key: PRNGKeyArray,
        structure_coordinates: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_index: ResidueIndex,
        chain_index: ChainIndex,
        Y: jnp.ndarray,    # Ligand coords [L, M, 3]
        Y_t: jnp.ndarray,  # Ligand types [L, M]
        Y_m: jnp.ndarray,  # Ligand mask [L, M]
        backbone_noise: float = 0.0,
        structure_mapping: jnp.ndarray | None = None,
        *,
        xyz_37: jnp.ndarray | None = None,
        xyz_37_m: jnp.ndarray | None = None,
        chain_mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Implement full forward pass
        # N, CA, C, O
        N = structure_coordinates[:, 0, :]
        Ca = structure_coordinates[:, 1, :]
        C = structure_coordinates[:, 2, :]
        O = structure_coordinates[:, 3, :]

        # Virtual Cb
        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # 1. Edge Features
        dist_ca = jnp.sqrt(jnp.sum((Ca[:, None, :] - Ca[None, :, :])**2, axis=-1) + 1e-6)
        mask_2d = mask[:, None] * mask[None, :]
        dist_ca = dist_ca * mask_2d + (1.0 - mask_2d) * 1e4
        if structure_mapping is not None:
            same_structure = structure_mapping[:, None] == structure_mapping[None, :]
            dist_ca = jnp.where(same_structure, dist_ca, 1e4)

        k = jnp.minimum(self.k_neighbors, Ca.shape[0])
        _, E_idx = jax.lax.top_k(-dist_ca, k)

        RBF_all = []
        # CA-CA RBF
        r_ca_ca = jnp.take_along_axis(dist_ca, E_idx, axis=1)
        RBF_all.append(self._rbf(r_ca_ca))

        # All other 24 pairs
        pairs = [
            (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca),
            (C, N), (O, N), (Cb, N), (C, Cb), (O, Cb), (C, O),
        ]
        for p_a, p_b in pairs:
            RBF_all.append(self._get_rbf(p_a, p_b, E_idx))

        RBF_all = jnp.concatenate(RBF_all, axis=-1)

        offset = residue_index[:, None] - residue_index[None, :]
        E_offset = jnp.take_along_axis(offset, E_idx, axis=1)
        E_chains = jnp.take_along_axis((chain_index[:, None] == chain_index[None, :]), E_idx, axis=1)

        E_pos = self.embeddings(E_offset, E_chains)
        E = jnp.concatenate([E_pos, RBF_all], axis=-1)
        E = jax.vmap(jax.vmap(self.edge_embedding))(E)
        E = jax.vmap(jax.vmap(self.norm_edges))(E)
        E = jax.vmap(jax.vmap(self.w_e_proj))(E)

        if self.use_side_chains:
            if xyz_37 is None or xyz_37_m is None:
                msg = "xyz_37 and xyz_37_m must be provided when use_side_chains=True"
                raise ValueError(msg)

            chain_mask_in = (
                jnp.zeros_like(mask, dtype=jnp.float32)
                if chain_mask is None
                else chain_mask.astype(jnp.float32)
            )

            e_idx_sub = E_idx[:, :16]
            xyz_37_m = xyz_37_m * (1.0 - chain_mask_in[:, None])
            r_m = xyz_37_m[:, 5:][e_idx_sub]
            r = xyz_37[:, 5:, :][e_idx_sub]
            r_t = jnp.broadcast_to(
                self.side_chain_atom_types[None, None, :],
                (mask.shape[0], e_idx_sub.shape[1], self.side_chain_atom_types.shape[0]),
            )

            r = r.reshape(mask.shape[0], -1, 3)
            r_m = r_m.reshape(mask.shape[0], -1)
            r_t = r_t.reshape(mask.shape[0], -1)

            Y = jnp.concatenate([r, Y], axis=1)
            Y_m = jnp.concatenate([r_m.astype(Y_m.dtype), Y_m], axis=1)
            Y_t = jnp.concatenate([r_t.astype(Y_t.dtype), Y_t], axis=1)

            cb_y_distances = jnp.sum((Cb[:, None, :] - Y) ** 2, axis=-1)
            mask_y = mask[:, None] * Y_m
            cb_y_distances_adjusted = cb_y_distances * mask_y + (1.0 - mask_y) * 10000.0
            _, e_idx_y = jax.lax.top_k(-cb_y_distances_adjusted, self.atom_context_num)

            Y = jnp.take_along_axis(Y, e_idx_y[:, :, None], axis=1)
            Y_t = jnp.take_along_axis(Y_t, e_idx_y, axis=1)
            Y_m = jnp.take_along_axis(Y_m, e_idx_y, axis=1)

        # 2. Node/Ligand Features
        # type_1hot: (L, M, 147)
        Y_t_cast = Y_t.astype(jnp.int32)
        Y_t_g = self.periodic_table_features[1, Y_t_cast]
        Y_t_p = self.periodic_table_features[2, Y_t_cast]

        Y_t_1hot_ = jnp.concatenate([
            jax.nn.one_hot(Y_t_cast, 120),
            jax.nn.one_hot(Y_t_g.astype(jnp.int32), 19),
            jax.nn.one_hot(Y_t_p.astype(jnp.int32), 8),
        ], axis=-1)

        Y_t_1hot = jax.vmap(jax.vmap(self.type_linear))(Y_t_1hot_)

        r_n_y = self._rbf(jnp.sqrt(jnp.sum((N[:, None, :] - Y)**2, axis=-1) + 1e-6))
        r_ca_y = self._rbf(jnp.sqrt(jnp.sum((Ca[:, None, :] - Y)**2, axis=-1) + 1e-6))
        r_c_y = self._rbf(jnp.sqrt(jnp.sum((C[:, None, :] - Y)**2, axis=-1) + 1e-6))
        r_o_y = self._rbf(jnp.sqrt(jnp.sum((O[:, None, :] - Y)**2, axis=-1) + 1e-6))
        r_cb_y = self._rbf(jnp.sqrt(jnp.sum((Cb[:, None, :] - Y)**2, axis=-1) + 1e-6))

        f_angles = self._make_angle_features(N, Ca, C, Y)

        D_all = jnp.concatenate([r_n_y, r_ca_y, r_c_y, r_o_y, r_cb_y, Y_t_1hot, f_angles], axis=-1)
        V = jax.vmap(jax.vmap(self.node_project_down))(D_all)
        V = jax.vmap(jax.vmap(self.norm_nodes))(V)

        # ligand-ligand edges
        Y_edges = self._rbf(jnp.sqrt(jnp.sum((Y[:, :, None, :] - Y[:, None, :, :])**2, axis=-1) + 1e-6))
        Y_edges = jax.vmap(jax.vmap(jax.vmap(self.y_edges)))(Y_edges)
        Y_nodes = jax.vmap(jax.vmap(self.y_nodes))(Y_t_1hot_)

        Y_edges = jax.vmap(jax.vmap(jax.vmap(self.norm_y_edges)))(Y_edges)
        Y_nodes = jax.vmap(jax.vmap(self.norm_y_nodes))(Y_nodes)

        return V, E, E_idx, Y_nodes, Y_edges, Y_m
