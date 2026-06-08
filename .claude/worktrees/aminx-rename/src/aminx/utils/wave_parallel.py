"""Wave-parallel decode tables (host-side graph coloring).

These utilities build the same wave metadata used by the design grid's flattened
multistate path: canonical k-NN graph coloring on CA coordinates, then packing
group members into per-wave slots.

Keeping this in aminx avoids importing design scripts from tests/samplers.
"""

from __future__ import annotations

import numpy as np


def compute_wave_assignments(
  ca_coords: np.ndarray,  # (n_can, 4, 3) — backbone coords, use CA (index 1)
  tie_group_flat: np.ndarray,  # (N_flat,) — group IDs for flat residue array
  group_indices_table: np.ndarray,  # (n_groups, max_group_size) — positions per group
  group_valid_table: np.ndarray,  # (n_groups, max_group_size) — valid mask per group
  k_neighbors: int = 48,
  n_canonical: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Graph-color the k-NN graph of canonical positions to find parallel decode waves.

  Returns:
    wave_group_ids:       (n_waves, max_wave_size) int32 — canonical group IDs per wave
    wave_group_positions: (n_waves, max_wave_size, max_group_size) int32 — flat indices
    wave_group_valid:     (n_waves, max_wave_size) bool — which group slots are active
    wave_position_valid:  (n_waves, max_wave_size, max_group_size) bool — valid members

  """
  from scipy.spatial.distance import cdist  # noqa: PLC0415

  if n_canonical is None:
    n_canonical = int(ca_coords.shape[0])

  # Extract CA coordinates (index 1 in backbone)
  ca = ca_coords[:, 1, :]  # (n_canonical, 3)

  # Build k-NN adjacency for canonical positions
  dists = cdist(ca, ca)
  np.fill_diagonal(dists, np.inf)
  nn_indices = np.argsort(dists, axis=1)[:, :k_neighbors]  # (n_canonical, K)

  # Build undirected adjacency list for canonical group IDs 0..n_canonical-1
  adj = [set() for _ in range(n_canonical)]
  for i in range(n_canonical):
    for j in nn_indices[i]:
      if j < n_canonical:
        adj[i].add(int(j))
        adj[j].add(int(i))

  # Greedy graph coloring (DSATUR-style: color highest-degree first)
  colors = np.full(n_canonical, -1, dtype=np.int32)
  order = sorted(range(n_canonical), key=lambda x: -len(adj[x]))
  for i in order:
    neighbor_colors = {colors[j] for j in adj[i] if colors[j] >= 0}
    c = 0
    while c in neighbor_colors:
      c += 1
    colors[i] = c

  n_waves = int(colors.max()) + 1
  wave_sizes = [int(np.sum(colors == w)) for w in range(n_waves)]
  max_wave_size = int(max(wave_sizes))
  max_group_size = int(group_indices_table.shape[1])

  wave_group_ids = np.full((n_waves, max_wave_size), -1, dtype=np.int32)
  wave_group_positions = np.zeros((n_waves, max_wave_size, max_group_size), dtype=np.int32)
  wave_group_valid = np.zeros((n_waves, max_wave_size), dtype=bool)
  wave_position_valid = np.zeros((n_waves, max_wave_size, max_group_size), dtype=bool)

  wave_slot = np.zeros(n_waves, dtype=np.int32)
  for canonical_id in range(n_canonical):
    w = int(colors[canonical_id])
    slot = int(wave_slot[w])
    wave_group_ids[w, slot] = canonical_id
    wave_group_valid[w, slot] = True
    wave_group_positions[w, slot] = group_indices_table[canonical_id]
    wave_position_valid[w, slot] = group_valid_table[canonical_id]
    wave_slot[w] += 1

  return wave_group_ids, wave_group_positions, wave_group_valid, wave_position_valid
