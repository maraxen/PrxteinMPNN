"""Decoding-schedule selection for wave-color (chromatic) scheduling research (W0.3).

Wires a schedule selector across the five arms required by the variance battery
(W1a.2): `random_ar`, `fixed_n_to_c`, `frozen_random_sigma`, `chromatic`,
`improper_coloring`. Coloring adjacency is built from the model's edge set
(per-state CA k-NN — same metric as `aminx.utils.wave_parallel`, extended with
ligand-mediated contacts) so it can be checked against the actual model k-NN via
`assert_model_adj_subset` (confound C_knn_alignment, B3).

See `mpnn_ext/.praxia/docs/preregistration/260630_wave-color-scheduling-fundamental-tests.md`
section 3 (W0.3) for the spec this implements.

Known gap (next step after this module): `AutoregressiveDecode.__call__`
(`aminx/inference/decode/autoregressive.py`, both the `lax.scan` and
`lax.while_loop` branches) only decodes group slot 0 of each wave
(`wave.group_positions[wave_idx, 0, ...]` / `wave.group_valid[wave_idx, 0]`).
The `chromatic` and `improper_coloring` arms built here produce multi-group
waves (G > 1 — the whole point of within-color Jacobi parallelism); until the
decode kernel's `step_fn` is extended to iterate all `G` group slots in a wave
(masked by `group_valid`), positions in slots 1..G-1 are silently never
sampled. `random_ar` / `fixed_n_to_c` / `frozen_random_sigma` are G=1 by
construction (via `WaveScheduleBundle.from_tie_groups`) and are unaffected.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.types.bundles import WaveScheduleBundle

DecodingSchedule = Literal[
  "random_ar",
  "fixed_n_to_c",
  "frozen_random_sigma",
  "chromatic",
  "improper_coloring",
]

_COLORING_SCHEDULES = ("chromatic", "improper_coloring")
_ORDER_SCHEDULES = ("random_ar", "fixed_n_to_c", "frozen_random_sigma")


def build_position_adjacency(
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  *,
  k_neighbors: int,
  structure_mapping: Int[Array, " L"] | None = None,
  ligand_coords: Float[Array, "A 3"] | None = None,
  ligand_mask: Float[Array, " A"] | None = None,
  ligand_contact_radius: float = 6.0,
) -> list[set[int]]:
  """Build the coloring adjacency from the model's edge set (confound B3).

  Protein-protein edges use CA k-NN — the same metric `aminx.utils.wave_parallel`
  uses for its existing wave-coloring utility — masked by `mask` and (for
  multistate union graphs, W3.1) `structure_mapping`. Ligand-mediated edges
  connect any two positions that both have a CA within `ligand_contact_radius`
  of a (masked-valid) ligand atom: a conservative superset of the model's
  learned ligand-conditioning edges. This adjacency is an approximation of the
  model's true edge set — `assert_model_adj_subset` and W1b.2 (edge-set-
  containment deliberate-failure test) calibrate/validate it against the
  model's actual `neighbor_indices`.

  Returns:
    adjacency: length-L list of neighbor-index sets (undirected, symmetric).

  """
  seq_len = coords.shape[0]
  ca = coords[:, 1, :]  # (L, 3) — CA, per wave_parallel.py convention
  distances = jnp.linalg.norm(ca[:, None, :] - ca[None, :], axis=-1)
  distances_masked = jnp.where(
    (mask[:, None] * mask[None, :]).astype(jnp.bool_),
    distances,
    jnp.inf,
  )
  distances_masked = distances_masked.at[jnp.arange(seq_len), jnp.arange(seq_len)].set(jnp.inf)
  if structure_mapping is not None:
    same_structure = structure_mapping[:, None] == structure_mapping[None, :]
    distances_masked = jnp.where(same_structure, distances_masked, jnp.inf)

  k = min(k_neighbors, seq_len - 1) if seq_len > 1 else 0
  adjacency: list[set[int]] = [set() for _ in range(seq_len)]
  if k > 0:
    _, knn_idx = jax.lax.top_k(-distances_masked, k)
    knn_idx_np = np.asarray(knn_idx)
    finite = np.asarray(jnp.take_along_axis(distances_masked, knn_idx, axis=-1) < jnp.inf)
    for i in range(seq_len):
      for slot in range(k):
        if not finite[i, slot]:
          continue
        j = int(knn_idx_np[i, slot])
        adjacency[i].add(j)
        adjacency[j].add(i)

  if ligand_coords is not None and ligand_mask is not None and ligand_coords.shape[0] > 0:
    lig_dists = jnp.linalg.norm(ca[:, None, :] - ligand_coords[None, :, :], axis=-1)
    lig_dists = jnp.where(ligand_mask[None, :].astype(jnp.bool_), lig_dists, jnp.inf)
    in_contact = jnp.any(lig_dists < ligand_contact_radius, axis=-1) & mask.astype(jnp.bool_)
    contact_positions = np.flatnonzero(np.asarray(in_contact))
    for i in contact_positions:
      for j in contact_positions:
        if i != j:
          adjacency[int(i)].add(int(j))

  return adjacency


def assert_model_adj_subset(
  model_neighbor_indices: Int[Array, "L K"],
  coloring_adjacency: list[set[int]],
  mask: Float[Array, " L"] | None = None,
) -> None:
  """Assert `model_adj ⊆ coloring_adj` (confound C_knn_alignment, B3).

  Raises AssertionError naming the first model edge with no corresponding entry
  in `coloring_adjacency`.
  """
  neighbor_np = np.asarray(model_neighbor_indices)
  mask_np = np.asarray(mask) if mask is not None else None
  for i in range(neighbor_np.shape[0]):
    if mask_np is not None and mask_np[i] <= 0.5:
      continue
    for neighbor in neighbor_np[i].reshape(-1):
      j = int(neighbor)
      if j == i:
        continue
      if mask_np is not None and mask_np[j] <= 0.5:
        continue
      if j not in coloring_adjacency[i]:
        msg = (
          f"model_adj is not a subset of coloring_adj: model edge ({i}, {j}) "
          "has no corresponding coloring-adjacency entry (C_knn_alignment, B3)."
        )
        raise AssertionError(msg)


def color_positions(
  adjacency: list[set[int]],
  *,
  improper: bool = False,
  key: PRNGKeyArray | None = None,
) -> np.ndarray:
  """Greedy DSATUR-style graph coloring of `adjacency` (chromatic / improper-coloring arms).

  `improper=True` produces the W1a.1/B1 control arm: edges are relabeled through
  a random permutation of node identities before coloring, so the resulting
  color classes have the same degree-sequence "shape" as a proper coloring but
  are not actually independent sets of the *real* adjacency — deliberately
  violating the conditional-independence property chromatic scheduling relies
  on (information-ablation probe).
  """
  n = len(adjacency)
  coloring_adjacency = adjacency
  if improper:
    if key is None:
      msg = "improper=True requires a PRNG `key` to permute the adjacency."
      raise ValueError(msg)
    perm = np.asarray(jax.random.permutation(key, n))
    coloring_adjacency = [{int(perm[j]) for j in adjacency[int(perm[i])]} for i in range(n)]

  colors = np.full(n, -1, dtype=np.int32)
  order = sorted(range(n), key=lambda x: -len(coloring_adjacency[x]))
  for i in order:
    neighbor_colors = {colors[j] for j in coloring_adjacency[i] if colors[j] >= 0}
    c = 0
    while c in neighbor_colors:
      c += 1
    colors[i] = c
  return colors


def select_decoding_order(
  key: PRNGKeyArray,
  seq_len: int,
  mode: Literal["random_ar", "fixed_n_to_c", "frozen_random_sigma"],
) -> Int[Array, " L"]:
  """Decoding order for the three non-coloring schedule arms.

  `random_ar` and `frozen_random_sigma` both draw a random permutation; the
  distinction is caller discipline (seed protocol M6): `random_ar` must be
  called with a fresh `key` per sample, `frozen_random_sigma` with the *same*
  `key` reused across samples (sigma fixed) so its `schedule_variance`
  converges to the categorical floor rather than the permutation-noise floor.
  """
  if mode == "fixed_n_to_c":
    return jnp.arange(seq_len, dtype=jnp.int32)
  return jax.random.permutation(key, seq_len).astype(jnp.int32)


def build_wave_schedule(
  mode: DecodingSchedule,
  *,
  key: PRNGKeyArray,
  tie_group_map: Int[Array, " L"],
  coords: Float[Array, "L 4 3"] | None = None,
  mask: Float[Array, " L"] | None = None,
  k_neighbors: int = 48,
  structure_mapping: Int[Array, " L"] | None = None,
  ligand_coords: Float[Array, "A 3"] | None = None,
  ligand_mask: Float[Array, " A"] | None = None,
) -> WaveScheduleBundle:
  """Build a `WaveScheduleBundle` for one of the five schedule arms (W0.3).

  `chromatic` and `improper_coloring` require `coords`/`mask` (and optionally
  `structure_mapping`/`ligand_coords`/`ligand_mask`) to build the model-edge-set
  coloring adjacency (B3); the other three arms only need `tie_group_map`. See
  module docstring for the known decode-kernel gap affecting the coloring arms.
  """
  seq_len = tie_group_map.shape[-1]
  if mode in _ORDER_SCHEDULES:
    decoding_order = select_decoding_order(key, seq_len, mode)
    return WaveScheduleBundle.from_tie_groups(tie_group_map, decoding_order)

  if mode not in _COLORING_SCHEDULES:
    msg = f"Unknown schedule mode: {mode!r}"
    raise ValueError(msg)

  if coords is None or mask is None:
    msg = f"schedule mode {mode!r} requires `coords` and `mask` to build the coloring adjacency."
    raise ValueError(msg)

  adjacency = build_position_adjacency(
    coords,
    mask,
    k_neighbors=k_neighbors,
    structure_mapping=structure_mapping,
    ligand_coords=ligand_coords,
    ligand_mask=ligand_mask,
  )
  group_ids_sorted = sorted({int(g) for g in tie_group_map.tolist()})
  representative_position = {
    g: int(jnp.argmax((tie_group_map == g).astype(jnp.int32))) for g in group_ids_sorted
  }
  group_adjacency: list[set[int]] = []
  for g in group_ids_sorted:
    pos = representative_position[g]
    neighbor_positions = adjacency[pos]
    neighbor_groups = {int(tie_group_map[p]) for p in neighbor_positions}
    neighbor_groups.discard(g)
    group_adjacency.append({group_ids_sorted.index(ng) for ng in neighbor_groups})

  group_colors = color_positions(group_adjacency, improper=(mode == "improper_coloring"), key=key)
  return WaveScheduleBundle.from_colors(jnp.asarray(group_colors), tie_group_map)


__all__ = [
  "DecodingSchedule",
  "assert_model_adj_subset",
  "build_position_adjacency",
  "build_wave_schedule",
  "color_positions",
  "select_decoding_order",
]
