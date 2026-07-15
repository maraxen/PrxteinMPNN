"""Parity checks for chunked ligand pairwise projections (jaxbeans-style inlined tiling)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Internal imports: not public API
from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi

# --- Analytic Cb offset for the canonical local backbone frame used below ---------------
# ProteinFeaturesLigand computes a virtual Cb as:
#   b = Ca - N; c = C - Ca; a = cross(b, c)
#   Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
# For the fixed local frame N=(0,0,0), Ca=(1,0,0), C=(1,1,0) (translated per-residue by T),
# b=(1,0,0), c=(0,1,0), a=(0,0,1), so Cb = Ca + (0.56802827, -0.54067466, -0.58273431)
# regardless of T (translation cancels out of b, c, a). This lets a test choose an exact
# target Cb position per residue by solving T = target_cb - (1,0,0) - _CB_OFFSET.
_CB_OFFSET = np.array([0.56802827, -0.54067466, -0.58273431])


def _make_backbone_for_cb_targets(cb_targets: np.ndarray) -> jnp.ndarray:
  """Build (L, 4, 3) N/Ca/C/O coordinates whose virtual Cb lands exactly on cb_targets."""
  translation = cb_targets - np.array([1.0, 0.0, 0.0]) - _CB_OFFSET
  n = translation
  ca = translation + np.array([1.0, 0.0, 0.0])
  c = translation + np.array([1.0, 1.0, 0.0])
  o = translation + np.array([0.0, 1.0, 0.0])
  return jnp.asarray(np.stack([n, ca, c, o], axis=1).astype(np.float32))


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


# ---------------------------------------------------------------------------
# M-scale regression: per-residue top-k ligand-atom selection (ligand_features.py:383-396).
#
# Existing fixtures above (and tests/model/test_ligandmpnn_equivalence.py) only exercise
# M == atom_context_num, a no-op case for the top-k reduction (jax.lax.top_k(-dist, k) with
# k == M just returns everything, sorted). Necklace's real production ligand context is
# M=57-82 heavy atoms per state, well above atom_context_num (16 or 25) -- the exact regime
# an earlier hardening cycle assumed was handled without verifying the mechanism at this
# scale. These tests construct M=72 ligand atoms placed at known, controlled distances from
# 3 residues' (analytically exact, see `_make_backbone_for_cb_targets`) Cb positions, and
# check the real forward pass (`ProteinFeaturesLigand.__call__`) selects the correct subset.
#
# `e_idx_y` itself is not returned by `__call__` (it is consumed internally to gather
# ligand_coords/types/mask before any downstream projection). So correctness is verified
# by comparing full per-residue node features `V` from the wide M=72 context against V from
# an independently-constructed "oracle" input pre-cropped (via a numpy argsort of exact
# Euclidean distances, an independent code path from the model's `jax.lax.top_k`) to exactly
# the ground-truth top-k atoms for that residue. Because ProteinFeaturesLigand.__call__ is a
# pure function of only the k *selected* ligand atoms' coordinates/types (unselected atoms
# provably cannot reach V at all -- see the third test below), exact numerical equivalence of
# V between the two runs is only possible if the wide-context run selected exactly the same
# set of atoms as the oracle.
# ---------------------------------------------------------------------------


def _line_ligand_atoms(num_atoms: int, spacing: float = 1.0) -> np.ndarray:
  """M atoms strung out along +x at `spacing` Angstrom increments, y=z=0."""
  x = np.arange(num_atoms) * spacing
  return np.stack([x, np.zeros(num_atoms), np.zeros(num_atoms)], axis=-1).astype(np.float32)


def _cb_targets_for_line(num_atoms: int, spacing: float = 1.0) -> np.ndarray:
  """3 Cb targets relative to `_line_ligand_atoms`: far-left, far-right, near-middle.

  Distances from each target to its nearest ~30 atoms stay within the RBF's calibrated
  [2, 22] Angstrom range (see ProteinFeaturesLigand._rbf) so downstream features are not
  saturated -- saturation would make a wrong atom selection numerically undetectable.
  """
  mid = (num_atoms - 1) / 2.0
  return np.array(
    [
      [-3.0, 0.0, 0.0],  # residue 0: nearest = smallest indices, in ascending-index order
      [(num_atoms - 1) * spacing + 3.0, 0.0, 0.0],  # residue 1: nearest = largest indices
      [mid * spacing + 0.37, 5.0, 0.0],  # residue 2: near middle; asymmetric offset avoids ties
    ],
  )


def _ground_truth_topk(
  cb_targets: np.ndarray,
  ligand_atoms: np.ndarray,
  k: int,
  ligand_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Independent (pure-numpy) nearest-k computation: argsort of exact Euclidean distance.

  Deliberately does not call jax.lax.top_k or any code from ligand_features.py -- this is
  the ground-truth oracle the model's own top-k selection is checked against.
  """
  dists = np.linalg.norm(cb_targets[:, None, :] - ligand_atoms[None, :, :], axis=-1)
  if ligand_mask is not None:
    dists = np.where(ligand_mask > 0, dists, np.inf)
  order = np.argsort(dists, axis=1)
  d_sorted = np.take_along_axis(dists, order, axis=1)
  boundary_gap = d_sorted[:, k] - d_sorted[:, k - 1]
  assert np.all(boundary_gap > 1e-6), (
    f"ground-truth top-{k} has a tie at the selection boundary for residue(s) "
    f"{np.nonzero(boundary_gap <= 1e-6)[0].tolist()} -- test geometry is ambiguous."
  )
  return order[:, :k]


def _make_features(atom_context_num: int, *, seed: int = 123) -> ProteinFeaturesLigand:
  return ProteinFeaturesLigand(
    node_features=32,
    edge_features=32,
    k_neighbors=3,
    atom_context_num=atom_context_num,
    ligand_l_chunk=-1,
    key=jax.random.PRNGKey(seed),
  )


@pytest.mark.parametrize("atom_context_num", [16, 25])
def test_topk_ligand_selection_matches_oracle_at_necklace_scale(atom_context_num: int) -> None:
  """At M=72 (necklace production scale), top-k selection picks the analytically-correct atoms.

  For 3 residues with exactly-known Cb positions, compares V from the real forward pass at
  M=72 against V from an oracle forward pass pre-cropped to the ground-truth top-k (a no-op
  M==atom_context_num case, like the pre-existing fixtures -- but now used as a checked
  oracle instead of the only case tested).
  """
  num_atoms = 72
  feat = _make_features(atom_context_num)

  cb_targets = _cb_targets_for_line(num_atoms)
  structure_coordinates = _make_backbone_for_cb_targets(cb_targets)
  num_residues = cb_targets.shape[0]
  mask = jnp.ones((num_residues,))
  residue_index = jnp.arange(num_residues)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

  ligand_atoms = _line_ligand_atoms(num_atoms)
  top_idx = _ground_truth_topk(cb_targets, ligand_atoms, atom_context_num)

  # Sanity-check residue 0's ground truth is exactly the hand-predicted set: nearest atoms
  # to a Cb 3 Angstrom to the "left" of a line starting at atom 0 are atoms 0..k-1.
  assert sorted(top_idx[0].tolist()) == list(range(atom_context_num))
  # Residue 1 (mirrored): nearest atoms are the last k atoms on the line.
  assert sorted(top_idx[1].tolist()) == list(range(num_atoms - atom_context_num, num_atoms))

  ligand_coords_wide = jnp.broadcast_to(jnp.asarray(ligand_atoms)[None], (num_residues, num_atoms, 3))
  types_wide = jnp.zeros((num_residues, num_atoms), dtype=jnp.int32)
  mask_wide = jnp.ones((num_residues, num_atoms))

  v_wide, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    ligand_coords_wide,
    types_wide,
    mask_wide,
    0.0,
    None,
  )
  assert v_wide.shape == (num_residues, atom_context_num, 32)
  assert np.all(np.isfinite(np.asarray(v_wide))), "no shape blowup/NaN at M=72 necklace scale"

  ligand_coords_oracle = jnp.asarray(ligand_atoms[top_idx])
  types_oracle = jnp.zeros((num_residues, atom_context_num), dtype=jnp.int32)
  mask_oracle = jnp.ones((num_residues, atom_context_num))

  v_oracle, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    ligand_coords_oracle,
    types_oracle,
    mask_oracle,
    0.0,
    None,
  )

  np.testing.assert_allclose(
    np.asarray(v_wide),
    np.asarray(v_oracle),
    rtol=1e-5,
    atol=1e-5,
    err_msg=(
      f"V from the wide M={num_atoms} context does not match V from the ground-truth "
      f"top-{atom_context_num} oracle crop -- the real top-k selection picked the wrong "
      "atoms at necklace production scale."
    ),
  )


def test_topk_ligand_selection_respects_ligand_mask() -> None:
  """Masked-out atoms are excluded from top-k selection even when geometrically nearest.

  Masks out the single nearest atom to residue 0's Cb; the correct new top-16 must be the
  *next* 16 nearest (atoms 1..16), not e.g. the original top-16 including the masked atom.
  """
  num_atoms, k = 72, 16
  feat = _make_features(k)

  cb_targets = _cb_targets_for_line(num_atoms)
  structure_coordinates = _make_backbone_for_cb_targets(cb_targets)
  num_residues = cb_targets.shape[0]
  mask = jnp.ones((num_residues,))
  residue_index = jnp.arange(num_residues)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

  ligand_atoms = _line_ligand_atoms(num_atoms)
  ligand_mask_np = np.ones((num_residues, num_atoms), dtype=np.float32)
  ligand_mask_np[0, 0] = 0.0  # mask out residue 0's single nearest atom

  top_idx = _ground_truth_topk(cb_targets, ligand_atoms, k, ligand_mask=ligand_mask_np)
  assert 0 not in top_idx[0].tolist(), "ground-truth oracle setup error: masked atom leaked in"
  assert sorted(top_idx[0].tolist()) == list(range(1, k + 1))

  ligand_coords_wide = jnp.broadcast_to(jnp.asarray(ligand_atoms)[None], (num_residues, num_atoms, 3))
  types_wide = jnp.zeros((num_residues, num_atoms), dtype=jnp.int32)

  v_masked, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    ligand_coords_wide,
    types_wide,
    jnp.asarray(ligand_mask_np),
    0.0,
    None,
  )

  ligand_coords_oracle = jnp.asarray(ligand_atoms[top_idx])
  types_oracle = jnp.zeros((num_residues, k), dtype=jnp.int32)
  mask_oracle = jnp.ones((num_residues, k))

  v_oracle, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    ligand_coords_oracle,
    types_oracle,
    mask_oracle,
    0.0,
    None,
  )

  np.testing.assert_allclose(
    np.asarray(v_masked),
    np.asarray(v_oracle),
    rtol=1e-5,
    atol=1e-5,
    err_msg="masked-out nearest atom was not correctly excluded from top-k selection.",
  )


def test_topk_ligand_selection_only_selected_atoms_influence_node_features() -> None:
  """Differential check: perturbing a selected atom changes V; perturbing an excluded one does not.

  This is a non-vacuous confirmation that the oracle-matching tests above are actually
  sensitive to selection correctness, and not an artifact of some degenerate no-op path
  (e.g. RBF saturation): distances here are kept within the RBF's calibrated [2, 22]
  Angstrom range so a wrong selection is numerically detectable.
  """
  num_atoms, k = 72, 16
  feat = _make_features(k)

  cb_targets = _cb_targets_for_line(num_atoms)
  structure_coordinates = _make_backbone_for_cb_targets(cb_targets)
  num_residues = cb_targets.shape[0]
  mask = jnp.ones((num_residues,))
  residue_index = jnp.arange(num_residues)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

  ligand_atoms = _line_ligand_atoms(num_atoms)
  top_idx = _ground_truth_topk(cb_targets, ligand_atoms, k)

  types_wide = jnp.zeros((num_residues, num_atoms), dtype=jnp.int32)
  mask_wide = jnp.ones((num_residues, num_atoms))
  ligand_coords_wide = jnp.broadcast_to(jnp.asarray(ligand_atoms)[None], (num_residues, num_atoms, 3))

  v_wide, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    ligand_coords_wide,
    types_wide,
    mask_wide,
    0.0,
    None,
  )

  # Mutate the farthest currently-selected atom for residue 0 far outside the RBF range --
  # this must knock it out of the top-16 (replaced by the next-nearest unselected atom) and
  # change V[0].
  selected_boundary_atom = int(top_idx[0, -1])
  mutated_selected = ligand_atoms.copy()
  mutated_selected[selected_boundary_atom, 0] += 50.0
  v_mut_selected, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    jnp.broadcast_to(jnp.asarray(mutated_selected)[None], (num_residues, num_atoms, 3)),
    types_wide,
    mask_wide,
    0.0,
    None,
  )
  changed = np.abs(np.asarray(v_mut_selected[0]) - np.asarray(v_wide[0])).max()
  assert changed > 1e-3, (
    "mutating a selected boundary atom did not change V -- top-k selection may not be "
    "responding to distance at necklace scale."
  )

  # Mutate an atom that is nowhere near residue 0's Cb (already unselected, already far
  # outside the RBF range) -- V[0] must be completely unaffected, since unselected atoms
  # cannot reach V by construction (they are dropped before any downstream projection).
  far_unselected_atom = 50
  assert far_unselected_atom not in top_idx[0].tolist()
  mutated_unselected = ligand_atoms.copy()
  mutated_unselected[far_unselected_atom, 0] += 50.0
  v_mut_unselected, *_ = feat(
    jax.random.PRNGKey(0),
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    jnp.broadcast_to(jnp.asarray(mutated_unselected)[None], (num_residues, num_atoms, 3)),
    types_wide,
    mask_wide,
    0.0,
    None,
  )
  unchanged = np.abs(np.asarray(v_mut_unselected[0]) - np.asarray(v_wide[0])).max()
  assert unchanged == 0.0, (
    "mutating an atom far outside residue 0's top-k selection changed V[0] -- an "
    "unselected atom is leaking into the selected feature computation."
  )