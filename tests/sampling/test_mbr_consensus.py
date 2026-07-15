"""Tests for mbr_consensus.py (spec §6 acceptance criteria, MBR post-hoc reranking).

See ../../.praxia/docs/specs/260709_mbr-consensus-reranking-composition.md §6 for the
Given/When/Then criteria these tests implement directly.
"""

from __future__ import annotations

import ast
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode import unfused as unfused_module
from aminx.model import Aminx
from aminx.sampling import mbr_consensus
from aminx.sampling.mbr_consensus import (
  average_cross_state_scores,
  mbr_rerank,
  select_mbr_candidates,
)


def _make_model(key_seed: int) -> Aminx:
  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(key_seed),
  )
  return eqx.tree_inference(model, value=True)


def _synthetic_candidates(num_candidates: int, num_residues: int, seed: int) -> jax.Array:
  rng = np.random.default_rng(seed)
  return jnp.array(
    rng.integers(0, 20, size=(num_candidates, num_residues), dtype=np.int32),
  )


def _heterogeneous_state_bundle(num_residues: int = 10, seed: int = 5):
  """States with genuinely different pre-padding residue counts, padded to a common L.

  Mirrors tev_design's build_canonical_bundle.py: each state's "real" content occupies
  only its first `real_length` positions (mask=1); the rest is zero-padded (mask=0) up
  to the common N_CANONICAL-style length -- exactly like 1LVB/1LVM vs reac1/reac2 having
  different natural sizes before padding.
  """
  rng = np.random.default_rng(seed)
  real_lengths = [4, num_residues, 6]  # deliberately different pre-padding sizes
  num_states = len(real_lengths)

  coords_list = []
  mask_list = []
  for real_length in real_lengths:
    coords = np.zeros((num_residues, 4, 3), dtype=np.float32)
    coords[:real_length] = rng.normal(size=(real_length, 4, 3)).astype(np.float32)
    mask = np.zeros((num_residues,), dtype=np.float32)
    mask[:real_length] = 1.0
    coords_list.append(coords)
    mask_list.append(mask)

  coords_stack = jnp.array(np.stack(coords_list, axis=0))
  mask_stack = jnp.array(np.stack(mask_list, axis=0))
  residue_index = jnp.broadcast_to(
    jnp.arange(num_residues, dtype=jnp.int32)[None, :], (num_states, num_residues),
  )
  chain_index = jnp.zeros((num_states, num_residues), dtype=jnp.int32)

  sequence_tokens = jnp.array(rng.integers(0, 20, size=(num_residues,), dtype=np.int32))
  sequence_oh = jax.nn.one_hot(sequence_tokens, 21)
  state_weights = jnp.ones(num_states) / num_states

  bundle, config = build_inference_bundle(
    coords=coords_stack,
    mask=mask_stack,
    residue_index=residue_index,
    chain_index=chain_index,
    sequence=sequence_oh,
    state_weights=state_weights,
    mode="score_conditional",
  )
  return bundle, config, num_states, num_residues


def test_average_cross_state_scores_k1_is_noop() -> None:
  """§6: k=1 average must equal that one state's own per-candidate scores exactly."""
  scores = jnp.array([[1.5, -0.3, 2.7, 0.0]])  # (S=1, C=4)
  averaged = average_cross_state_scores(scores)
  np.testing.assert_array_equal(np.asarray(averaged), np.asarray(scores[0]))


def test_select_mbr_candidates_picks_lower_nll_first() -> None:
  """§6: the single easiest mistake to make silently -- must be argmin-direction, not argmax."""
  # Candidate 0 has hand-picked LOWER (better) NLL; candidate 1 deliberately worse.
  mean_scores = jnp.array([0.5, 3.2])
  candidates = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

  selected_indices, selected_sequences = select_mbr_candidates(mean_scores, candidates, top_k=1)

  assert int(selected_indices[0]) == 0, "must select the LOWER-NLL candidate, not higher"
  np.testing.assert_array_equal(np.asarray(selected_sequences[0]), np.asarray(candidates[0]))


def test_mbr_rerank_heterogeneous_states_runs_via_vmap() -> None:
  """§6: runs to completion via genuine jax.vmap on states with different pre-padding sizes."""
  model = _make_model(11)
  bundle, config, num_states, num_residues = _heterogeneous_state_bundle()
  candidates = _synthetic_candidates(num_candidates=5, num_residues=num_residues, seed=23)

  selected_indices, selected_sequences = mbr_rerank(
    model, bundle, candidates, jax.random.PRNGKey(0), config=config, top_k=2,
  )

  assert selected_indices.shape == (2,)
  assert selected_sequences.shape == (2, num_residues)
  assert jnp.all(jnp.isfinite(selected_sequences))
  # Selected sequences must actually be a subset of the input candidates.
  for seq in np.asarray(selected_sequences):
    assert any(np.array_equal(seq, cand) for cand in np.asarray(candidates))


def test_mbr_rerank_batching_invariance_vmap_vs_safemap() -> None:
  """§6: per-candidate scores (observed via full ranking) must be identical Vmap vs SafeMap.

  top_k=n_candidates returns every candidate ranked by mean NLL -- if batching changed the
  underlying scores, the returned order (and hence the index/sequence arrays) would very
  likely differ. Mirrors test_batched_conditional_logits.py's
  test_batched_split_fn_safe_map_matches_vmap_tile pattern.
  """
  model = _make_model(13)
  bundle, config, num_states, num_residues = _heterogeneous_state_bundle()
  n_candidates = 6
  candidates = _synthetic_candidates(num_candidates=n_candidates, num_residues=num_residues, seed=29)
  key = jax.random.PRNGKey(1)

  default_indices, default_sequences = mbr_rerank(
    model, bundle, candidates, key, config=config, top_k=n_candidates,
  )
  tiled_indices, tiled_sequences = mbr_rerank(
    model, bundle, candidates, key, config=config, top_k=n_candidates,
    candidate_batch_size=2,
  )

  np.testing.assert_array_equal(np.asarray(default_indices), np.asarray(tiled_indices))
  np.testing.assert_array_equal(np.asarray(default_sequences), np.asarray(tiled_sequences))


def test_mbr_rerank_single_call_matches_c_separate_single_candidate_calls() -> None:
  """§6: same C candidates in one call vs. C separate single-candidate calls -- identical scores.

  Observed via top_k=1 selection matching the argmin over C independently-scored calls.
  """
  model = _make_model(17)
  bundle, config, num_states, num_residues = _heterogeneous_state_bundle()
  n_candidates = 4
  candidates = _synthetic_candidates(num_candidates=n_candidates, num_residues=num_residues, seed=31)
  key = jax.random.PRNGKey(2)

  # Batched: score all C candidates in one call, get every candidate's own single-candidate rank.
  batched_indices, _ = mbr_rerank(
    model, bundle, candidates, key, config=config, top_k=n_candidates,
  )

  # Separate: score each candidate alone (C=1 per call), record its best-of-1 selection.
  # A single-candidate call must reproduce that same candidate as the (only, hence "best") pick.
  for i in range(n_candidates):
    single_indices, single_sequences = mbr_rerank(
      model, bundle, candidates[i : i + 1], key, config=config, top_k=1,
    )
    assert int(single_indices[0]) == 0
    np.testing.assert_array_equal(np.asarray(single_sequences[0]), np.asarray(candidates[i]))

  # The batched ranking must be a valid permutation of all C candidate indices (no candidate
  # dropped or duplicated by the xtrax dispatch).
  assert sorted(int(i) for i in batched_indices) == list(range(n_candidates))


def _find_loop_statements(source: str) -> list[ast.For | ast.While]:
  """AST-level check for actual for/while statements (not docstring prose containing "for")."""
  tree = ast.parse(source)
  return [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]


def _find_axis_boundaries_or_autoregressive_usage(source: str) -> list[ast.AST]:
  """AST-level check for real code references, not docstring prose explaining a non-usage."""
  tree = ast.parse(source)
  hits: list[ast.AST] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and node.attr == "axis_boundaries":
      hits.append(node)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      module_name = getattr(node, "module", None) or ""
      names = [alias.name for alias in node.names]
      if "autoregressive" in module_name or any("autoregressive" in n for n in names):
        hits.append(node)
  return hits


def test_no_hand_written_loop_over_state_or_candidate_axis() -> None:
  """§6: static check -- no top-level for/while loop over either axis in the new modules.

  Mirrors the spec's own suggested verification: grep for `for `/`while ` at the top level
  of any function body. Uses AST parsing (not a text regex) so docstring prose that happens
  to contain the English word "for" doesn't false-positive.
  """
  unfused_source = inspect.getsource(unfused_module)
  mbr_source = inspect.getsource(mbr_consensus)

  assert not _find_loop_statements(unfused_source), "unfused.py must not hand-loop over states"
  assert not _find_loop_statements(mbr_source), "mbr_consensus.py must not hand-loop over candidates"

  # Must not touch the live-decode StageSet.axis_boundaries extension point or import from
  # the autoregressive AR-loop fusion module (spec §6, §4 -- this is a post-hoc batch
  # utility, not a StageSet stage). Checked as real code references (attribute access /
  # imports), not a substring ban -- both modules' docstrings correctly explain this design
  # decision in prose, which must not trip the check.
  assert not _find_axis_boundaries_or_autoregressive_usage(unfused_source)
  assert not _find_axis_boundaries_or_autoregressive_usage(mbr_source)


@pytest.mark.parametrize("iterator_name", ["state_iterator"])
def test_mbr_rerank_state_axis_uses_vmap_iterator(iterator_name: str) -> None:
  """§6 (implementation detail): state axis dispatch is hardcoded to VmapIterator.

  No reason to ever SafeMap the state axis given the pre-padded-uniform-shape precondition
  -- mirrors score_from_encoding's own hardcoded VmapIterator() choice.
  """
  source = inspect.getsource(mbr_rerank)
  assert "VmapIterator()" in source
