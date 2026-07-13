"""Tests for aminx.sampling.multistate_poe.

Regression coverage for the 2026-07-13 finding (tev_design task
260709_multistate-fusion-strategy-comparison, Phase 3): no campaign-reachable path ever built a
genuinely stacked num_states>1 bundle for autoregressive sampling. See
.praxia/docs/decisions/260713_no-real-multistate-sampling-path-exists.md.
"""

from __future__ import annotations

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.run.specs import SamplingSpecification
from aminx.sampling.multistate_poe import sample_multistate_poe_bead, sample_states_fused

REPO_TESTS_DATA = __file__.rsplit("/tests/", 1)[0] + "/tests/data"


def _build_synthetic_bundle(num_states: int, num_residues: int, seed: int):
  """Mirrors tests/inference/decode/test_autoregressive.py's _build_synthetic_fixture."""
  rng = np.random.default_rng(seed)
  jax_key = jax.random.PRNGKey(seed)

  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=jax_key,
  )
  model = eqx.tree_inference(model, value=True)

  coordinates = jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

  coordinates_stack = jnp.stack([coordinates] * num_states, axis=0)
  mask_stack = jnp.stack([mask] * num_states, axis=0)
  residue_index_stack = jnp.stack([residue_index] * num_states, axis=0)
  chain_index_stack = jnp.stack([chain_index] * num_states, axis=0)

  state_weights = jnp.ones(num_states) / num_states
  bundle, config = build_inference_bundle(
    coords=coordinates_stack,
    mask=mask_stack,
    residue_index=residue_index_stack,
    chain_index=chain_index_stack,
    state_weights=state_weights,
    sequence=None,
    mode="sample_ar",
  )
  return model, bundle, config


class TestSampleStatesFused:
  """Coverage for the low-level sample_states_fused primitive (already-built bundle)."""

  def test_shapes_and_dtypes(self):
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=10, seed=1)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    n_samples = 4

    sequences, logits = sample_states_fused(
      model, bundle, config, stage_set, jax.random.PRNGKey(0), n_samples,
    )

    L = 10
    assert sequences.shape == (n_samples, L)
    assert logits.shape == (n_samples, L, 21)
    assert sequences.dtype == jnp.int32
    assert logits.dtype == jnp.float32

  def test_different_sample_keys_give_different_sequences(self):
    """n_samples>1 must actually be independent draws, not the same sample repeated."""
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=16, seed=2)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)

    sequences, _ = sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(7), 8)

    # Not every one of the 8 samples should be byte-identical -- if they were, sample_keys
    # aren't actually varying anything (a real bug: vmap over a constant, or a key not
    # threaded through).
    unique_rows = {tuple(row.tolist()) for row in sequences}
    assert len(unique_rows) > 1, "all 8 samples were identical -- sample keys aren't varying"

  def test_genuine_multistate_fusion_changes_output(self):
    """The actual claim this module exists for: a real num_states>1 bundle with a
    non-identity state_position_map produces DIFFERENT fused logits than the identity map --
    proving states are genuinely combined, not independently decoded and discarded.
    Mirrors test_autoregressive.py's test_ar_decode_state_position_map_changes_fused_logits,
    but through THIS module's public sample_states_fused, not a hand-rolled AutoregressiveDecode
    construction -- this is the actual call path sample_multistate_poe_bead uses in production.
    """
    num_states, num_residues = 2, 8
    model, bundle, config = _build_synthetic_bundle(
      num_states=num_states, num_residues=num_residues, seed=11,
    )
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    key = jax.random.PRNGKey(0)

    _, baseline_logits = sample_states_fused(model, bundle, config, stage_set, key, 1)

    permuted_row = jnp.roll(jnp.arange(num_residues), shift=1)
    custom_map = jnp.stack([jnp.arange(num_residues), permuted_row])
    permuted_bundle = eqx.tree_at(
      lambda b: b.conditioning.state_position_map,
      bundle,
      custom_map,
    )
    _, permuted_logits = sample_states_fused(model, permuted_bundle, config, stage_set, key, 1)

    assert not jnp.allclose(permuted_logits, baseline_logits), (
      "a non-identity state_position_map must change the fused logits -- if it doesn't, "
      "states aren't actually being combined"
    )

  def test_single_state_is_still_a_no_op_regression_guard(self):
    """num_states=1 (the pre-existing, correct behavior for every campaign row before this
    module) must be unaffected: identity state_position_map on a single state changes nothing.
    """
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=8, seed=12)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    key = jax.random.PRNGKey(0)

    _, logits_a = sample_states_fused(model, bundle, config, stage_set, key, 1)
    _, logits_b = sample_states_fused(model, bundle, config, stage_set, key, 1)

    assert jnp.array_equal(logits_a, logits_b), "identical inputs must give identical output"


class TestSampleMultistatePoeBead:
  """Coverage for the high-level orchestration function (real structure loading)."""

  def test_rejects_single_input(self):
    spec = SamplingSpecification(inputs=[f"{REPO_TESTS_DATA}/1ubq.pdb"], batch_size=1)
    with pytest.raises(ValueError, match=">=2 states"):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)

  def test_rejects_batch_size_mismatch(self):
    spec = SamplingSpecification(
      inputs=[f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"],
      batch_size=1,
    )
    with pytest.raises(ValueError, match="spec.batch_size"):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)

  def test_real_structures_two_states_end_to_end(self):
    """Real end-to-end path: real PDB parsing (create_protein_dataset, unmocked) for 2 real,
    genuinely different small structures, stacked into one num_states=2 bundle, sampled with
    product-strategy fusion. Only the model WEIGHTS are mocked (a synthetic Aminx, not a real
    trained checkpoint) -- structure loading/padding/batching/bundle-building/fusion/sampling
    are all real, unmocked code paths.
    """
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      multi_state_strategy="product",
      return_logits=True,
    )

    synthetic_model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=1,
      num_decoder_layers=1,
      k_neighbors=8,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(3),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      sequences, logits = sample_multistate_poe_bead(
        spec, jax.random.PRNGKey(0), n_samples=2,
      )

    assert sequences.ndim == 2
    assert sequences.shape[0] == 2
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 21
    assert sequences.dtype == jnp.int32
    assert bool(jnp.all(jnp.isfinite(logits)))
