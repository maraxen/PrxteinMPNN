"""Guard: temperature autoregressive sampler only forwards kwargs the model accepts."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences


def _kw_names_temperature_autoreg_flat() -> set[str]:
  """Frozen set reflecting ``sample_sequences`` branch that builds ``call_kwargs``."""
  protein_only = {"precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"}
  common = {
    "decoding_approach",
    "prng_key",
    "ar_mask",
    "temperature",
    "bias",
    "backbone_noise",
    "tie_group_map",
    "num_groups",
    "multi_state_strategy",
    "multistate_mode",
    "structure_mapping",
    "group_indices_table",
    "group_valid_table",
    "wave_group_ids",
    "wave_group_positions",
    "wave_group_valid",
    "wave_position_valid",
    "multi_state_temperature",
    "state_weights",
    "state_mapping",
    "fixed_mask",
    "fixed_tokens",
    "Y",
    "Y_t",
    "Y_m",
    "xyz_37",
    "xyz_37_m",
    "chain_mask",
  }
  return common | protein_only


def _ligand_autoreg_kw() -> set[str]:
  return {"Y", "Y_t", "Y_m", "xyz_37", "xyz_37_m", "chain_mask"}


def test_temperature_autoreg_call_kwargs_subset_prxteinmpnn():
  key = jax.random.PRNGKey(0)
  m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)
  sig = inspect.signature(m.__call__)
  ligand_kw = _ligand_autoreg_kw()
  hoist_kw = {"precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"}
  for n in sorted(_kw_names_temperature_autoreg_flat() - ligand_kw - hoist_kw):
    assert n in sig.parameters, n


def test_temperature_autoreg_call_kwargs_subset_ligand():
  key = jax.random.PRNGKey(3)
  m = PrxteinLigandMPNN(
    16,
    16,
    16,
    1,
    1,
    6,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  sig = inspect.signature(m.__call__)
  hoist_kw = {"precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"}
  for n in sorted(_kw_names_temperature_autoreg_flat() - hoist_kw):
    assert n in sig.parameters, n


def test_temperature_autoreg_no_ligand_precomputed_node_aliases():
  key = jax.random.PRNGKey(1)
  m = PrxteinLigandMPNN(
    16,
    16,
    16,
    1,
    1,
    6,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  sig = inspect.signature(m.__call__)
  for bad in ("precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"):
    assert bad not in sig.parameters, (
      "Ligand forward must not accept protein hoist kw names "
      "(use precomputed_Y_nodes / edges / mask when wired)."
    )


def test_make_sample_sequences_constructible_smoke():
  key = jax.random.PRNGKey(2)
  m = PrxteinMPNN(12, 12, 12, 1, 1, 4, key=key)
  fn = make_sample_sequences(m, sampling_strategy="temperature")
  assert callable(fn)
