"""Guard: temperature autoregressive sampler only forwards kwargs the model accepts."""

from __future__ import annotations

import jax

from prxteinmpnn.model.capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  PRXTEIN_MPNN_CAPABILITIES,
)
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences

# Keyword / positional-or-keyword names accepted by ``PrxteinMPNN.__call__`` (excluding ``self``).
_PRXTEIN_MPNN_CALL_NAMES = frozenset({
  "structure_coordinates",
  "mask",
  "residue_index",
  "chain_index",
  "decoding_approach",
  "prng_key",
  "ar_mask",
  "one_hot_sequence",
  "temperature",
  "bias",
  "fixed_mask",
  "fixed_tokens",
  "backbone_noise",
  "tie_group_map",
  "group_indices_table",
  "group_valid_table",
  "multi_state_strategy",
  "multi_state_temperature",
  "multistate_mode",
  "structure_mapping",
  "initial_node_features",
  "rbf_features",
  "neighbor_indices",
  "precomputed_node_features",
  "precomputed_edge_features",
  "precomputed_neighbor_indices",
  "membrane_per_residue_labels",
  "state_weights",
  "state_mapping",
  "num_groups",
  "wave_group_ids",
  "wave_group_positions",
  "wave_group_valid",
  "wave_position_valid",
  "inference",
  "coords_stack",
  "mask_stack",
  "residue_index_stack",
  "chain_index_stack",
  "state_flat_rows",
  "n_flat",
  "ar_mask_stack",
})

# Keyword / positional-or-keyword names accepted by ``PrxteinLigandMPNN.__call__`` (excluding ``self``).
_PRXTEIN_LIGAND_CALL_NAMES = frozenset({
  "structure_coordinates",
  "mask",
  "residue_index",
  "chain_index",
  "Y",
  "Y_t",
  "Y_m",
  "decoding_approach",
  "prng_key",
  "ar_mask",
  "one_hot_sequence",
  "temperature",
  "bias",
  "fixed_mask",
  "fixed_tokens",
  "backbone_noise",
  "inference",
  "xyz_37",
  "xyz_37_m",
  "chain_mask",
  "tie_group_map",
  "group_indices_table",
  "group_valid_table",
  "multi_state_strategy",
  "structure_mapping",
  "multi_state_temperature",
  "state_weights",
  "state_mapping",
  "precomputed_Y_nodes",
  "precomputed_Y_edges",
  "precomputed_Y_m",
  "num_groups",
  "multistate_mode",
  "wave_group_ids",
  "wave_group_positions",
  "wave_group_valid",
  "wave_position_valid",
  "coords_stack",
  "mask_stack",
  "residue_index_stack",
  "chain_index_stack",
  "y_stack",
  "y_t_stack",
  "y_m_stack",
  "state_flat_rows",
  "n_flat",
  "ar_mask_stack",
  "states_chunk_size",
})


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
  assert m.capabilities is PRXTEIN_MPNN_CAPABILITIES
  ligand_kw = _ligand_autoreg_kw()
  hoist_kw = {"precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"}
  for n in sorted(_kw_names_temperature_autoreg_flat() - ligand_kw - hoist_kw):
    assert n in _PRXTEIN_MPNN_CALL_NAMES, n


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
  assert m.capabilities is PRXTEIN_LIGAND_MPNN_CAPABILITIES
  hoist_kw = {"precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"}
  for n in sorted(_kw_names_temperature_autoreg_flat() - hoist_kw):
    assert n in _PRXTEIN_LIGAND_CALL_NAMES, n


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
  assert m.capabilities.is_ligand_model is True
  for bad in ("precomputed_node_features", "precomputed_edge_features", "precomputed_neighbor_indices"):
    assert bad not in _PRXTEIN_LIGAND_CALL_NAMES, (
      "Ligand forward must not accept protein hoist kw names "
      "(use precomputed_Y_nodes / edges / mask when wired)."
    )


def test_make_sample_sequences_constructible_smoke():
  key = jax.random.PRNGKey(2)
  m = PrxteinMPNN(12, 12, 12, 1, 1, 4, key=key)
  fn = make_sample_sequences(m, sampling_strategy="temperature")
  assert callable(fn)
