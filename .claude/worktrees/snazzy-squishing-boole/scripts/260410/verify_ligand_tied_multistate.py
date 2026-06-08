"""Tests for tied-position sampling in PrxteinLigandMPNN."""

import jax
import jax.numpy as jnp
import equinox as eqx
from prxteinmpnn.model.mpnn import PrxteinLigandMPNN

def test_ligand_tied_sampling():
  key = jax.random.PRNGKey(0)
  model = PrxteinLigandMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    key=key,
  )
  
  num_res = 10
  coords = jnp.zeros((num_res, 4, 3))
  mask = jnp.ones((num_res,))
  residue_index = jnp.arange(num_res)
  chain_index = jnp.zeros((num_res,), dtype=jnp.int32)
  
  # Ligand context
  Y = jnp.zeros((num_res, 1, 3))
  Y_t = jnp.zeros((num_res, 1), dtype=jnp.int32)
  Y_m = jnp.zeros((num_res, 1))
  
  # Tie positions 0 and 1, 2 and 3
  tie_group_map = jnp.arange(num_res)
  tie_group_map = tie_group_map.at[1].set(0)
  tie_group_map = tie_group_map.at[3].set(2)
  
  seq, logits = model(
    coords, mask, residue_index, chain_index,
    Y, Y_t, Y_m,
    decoding_approach="autoregressive",
    tie_group_map=tie_group_map,
    temperature=0.1,
    prng_key=key,
  )
  
  # Check if tied positions have same sequence
  sampled_seq = seq.argmax(axis=-1)
  assert sampled_seq[0] == sampled_seq[1]
  assert sampled_seq[2] == sampled_seq[3]
  
  # Tied positions should have same logits (the combined ones)
  assert jnp.allclose(logits[0], logits[1])
  assert jnp.allclose(logits[2], logits[3])

def test_ligand_weighted_multistate_sampling():
  key = jax.random.PRNGKey(0)
  model = PrxteinLigandMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    key=key,
  )
  
  # 2 states, 5 residues each
  num_res_total = 10
  coords = jnp.zeros((num_res_total, 4, 3))
  mask = jnp.ones((num_res_total,))
  residue_index = jnp.concatenate([jnp.arange(5), jnp.arange(5)])
  chain_index = jnp.zeros((num_res_total,), dtype=jnp.int32)
  
  # Tie residue i in state 0 to residue i in state 1
  tie_group_map = jnp.concatenate([jnp.arange(5), jnp.arange(5)])
  # state_mapping identifies which state each residue belongs to
  state_mapping = jnp.concatenate([jnp.zeros(5, dtype=jnp.int32), jnp.ones(5, dtype=jnp.int32)])
  state_weights = jnp.array([0.8, 0.2])
  
  Y = jnp.zeros((num_res_total, 1, 3))
  Y_t = jnp.zeros((num_res_total, 1), dtype=jnp.int32)
  Y_m = jnp.zeros((num_res_total, 1))
  
  seq, logits = model(
    coords, mask, residue_index, chain_index,
    Y, Y_t, Y_m,
    decoding_approach="autoregressive",
    tie_group_map=tie_group_map,
    state_weights=state_weights,
    state_mapping=state_mapping,
    temperature=0.1,
    prng_key=key,
  )
  
  sampled_seq = seq.argmax(axis=-1)
  # Residue 0 in state 0 and state 1 should be same
  for i in range(5):
    assert sampled_seq[i] == sampled_seq[i+5]
    assert jnp.allclose(logits[i], logits[i+5])
