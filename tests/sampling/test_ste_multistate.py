"""Test straight-through estimator optimization for multi-state design."""

import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn

def test_optimize_sequence_multistate():
    """Verify that optimize_sequence can handle multi-state parameters."""
    key = jax.random.PRNGKey(42)
    
    # Simple model configuration
    model = PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        key=key
    )
    
    # Dummy data for 2 states of 10 residues each
    n_res = 10
    n_states = 2
    n_total = n_res * n_states
    
    coords = jnp.zeros((n_total, 4, 3))
    mask = jnp.ones(n_total)
    res_idx = jnp.tile(jnp.arange(n_res), n_states)
    chain_idx = jnp.zeros(n_total, dtype=jnp.int32)
    
    # Multi-state mapping
    structure_mapping = jnp.concatenate([jnp.zeros(n_res), jnp.ones(n_res)]).astype(jnp.int32)
    state_weights = jnp.array([1.0, 2.0])
    
    # Tied positions: residue i in state 0 tied to residue i in state 1
    tie_group_map = jnp.tile(jnp.arange(n_res), n_states)
    num_groups = n_res
    
    optimize_fn = make_optimize_sequence_fn(model, batch_size=2)
    
    # Run optimization
    seq, logits, opt_logits = optimize_fn(
        key,
        coords,
        mask,
        res_idx,
        chain_idx,
        iterations=5,
        learning_rate=0.1,
        temperature=1.0,
        tie_group_map=tie_group_map,
        num_groups=num_groups,
        structure_mapping=structure_mapping,
        multi_state_strategy="arithmetic_mean",
        state_weights=state_weights
    )
    
    # Verify outputs
    assert seq.shape == (n_total,)
    assert logits.shape == (n_total, 21)
    
    # Verify tied positions have identical sequences
    for i in range(n_res):
        assert seq[i] == seq[i + n_res]
    
    # Verify logits are identical for tied positions (due to arithmetic_mean internal logit combining)
    for i in range(n_res):
        assert jnp.allclose(logits[i], logits[i + n_res], atol=1e-5)

if __name__ == "__main__":
    test_optimize_sequence_multistate()
