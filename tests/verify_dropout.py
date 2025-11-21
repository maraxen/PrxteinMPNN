import jax
import jax.numpy as jnp
import equinox as eqx
from prxteinmpnn.model.mpnn import PrxteinMPNN

def verify_dropout():
    print("Verifying Dropout Implementation...")
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    training_model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=30,
        dropout_rate=0.5, # High dropout to ensure differences
        key=key
    )
    
    # Create inference model using eqx.nn.inference_mode
    inference_model = eqx.nn.inference_mode(training_model)
    
    # Create dummy inputs (single example, not batched)
    num_residues = 10
    coords = jnp.zeros((num_residues, 4, 3))
    mask = jnp.ones((num_residues,))
    residue_idx = jnp.arange(num_residues)
    chain_idx = jnp.zeros((num_residues,), dtype=jnp.int32)
    
    # Test 1: Inference Determinism (Dropout OFF)
    print("\nTest 1: Inference Determinism (Dropout OFF)")
    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(2)
    
    _, logits_inf_1 = inference_model(
        coords, mask, residue_idx, chain_idx, 
        decoding_approach="unconditional", 
        prng_key=key1
    )
    
    _, logits_inf_2 = inference_model(
        coords, mask, residue_idx, chain_idx, 
        decoding_approach="unconditional", 
        prng_key=key2
    )
    
    diff_inf = jnp.max(jnp.abs(logits_inf_1 - logits_inf_2))
    print(f"Max difference between inference runs with different keys: {diff_inf}")
    
    if diff_inf < 1e-5:
        print("✓ PASS: Inference is deterministic (dropout disabled).")
    else:
        print("✗ FAIL: Inference is NOT deterministic (dropout might be active).")
        
    # Test 2: Training Stochasticity (Dropout ON)
    print("\nTest 2: Training Stochasticity (Dropout ON)")
    
    _, logits_train_1 = training_model(
        coords, mask, residue_idx, chain_idx, 
        decoding_approach="unconditional", 
        prng_key=key1
    )
    
    _, logits_train_2 = training_model(
        coords, mask, residue_idx, chain_idx, 
        decoding_approach="unconditional", 
        prng_key=key2
    )
    
    diff_train = jnp.max(jnp.abs(logits_train_1 - logits_train_2))
    print(f"Max difference between training runs with different keys: {diff_train}")
    
    if diff_train > 1e-5:
        print("✓ PASS: Training is stochastic (dropout enabled).")
    else:
        print("✗ FAIL: Training is NOT stochastic (dropout might be disabled or rate is 0).")

    # Test 3: Training Determinism with SAME key
    print("\nTest 3: Training Determinism with SAME key")
    _, logits_train_3 = training_model(
        coords, mask, residue_idx, chain_idx, 
        decoding_approach="unconditional", 
        prng_key=key1
    )
    
    diff_same_key = jnp.max(jnp.abs(logits_train_1 - logits_train_3))
    print(f"Max difference between training runs with SAME key: {diff_same_key}")
    
    if diff_same_key < 1e-5:
        print("✓ PASS: Training is deterministic with same key.")
    else:
        print("✗ FAIL: Training is NOT deterministic with same key (something else is changing).")
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)

if __name__ == "__main__":
    verify_dropout()
