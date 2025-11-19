"""Test script for diffusion training loop."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
from prxteinmpnn.training.diffusion import NoiseSchedule
from prxteinmpnn.training.train_diffusion import train_step
from prxteinmpnn.io.loaders import create_protein_dataset

def test_diffusion_training():
    print("Initializing model and optimizer...")
    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)
    
    model = DiffusionPrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=30,
        key=model_key,
    )
    
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    noise_schedule = NoiseSchedule(num_steps=100)
    print(f"Schedule shape: {noise_schedule.sqrt_alphas_cumprod.shape}")
    lr_schedule = optax.constant_schedule(1e-4)
    
    print("Loading data...")
    # Use the same sample data as before
    ds = create_protein_dataset(
        inputs="src/prxteinmpnn/training/data/pdb_sample.array_record",
        batch_size=2,
        use_preprocessed=True,
        preprocessed_index_path="src/prxteinmpnn/training/data/pdb_sample.index.json",
    )
    
    iterator = iter(ds)
    
    print("Starting training loop...")
    for i, batch in enumerate(iterator):
        if i >= 5: break
        
        print(f"Step {i+1}")
        
        # Prepare inputs
        coordinates = jnp.array(batch.coordinates)
        mask = jnp.array(batch.mask)
        residue_index = jnp.array(batch.residue_index)
        chain_index = jnp.array(batch.chain_index)
        sequence = jnp.array(batch.aatype)
        physics_features = jnp.array(batch.physics_features)
        
        step_key = jax.random.fold_in(train_key, i)
        
        model, opt_state, metrics = train_step(
            model,
            opt_state,
            optimizer,
            coordinates,
            mask,
            residue_index,
            chain_index,
            sequence,
            step_key,
            noise_schedule,
            lr_schedule,
            i,
            physics_features,
            physics_noise_scale=0.5,
        )
        
        print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")

    print("Diffusion training test passed!")

if __name__ == "__main__":
    test_diffusion_training()
