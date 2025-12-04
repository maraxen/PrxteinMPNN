
import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
import optax
from prxteinmpnn.training.trainer import train_step, create_optimizer
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.training.diffusion import NoiseSchedule

def test_diffusion_train_step():
    # Setup
    key = jax.random.PRNGKey(0)
    spec = TrainingSpecification(
        inputs="dummy",
        training_mode="diffusion",
        batch_size=2,
        learning_rate=1e-4
    )
    
    # Initialize model
    model = load_model(model_version="v_48_020", training_mode="diffusion", key=key)
    optimizer, lr_schedule = create_optimizer(spec)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    noise_schedule = NoiseSchedule(num_steps=100)
    
    # Dummy batch
    batch_size = 2
    seq_len = 10
    
    coordinates = jnp.zeros((batch_size, seq_len, 4, 3))
    mask = jnp.ones((batch_size, seq_len))
    residue_index = jnp.tile(jnp.arange(seq_len), (batch_size, 1))
    chain_index = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    sequence = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Run step
    new_model, new_opt_state, metrics = train_step(
        model,
        opt_state,
        optimizer,
        coordinates,
        mask,
        residue_index,
        chain_index,
        sequence,
        key,
        label_smoothing=0.0,
        current_step=0,
        lr_schedule=lr_schedule,
        physics_features=None, # Should work without physics features
        training_mode="diffusion",
        noise_schedule=noise_schedule
    )
    
    assert metrics.loss is not None
    assert not jnp.isnan(metrics.loss)

def test_dimension_mismatch_fix():
    # Verify that passing physics_features=None to a standard MPNN works
    # even if the data loader theoretically could have provided them (simulated here by just passing None)
    
    key = jax.random.PRNGKey(0)
    spec = TrainingSpecification(
        inputs="dummy",
        training_mode="autoregressive",
        batch_size=2
    )
    
    model = load_model(model_version="v_48_020", training_mode="autoregressive", key=key)
    optimizer, lr_schedule = create_optimizer(spec)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    batch_size = 2
    seq_len = 10
    
    coordinates = jnp.zeros((batch_size, seq_len, 4, 3))
    mask = jnp.ones((batch_size, seq_len))
    residue_index = jnp.tile(jnp.arange(seq_len), (batch_size, 1))
    chain_index = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    sequence = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Run step with physics_features=None explicitly
    new_model, new_opt_state, metrics = train_step(
        model,
        opt_state,
        optimizer,
        coordinates,
        mask,
        residue_index,
        chain_index,
        sequence,
        key,
        label_smoothing=0.0,
        current_step=0,
        lr_schedule=lr_schedule,
        physics_features=None, 
        training_mode="autoregressive"
    )
    
    assert metrics.loss is not None
