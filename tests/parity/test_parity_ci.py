import os
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from prxteinmpnn.model.mpnn import PrxteinMPNN, PrxteinLigandMPNN

MODEL_PARAMS_DIR = "model_params"

@pytest.mark.parametrize("model_name", [
    "proteinmpnn_v_48_002_converted.eqx",
    "proteinmpnn_v_48_010_converted.eqx",
    "proteinmpnn_v_48_020_converted.eqx",
    "proteinmpnn_v_48_030_converted.eqx",
    "solublempnn_v_48_002_converted.eqx",
    "solublempnn_v_48_010_converted.eqx",
    "solublempnn_v_48_020_converted.eqx",
    "solublempnn_v_48_030_converted.eqx",
    "global_label_membrane_mpnn_v_48_020_converted.eqx",
    "per_residue_label_membrane_mpnn_v_48_020_converted.eqx",
    "ligandmpnn_v_32_005_25_converted.eqx",
    "ligandmpnn_v_32_010_25_converted.eqx",
    "ligandmpnn_v_32_020_25_converted.eqx",
    "ligandmpnn_v_32_030_25_converted.eqx",
])
def test_load_all_converted_weights(model_name):
    """Test that all converted .eqx files can be loaded into the respective architectures."""
    model_path = os.path.join(MODEL_PARAMS_DIR, model_name)
    assert os.path.exists(model_path), f"Weight file {model_path} missing"
    
    key = jax.random.PRNGKey(0)
    
    # Selection based on model name
    if "ligandmpnn" in model_name:
        k = 32 if "32" in model_name else 48
        model = PrxteinLigandMPNN(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k,
            key=key,
        )
    else:
        model = PrxteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            key=key,
        )
        
    # Attempt loading
    try:
        model = eqx.tree_serialise_leaves(model_path, model)
    except Exception as e:
        pytest.fail(f"Failed to load {model_name}: {str(e)}")
        
    print(f"Successfully loaded {model_name}")
