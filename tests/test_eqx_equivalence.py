"""Tests for numerical equivalence between the new eqx_new.PrxteinMPNN
model (as populated by the conversion script) and the original functional
implementation.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu as Gelu
import equinox as eqx
# --- Imports for the NEW Equinox model ---
# This is the model structure we are testing
from prxteinmpnn.model import PrxteinMPNN
from functools import partial
gelu = partial(Gelu, approximate=False)  # For easier use later
# --- Imports for the conversion script ---
# This script *creates* the new model instance we're testing
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
VOCAB_SIZE = 21
NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 512  # This is the hidden dimension in the dense FFN layers
K_NEIGHBORS = 48
NUM_AMINO_ACIDS = 21
MAX_RELATIVE_FEATURES = 32
from pathlib import Path
# --- Imports for the ORIGINAL functional model ---
# This is our "ground truth"
from prxteinmpnn.functional import (
    make_encoder,
    make_decoder,
    final_projection,
    extract_features,
    embed_sequence, # Need for conditional test
)
from prxteinmpnn.utils.types import ModelParameters
from prxteinmpnn.utils.autoregression import generate_ar_mask as get_autoregressive_mask # Need for AR test

# --- Import to load the weights ---
from prxteinmpnn.io.weights import load_weights

# Define the functional projection step (which was `protein_mpnn/~/W_e`)
# This matches the mapping in convert_from_old
def functional_project_features(param_dict: ModelParameters, edge_features: jax.Array) -> jax.Array:
    """Applies the final edge projection (W_e) as done in the functional model."""
    w_e_params = param_dict['protein_mpnn/~/W_e']
    w = w_e_params['w']
    b = w_e_params.get('b') # Use .get for safety
    
    output = jnp.dot(edge_features, w)
    if b is not None:
        output = output + b
    # The new model's features.w_e_proj is a simple eqx.nn.Linear,
    # which does *not* have a gelu. This mapping is correct.
    return output


class TestNewEqxEquivalence:
    """Compares the converted PrxteinMPNN (eqx_new) to the functional original."""

    # --- Class-level setup to avoid re-loading weights ---
    _old_weights_dict = None
    _param_dict = None
    _new_model = None
    _key = jax.random.PRNGKey(789)
    _test_inputs_generated = False
    
    # Input data
    _coords = None
    _mask = None
    _res_idx = None
    _chain_idx = None
    _ar_mask = None
    _one_hot_sequence = None

    # Ground truth functional outputs
    _func_edge_feat_raw = None
    _func_neighbors = None
    _func_edge_features = None # After projection

    # ***Path to the model you created with the conversion script***
    # ***You may need to change this path***
    CONVERTED_MODEL_PATH = Path("models/new_format/original_v_48_002.eqx")


    @classmethod
    def setup_class(cls):
        """Load functional weights and the converted .eqx model."""
        print("\n--- Setting up Equivalence Test Class ---")
        print("  Loading raw functional weights (original/v_48_002)...")
        try:
            # Use get_functional_model to load legacy functional weights
            from prxteinmpnn.functional.model import get_functional_model  # noqa: PLC0415
            cls._param_dict = get_functional_model(
                model_version="v_48_002",
                model_weights="original",
                use_new_architecture=False,  # Load legacy functional PyTree
            )
        except Exception as e:
            assert False, f"Functional weight loading failed: {e}"

        print(f"  Loading converted .eqx model from {cls.CONVERTED_MODEL_PATH}...")
        if not cls.CONVERTED_MODEL_PATH.exists():
            assert False, (
                f"Converted model file not found at {cls.CONVERTED_MODEL_PATH}. "
                f"Please run the conversion script first and ensure the path is correct."
            )

        # Create a skeleton of the new model to load the weights into
        skeleton_key = jax.random.PRNGKey(42)
        skeleton = PrxteinMPNN(
            node_features=NODE_FEATURES,
            edge_features=EDGE_FEATURES,
            hidden_features=HIDDEN_FEATURES,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            k_neighbors=K_NEIGHBORS,
            num_amino_acids=NUM_AMINO_ACIDS,
            vocab_size=VOCAB_SIZE,
            key=skeleton_key,
        )

        # Load the saved model leaves into the skeleton
        try:
            cls._new_model = eqx.tree_deserialise_leaves(cls.CONVERTED_MODEL_PATH, skeleton)
        except Exception as e:
            assert False, f"Failed to load .eqx model: {e}"
        
        print("  Generating test inputs...")
        key_inputs, key_seq, key_feat = jax.random.split(jax.random.PRNGKey(123), 3)
        num_atoms = 25
        
        cls._coords = jax.random.normal(key_inputs, (num_atoms, 14, 3)) # Full atom coords
        cls._mask = jnp.ones(num_atoms, dtype=jnp.int32)
        cls._res_idx = jnp.arange(num_atoms)
        cls._chain_idx = jnp.zeros(num_atoms, dtype=jnp.int32)
        
        # Inputs for conditional/AR tests
        cls._ar_mask = get_autoregressive_mask(cls._res_idx, cls._chain_idx)
        random_indices = jax.random.randint(key_seq, (num_atoms,), 0, VOCAB_SIZE - 1) # Sample 0-19
        cls._one_hot_sequence = jax.nn.one_hot(random_indices, VOCAB_SIZE)
        
        print("  Generating ground truth functional features...")
        # 2a. `extract_features` (corresponds to new `features` internal logic)
        # NOTE: prng_key is now first argument, then model_parameters
        cls._func_edge_feat_raw, cls._func_neighbors, _ = extract_features(
            key_feat,
            cls._param_dict,
            cls._coords,
            cls._mask,
            cls._res_idx,
            cls._chain_idx,
            backbone_noise=0.0
        )
        
        # 2b. `project_features` (corresponds to new `features.w_e_proj`)
        cls._func_edge_features = functional_project_features(
            cls._param_dict, cls._func_edge_feat_raw
        )
        
        cls._test_inputs_generated = True
        print("--- Setup Complete ---")

    def test_01_feature_extraction_equivalence(self):
        """Tests if the new `features` module matches the functional implementation."""
        assert self._test_inputs_generated, "Test inputs not generated"
        print("\nRunning Test 01: Feature Extraction Equivalence...")

        key_feat = jax.random.PRNGKey(123) # Must use the *same key* as setup
        
        # --- 1. New Equinox Path ---
        # The new model's feature block does it all in one go
        print("  Running new eqx.features module...")
        eqx_edge_feat, eqx_neighbors, _ = self._new_model.features(
            key_feat,
            self._coords,
            self._mask,
            self._res_idx,
            self._chain_idx,
            backbone_noise=0.0
        )

        # --- 2. Functional Path (already computed in setup) ---
        print("  Loading pre-computed functional features...")
        func_edge_feat = self._func_edge_features
        func_neighbors = self._func_neighbors

        # --- 3. Compare ---
        print("  Comparing feature extraction outputs...")
        
        # Compare neighbor indices
        assert jnp.all(eqx_neighbors == func_neighbors), "Neighbor indices do not match"
        
        # Compare final edge features
        assert jnp.allclose(
            eqx_edge_feat, 
            func_edge_feat, 
            rtol=1e-5, 
            atol=1e-5
        ), "Edge features do not match"
        print("  Feature Extraction Test PASSED.")

    def test_02_core_model_UNCONDITIONAL_equivalence(self):
        """Tests Encoder -> Decoder -> Proj pipeline equivalence (Unconditional)."""
        assert self._new_model is not None, "Model not loaded"
        print("\nRunning Test 02: Core Model (Unconditional) Equivalence...")

        # --- 1. Get Common Inputs ---
        edge_features = self._func_edge_features
        neighbor_indices = self._func_neighbors
        mask = self._mask

        # --- 2. New Equinox Path ---
        print("  Running new eqx model (unconditional)...")
        # Call the specific branch
        eqx_seq_out, eqx_logits = self._new_model._call_unconditional(
            edge_features, 
            neighbor_indices, 
            mask
        )

        # --- 3. Functional Path ---
        print("  Running func.encoder -> func.decoder (unconditional) -> func.projection...")
        # 3a. Encoder
        func_encoder = make_encoder(
            self._param_dict, 
            num_encoder_layers=NUM_ENCODER_LAYERS, 
            scale=30.0
        )
        func_nodes, func_edges = func_encoder(
            edge_features, 
            neighbor_indices, 
            mask
        )

        # 3b. Decoder (Unconditional)
        func_decoder = make_decoder(
            self._param_dict,
            attention_mask_type=None, # Unconditional
            num_decoder_layers=NUM_DECODER_LAYERS,
            scale=30.0,
        )
        func_nodes_decoded = func_decoder(func_nodes, func_edges, mask)

        # 3c. Projection
        func_logits = final_projection(self._param_dict, func_nodes_decoded)

        # --- 4. Compare ---
        print("  Comparing core model (unconditional) outputs...")
        assert jnp.allclose(
            eqx_logits, 
            func_logits, 
            rtol=1e-5, 
            atol=1e-5
        ), "Final logits do not match"
        print("  Core Model (Unconditional) Test PASSED.")

    def test_03_core_model_CONDITIONAL_equivalence(self):
        """Tests Encoder -> Decoder -> Proj pipeline equivalence (Conditional)."""
        assert self._new_model is not None, "Model not loaded"
        print("\nRunning Test 03: Core Model (Conditional) Equivalence...")

        # --- 1. Get Common Inputs ---
        edge_features = self._func_edge_features
        neighbor_indices = self._func_neighbors
        mask = self._mask
        ar_mask = self._ar_mask
        one_hot_sequence = self._one_hot_sequence

        # --- 2. New Equinox Path ---
        print("  Running new eqx model (conditional)...")
        eqx_seq_out, eqx_logits = self._new_model._call_conditional(
            edge_features,
            neighbor_indices,
            mask,
            ar_mask=ar_mask,
            one_hot_sequence=one_hot_sequence
        )

        # --- 3. Functional Path ---
        print("  Running func.encoder -> func.decoder (conditional) -> func.projection...")
        # 3a. Encoder
        func_encoder = make_encoder(
            self._param_dict, 
            num_encoder_layers=NUM_ENCODER_LAYERS, 
            scale=30.0
        )
        func_nodes, func_edges = func_encoder(
            edge_features, 
            neighbor_indices, 
            mask
        )
        
        # 3b. Decoder (Conditional)
        func_decoder = make_decoder(
            self._param_dict,
            attention_mask_type="conditional", # Key change
            decoding_approach="conditional",   # Add decoding approach
            num_decoder_layers=NUM_DECODER_LAYERS,
            scale=30.0,
        )
        func_nodes_decoded = func_decoder(
            func_nodes, 
            func_edges, 
            neighbor_indices, # Required for conditional
            mask,
            ar_mask,          # Required for conditional
            one_hot_sequence  # Required for conditional
        )

        # 3d. Projection
        func_logits = final_projection(self._param_dict, func_nodes_decoded)

        # --- 4. Compare ---
        print("  Comparing core model (conditional) outputs...")
        max_diff = jnp.max(jnp.abs(eqx_logits - func_logits))
        rel_diff = jnp.max(jnp.abs((eqx_logits - func_logits) / (jnp.abs(func_logits) + 1e-8)))
        print(f"    Max absolute difference: {max_diff:.6e}")
        print(f"    Max relative difference: {rel_diff:.6e}")
        assert jnp.allclose(
            eqx_logits, 
            func_logits, 
            rtol=1e-5, 
            atol=1e-5,
        ), f"Final logits do not match (max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e})"
        print("  Core Model (Conditional) Test PASSED.")

    def test_04_core_model_AUTOREGRESSIVE_equivalence(self):
        """Tests conditional decoder with zero sequence (first AR step)."""
        assert self._new_model is not None, "Model not loaded"
        print("\nRunning Test 04: Core Model (Autoregressive - First Step) Equivalence...")

        # --- 1. Get Common Inputs ---
        edge_features = self._func_edge_features
        neighbor_indices = self._func_neighbors
        mask = self._mask
        ar_mask = self._ar_mask

        # --- 2. Encoder (run once, shared) ---
        print("  Running encoder...")
        func_encoder = make_encoder(
            self._param_dict,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            scale=30.0
        )
        func_nodes, func_edges = func_encoder(
            edge_features,
            neighbor_indices,
            mask
        )

        eqx_nodes, eqx_edges = self._new_model.encoder(
            edge_features,
            neighbor_indices,
            mask
        )

        # --- 3. Conditional decoder with zero sequence (first AR step) ---
        print("  Testing conditional decoder with zero sequence...")
        num_positions = mask.shape[0]
        vocab_size = VOCAB_SIZE
        zero_sequence = jnp.zeros((num_positions, vocab_size), dtype=jnp.float32)

        # 3a. Functional path
        func_decoder = make_decoder(
            self._param_dict,
            attention_mask_type="conditional",
            decoding_approach="conditional",
            num_decoder_layers=NUM_DECODER_LAYERS,
            scale=30.0,
        )
        func_decoded = func_decoder(
            func_nodes,
            func_edges,
            neighbor_indices,
            mask,
            ar_mask,
            zero_sequence
        )
        func_logits = final_projection(self._param_dict, func_decoded)

        # 3b. Eqx path
        eqx_decoded = self._new_model.decoder.call_conditional(
            eqx_nodes,
            eqx_edges,
            neighbor_indices,
            mask,
            ar_mask,
            zero_sequence,
            self._new_model.w_s_embed.weight,
        )
        eqx_logits = jax.vmap(self._new_model.w_out)(eqx_decoded)

        # --- 4. Compare ---
        print("  Comparing logits from first AR step...")
        max_diff = jnp.abs(eqx_logits - func_logits).max()
        rel_diff = max_diff / (jnp.abs(func_logits).max() + 1e-9)
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Max relative difference: {rel_diff:.6e}")

        assert jnp.allclose(
            eqx_logits,
            func_logits,
            rtol=1e-5,
            atol=1e-5
        ), f"Logits do not match: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}"

        print("  Core Model (Autoregressive) Test PASSED.")