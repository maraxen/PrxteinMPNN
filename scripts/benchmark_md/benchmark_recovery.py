import os
import time
import itertools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
jax.config.update("jax_enable_x64", True)
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import argparse

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.io.weights import load_weights, NODE_FEATURES, EDGE_FEATURES, HIDDEN_FEATURES, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, K_NEIGHBORS, VOCAB_SIZE
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.scoring.score import make_score_sequence, score_sequence_with_encoding
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge
from jax_md import space

from prxteinmpnn.io.process import frame_iterator_from_inputs
from prxteinmpnn.utils.data_structures import ProteinTuple

# Dev Set PDB IDs
DEV_SET = ["1UBQ", "1CRN", "2GB1", "1L2Y"]
QUICK_DEV_SET = ["1UAO"]

# Full Factorial Design Parameters
NOISE_METHODS = ["none", "gaussian", "md"]
ENSEMBLE_STRATEGIES = ["single", "feature_avg"]
TEMPERATURES = [0.1, 0.5, 0.9]
SAMPLING_ALGOS = ["autoregressive"]
MODEL_WEIGHTS = ["original", "soluble"]
MODEL_VERSIONS = ["v_48_002", "v_48_020", "v_48_030"]
NUM_SAMPLES = 32
ENSEMBLE_SIZE = 4

# Noise Parameters
MD_TEMPS = [270, 300, 330, 360, 390, 420, 450]
GAUSSIAN_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

def protein_tuple_to_jax_md_input(protein_tuple, ff):
    """Convert ProteinTuple to JAX MD input arrays."""
    # Unpack ProteinTuple
    # We need to reconstruct atom_names and res_names for parameterize_system
    restypes = residue_constants.restypes + ["X"]
    seq_str = [restypes[i] for i in protein_tuple.aatype]
    res_names = [residue_constants.restype_1to3.get(r, "UNK") for r in seq_str]
    
    # 2. Flatten coordinates and generate atom names
    flat_coords = []
    flat_atom_names = []
    atom_counts = []
    
    L = protein_tuple.coordinates.shape[0]
    
    for i in range(L):
        res_name = res_names[i]
        if res_name == "UNK": 
            atom_counts.append(0)
            continue
        
        # Get standard atoms for this residue to ensure correct order/naming
        # We rely on residue_constants.residue_atoms
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        
        count = 0
        for atom_name in std_atoms:
            # Find index in atom37
            atom_idx = residue_constants.atom_order.get(atom_name)
            if atom_idx is not None and protein_tuple.atom_mask[i, atom_idx] > 0.5:
                flat_coords.append(protein_tuple.coordinates[i, atom_idx])
                flat_atom_names.append(atom_name)
                count += 1
        atom_counts.append(count)
                
    # Parameterize system to get params
    params = jax_md_bridge.parameterize_system(ff, res_names, flat_atom_names, atom_counts=atom_counts)
    
    return jnp.array(flat_coords), params, flat_atom_names, protein_tuple

def apply_gaussian_noise(coords, scale, key):
    """Apply Gaussian noise."""
    noise = jax.random.normal(key, coords.shape) * scale
    return coords + noise

def apply_md_sampling(coords, params, temperature, key):
    """Apply MD sampling."""
    # Run simulation: Minimization + NVT
    # We use a shorter run for the benchmark to be feasible
    r_final = simulate.run_simulation(
        params,
        coords,
        temperature=temperature * 300.0, 
        min_steps=100,
        therm_steps=500,
        implicit_solvent=True,
        solvent_dielectric=78.5,
        solute_dielectric=1.0,
        key=key
    )
    return r_final

def load_model_x64(model_weights, model_version):
    """Load model in x64 mode by casting skeleton to f32 then back to f64."""
    key = jax.random.PRNGKey(0)
    
    # Create skeleton (defaults to f64 because of config)
    skeleton = PrxteinMPNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        k_neighbors=K_NEIGHBORS,
        vocab_size=VOCAB_SIZE,
        key=key
    )
    
    # Cast to f32 for loading
    skeleton_f32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, skeleton)
    
    # Load weights
    loaded_f32 = load_weights(
        model_version=model_version,
        model_weights=model_weights,
        skeleton=skeleton_f32
    )
    
    # Cast back to f64
    loaded_f64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, loaded_f32)
    
    return loaded_f64

def run_benchmark(pdb_set=DEV_SET):
    """Run the sequence recovery benchmark."""
    print(f"Benchmarking Sequence Recovery on {pdb_set}")
    print(f"Model Weights: {MODEL_WEIGHTS}")
    print(f"Model Versions: {MODEL_VERSIONS}")
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    # Pre-load proteins
    proteins = {}
    
    # Download PDBs first
    pdb_dir = "data/pdb"
    os.makedirs(pdb_dir, exist_ok=True)
    pdb_paths = []
    for pdb_id in pdb_set:
        pdb_path = os.path.join(pdb_dir, f"{pdb_id.lower()}.pdb")
        if not os.path.exists(pdb_path):
            try:
                rcsb.fetch(pdb_id, "pdb", pdb_dir)
            except Exception as e:
                print(f"Failed to fetch {pdb_id}: {e}")
                continue
        if os.path.exists(pdb_path):
            pdb_paths.append(pdb_path)

    # Use unified data loader
    from prxteinmpnn.io import process
    
    for pdb_path in pdb_paths:
        pdb_id = os.path.basename(pdb_path).split(".")[0].upper()
        print(f"Processing {pdb_id}...")
        
        # Create iterator for single file to get first frame
        iterator = process.frame_iterator_from_inputs([pdb_path])
        
        try:
            protein_tuple = next(iterator)
        except StopIteration:
            print(f"No frames found in {pdb_id}")
            continue
        except Exception as e:
            print(f"Error loading {pdb_id}: {e}")
            continue
        
        try:
            jax_md_input = protein_tuple_to_jax_md_input(protein_tuple, ff)
            if jax_md_input is None:
                print(f"Skipping {pdb_id} due to processing failure")
                continue
                
            full_coords, params, atom_names, mpnn_feats = jax_md_input
            
            proteins[pdb_id] = {
                "full_coords": full_coords,
                "params": params,
                "mpnn_feats": mpnn_feats,
                "atom_names": atom_names
            }
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue

    results = []
    total_runs = 0
    
    # Iterate over models
    for model_weights in MODEL_WEIGHTS:
        for model_version in MODEL_VERSIONS:
            model_name = f"{model_weights}_{model_version}"
            print(f"Loading model: {model_name}")
            try:
                model = load_model_x64(model_weights, model_version)
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                continue
                
            # Create JIT-compiled functions
            sampler_fn = make_sample_sequences(model, random_decoding_order, sampling_strategy="temperature")
            sampler_fn = jax.jit(sampler_fn, static_argnames=["temperature"])
            
            encode_fn, split_sample_fn, _ = make_encoding_sampling_split_fn(model)
            encode_fn = jax.jit(encode_fn)
            split_sample_fn = jax.jit(split_sample_fn, static_argnames=["temperature"])
            
            score_fn = make_score_sequence(model)
            score_fn = jax.jit(score_fn)
            
            # Iterate over noise methods
            for noise_method in NOISE_METHODS:
                if noise_method == "none": noise_params = [None]
                elif noise_method == "gaussian": noise_params = GAUSSIAN_SCALES
                elif noise_method == "md": noise_params = MD_TEMPS
                
                for noise_param in noise_params:
                    for ensemble_strategy in ENSEMBLE_STRATEGIES:
                        if noise_method == "none" and ensemble_strategy == "feature_avg":
                            continue
                            
                        for temp in TEMPERATURES:
                            config = {
                                "model_weights": model_weights,
                                "model_version": model_version,
                                "noise_method": noise_method,
                                "noise_param": noise_param,
                                "ensemble_strategy": ensemble_strategy,
                                "temperature": temp,
                                "sampling_algo": "autoregressive"
                            }
                            
                            for pdb_id, p_data in proteins.items():
                                key = jax.random.PRNGKey(total_runs)
                                
                                # Extract features from ProteinTuple
                                protein_tuple = p_data["mpnn_feats"]
                                full_coords = p_data["full_coords"]
                                params = p_data["params"]
                                
                                native_seq = protein_tuple.aatype
                                res_idx = protein_tuple.residue_index
                                chain_idx = protein_tuple.chain_index
                                
                                # Construct backbone coordinates (L, 4, 3) for N, CA, C, O
                                N_idx = residue_constants.atom_order["N"]
                                CA_idx = residue_constants.atom_order["CA"]
                                C_idx = residue_constants.atom_order["C"]
                                O_idx = residue_constants.atom_order["O"]
                                
                                native_bb = protein_tuple.coordinates[:, [N_idx, CA_idx, C_idx, O_idx], :]
                                
                                # Mask: 1 if CA present
                                mask = protein_tuple.atom_mask[:, CA_idx]
                                
                                bb_indices = [N_idx, CA_idx, C_idx, O_idx]
                                
                                # Apply noise to full_coords and map back to ProteinTuple format
                                key, subkey_noise = jax.random.split(key)
                                if noise_method == "gaussian":
                                    noisy_full_coords = apply_gaussian_noise(full_coords, noise_param, subkey_noise)
                                elif noise_method == "md":
                                    noisy_full_coords = apply_md_sampling(full_coords, params, noise_param/300.0, subkey_noise)
                                else:
                                    noisy_full_coords = full_coords
                                
                                # Map noisy_full_coords (flat) back to (L, 37, 3) for ProteinTuple
                                L = protein_tuple.coordinates.shape[0]
                                new_coords_array = np.array(protein_tuple.coordinates) # Copy to modify
                                
                                curr_idx = 0
                                for i in range(L):
                                    res_name = residue_constants.restype_1to3.get(residue_constants.restypes[protein_tuple.aatype[i]], "UNK")
                                    if res_name == "UNK": continue
                                    
                                    std_atoms = residue_constants.residue_atoms.get(res_name, [])
                                    for atom_name in std_atoms:
                                        atom_idx = residue_constants.atom_order.get(atom_name)
                                        if atom_idx is not None and protein_tuple.atom_mask[i, atom_idx] > 0.5:
                                            new_coords_array[i, atom_idx] = noisy_full_coords[curr_idx]
                                            curr_idx += 1
                                            
                                noisy_protein_tuple = protein_tuple._replace(coordinates=new_coords_array)
                                
                                # Extract backbone coordinates from noisy_protein_tuple
                                noisy_bb = noisy_protein_tuple.coordinates[:, bb_indices, :]
                                
                                if ensemble_strategy == "feature_avg":
                                    encodings_list = []
                                    for e in range(ENSEMBLE_SIZE):
                                        key, subkey = jax.random.split(key)
                                        # For feature_avg, noise is applied per ensemble member
                                        if noise_method == "gaussian":
                                            c_noisy_full = apply_gaussian_noise(full_coords, noise_param, subkey)
                                        elif noise_method == "md":
                                            c_noisy_full = apply_md_sampling(full_coords, params, noise_param/300.0, subkey)
                                        else:
                                            c_noisy_full = full_coords
                                        
                                        # Map c_noisy_full back to ProteinTuple format to get backbone
                                        c_new_coords_array = np.array(protein_tuple.coordinates)
                                        c_curr_idx = 0
                                        for i in range(L):
                                            res_name = residue_constants.restype_1to3.get(residue_constants.restypes[protein_tuple.aatype[i]], "UNK")
                                            if res_name == "UNK": continue
                                            std_atoms = residue_constants.residue_atoms.get(res_name, [])
                                            for atom_name in std_atoms:
                                                atom_idx = residue_constants.atom_order.get(atom_name)
                                                if atom_idx is not None and protein_tuple.atom_mask[i, atom_idx] > 0.5:
                                                    c_new_coords_array[i, atom_idx] = c_noisy_full[c_curr_idx]
                                                    c_curr_idx += 1
                                        c_noisy_protein_tuple = protein_tuple._replace(coordinates=c_new_coords_array)
                                        c_bb = c_noisy_protein_tuple.coordinates[:, bb_indices, :]

                                        enc = encode_fn(subkey, c_bb, mask, res_idx, chain_idx)
                                        encodings_list.append(enc)
                                    
                                    avg_node = jnp.mean(jnp.stack([e[0] for e in encodings_list]), axis=0)
                                    avg_edge = jnp.mean(jnp.stack([e[1] for e in encodings_list]), axis=0)
                                    avg_enc = (avg_node, avg_edge, encodings_list[0][2], encodings_list[0][3], encodings_list[0][4])
                                    
                                    current_sampler = lambda k, t: split_sample_fn(k, avg_enc, random_decoding_order(k, len(native_seq))[0], temperature=t)
                                    
                                else:
                                    pass

                                for s in range(NUM_SAMPLES):
                                    key, subkey_sample = jax.random.split(key)
                                    
                                    if ensemble_strategy == "feature_avg":
                                        sampled_seq = current_sampler(subkey_sample, temp)
                                        ppl_noised = float("nan")
                                    else:
                                        sampled_seq, _, _ = sampler_fn(
                                            subkey_sample, noisy_bb, mask, res_idx, chain_idx, temperature=temp
                                        )
                                        
                                        ppl_noised_nll, _, _ = score_fn(
                                            subkey_sample, sampled_seq, noisy_bb, mask, res_idx, chain_idx
                                        )
                                        ppl_noised = float(jnp.exp(ppl_noised_nll))

                                    ppl_baseline_nll, _, _ = score_fn(
                                        subkey_sample, sampled_seq, native_bb, mask, res_idx, chain_idx
                                    )
                                    ppl_baseline = float(jnp.exp(ppl_baseline_nll))
                                    
                                    if ensemble_strategy == "feature_avg":
                                        ppl_avg_nll, _, _ = score_sequence_with_encoding(model, sampled_seq, avg_enc)
                                        ppl_avg = float(jnp.exp(ppl_avg_nll))
                                    else:
                                        ppl_avg = float("nan")

                                    recovery = float(jnp.mean(sampled_seq == native_seq))
                                    
                                    results.append({
                                        **config,
                                        "pdb": pdb_id,
                                        "sample_idx": s,
                                        "recovery": recovery,
                                        "perplexity_baseline": ppl_baseline,
                                        "perplexity_noised": ppl_noised,
                                        "perplexity_avg": ppl_avg
                                    })
                                    total_runs += 1
                                    
                                if total_runs % 100 == 0:
                                    print(f"Runs: {total_runs}...")
                                    pd.DataFrame(results).to_csv("benchmark_sequence_recovery_partial.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_sequence_recovery.csv", index=False)
    print("Benchmark complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequence recovery benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run on quick dev set (Chignolin).")
    args = parser.parse_args()
    
    target_set = QUICK_DEV_SET if args.quick else DEV_SET
    run_benchmark(target_set)
