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

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.io.weights import load_weights, NODE_FEATURES, EDGE_FEATURES, HIDDEN_FEATURES, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, K_NEIGHBORS, VOCAB_SIZE
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.scoring.score import make_score_sequence, score_sequence_with_encoding
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge
from jax_md import space

# Dev Set PDB IDs
DEV_SET = ["1UBQ", "1CRN", "2GB1", "1L2Y"]

# Full Factorial Design Parameters
NOISE_METHODS = ["none", "gaussian", "md"]
ENSEMBLE_STRATEGIES = ["single", "feature_avg"]
TEMPERATURES = [0.1, 0.5, 0.9]
SAMPLING_ALGOS = ["autoregressive"]
MODELS = ["original", "v_48_002", "v_48_020", "v_48_030", "soluble", "s_48_002", "s_48_020", "s_48_030"]
NUM_SAMPLES = 32
ENSEMBLE_SIZE = 4

# Noise Parameters
MD_TEMPS = [270, 300, 330, 360, 390, 420, 450]
GAUSSIAN_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    """Download and load PDB using Biotite."""
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception as e:
            print(f"Failed to fetch {pdb_id}: {e}")
            return None
    
    try:
        file = pdb.PDBFile.read(pdb_path)
        structure = file.get_structure(model=1)
        return structure
    except Exception as e:
        print(f"Failed to parse {pdb_id}: {e}")
        return None

def extract_system_from_biotite(atom_array):
    """Extract coordinates and topology from Biotite AtomArray."""
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    chains = struc.get_chains(atom_array)
    if len(chains) > 0:
        atom_array = atom_array[atom_array.chain_id == chains[0]]
        
    res_names = []
    atom_names = []
    coords_list = []
    backbone_indices_list = []
    current_atom_idx = 0
    
    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1:
            continue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4:
            continue
            
        res_names.append(res_name)
        atom_names.extend(std_atoms)
        
        res_coords = np.full((len(std_atoms), 3), np.nan)
        res_bb_indices = np.full(4, -1, dtype=int)
        
        for i, atom_name in enumerate(std_atoms):
            mask = res.atom_name == atom_name
            if np.any(mask):
                res_coords[i] = res[mask][0].coord
            else:
                ca_mask = res.atom_name == "CA"
                if np.any(ca_mask):
                    res_coords[i] = res[ca_mask][0].coord
                else:
                    res_coords[i] = np.array([0., 0., 0.])
            
            # Check if this atom is backbone
            if atom_name == "N": res_bb_indices[0] = current_atom_idx + i
            elif atom_name == "CA": res_bb_indices[1] = current_atom_idx + i
            elif atom_name == "C": res_bb_indices[2] = current_atom_idx + i
            elif atom_name == "O": res_bb_indices[3] = current_atom_idx + i
            
        coords_list.append(res_coords)
        backbone_indices_list.append(res_bb_indices)
        current_atom_idx += len(std_atoms)
        
    if not coords_list:
        return None, None, None, None
        
    coords = np.vstack(coords_list)
    backbone_indices = np.array(backbone_indices_list) # (L, 4)
    
    # Extract backbone coords
    # Handle missing atoms (indices -1) - though we filled coords with placeholders, 
    # we need valid indices into 'coords' array.
    # If we filled coords, the index exists.
    # But wait, std_atoms includes all atoms. If an atom was missing in PDB but we filled it in res_coords,
    # it corresponds to index 'i' in res_coords.
    # So 'current_atom_idx + i' is valid.
    
    backbone_coords = coords[backbone_indices] # (L, 4, 3)
    
    aa_map = residue_constants.restype_order_with_x
    seq = jnp.array([aa_map.get(r, 20) for r in res_names])
    
    N = len(backbone_coords)
    mask = jnp.ones((N,), dtype=jnp.float32)
    residue_index = jnp.arange(N)
    chain_index = jnp.zeros((N,), dtype=jnp.int32)
    
    return coords, res_names, atom_names, (backbone_coords, seq, mask, residue_index, chain_index, backbone_indices)

def apply_gaussian_noise(coords, scale, key):
    """Apply Gaussian noise."""
    noise = jax.random.normal(key, coords.shape) * scale
    return coords + noise

def apply_md_sampling(coords, params, temperature, key):
    """Apply MD sampling."""
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

def get_model_kwargs(model_name):
    """Parse model name into load_model kwargs."""
    if model_name == "original":
        return {"model_weights": "original", "model_version": "v_48_020"}
    if model_name == "soluble":
        return {"model_weights": "soluble", "model_version": "v_48_020"}
    
    if model_name.startswith("v_"):
        return {"model_weights": "original", "model_version": model_name}
    if model_name.startswith("s_"):
        version = model_name.replace("s_", "v_")
        return {"model_weights": "soluble", "model_version": version}
        
    return {"model_weights": "original", "model_version": "v_48_020"}

def load_model_x64(model_name):
    """Load model in x64 mode by casting skeleton to f32 then back to f64."""
    kwargs = get_model_kwargs(model_name)
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
        model_version=kwargs["model_version"],
        model_weights=kwargs["model_weights"],
        skeleton=skeleton_f32
    )
    
    # Cast back to f64
    loaded_f64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, loaded_f32)
    
    return loaded_f64

def run_benchmark():
    """Run the sequence recovery benchmark."""
    print(f"Benchmarking Sequence Recovery on {DEV_SET}")
    print(f"Models: {MODELS}")
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    # Pre-load proteins
    proteins = {}
    for pdb_id in DEV_SET:
        print(f"Loading {pdb_id}...")
        atom_array = download_and_load_pdb(pdb_id)
        if atom_array is not None:
            full_coords, res_names, atom_names, mpnn_feats = extract_system_from_biotite(atom_array)
            if full_coords is not None:
                params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
                proteins[pdb_id] = {
                    "full_coords": jnp.array(full_coords),
                    "params": params,
                    "mpnn_feats": mpnn_feats, # (bb, seq, mask, res, chain, bb_indices)
                    "atom_names": atom_names
                }

    results = []
    total_runs = 0
    
    # Iterate over models
    for model_name in MODELS:
        print(f"Loading model: {model_name}")
        try:
            model = load_model_x64(model_name)
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
                            "model": model_name,
                            "noise_method": noise_method,
                            "noise_param": noise_param,
                            "ensemble_strategy": ensemble_strategy,
                            "temperature": temp,
                            "sampling_algo": "autoregressive"
                        }
                        
                        for pdb_id, p_data in proteins.items():
                            key = jax.random.PRNGKey(total_runs)
                            
                            native_bb, native_seq, mask, res_idx, chain_idx, bb_indices = p_data["mpnn_feats"]
                            full_coords = p_data["full_coords"]
                            params = p_data["params"]
                            
                            if ensemble_strategy == "feature_avg":
                                encodings_list = []
                                for e in range(ENSEMBLE_SIZE):
                                    key, subkey = jax.random.split(key)
                                    if noise_method == "gaussian":
                                        c = apply_gaussian_noise(full_coords, noise_param, subkey)
                                    elif noise_method == "md":
                                        c = apply_md_sampling(full_coords, params, noise_param/300.0, subkey)
                                    else:
                                        c = full_coords
                                    
                                    c_bb = c[bb_indices]
                                    enc = encode_fn(subkey, c_bb, mask, res_idx, chain_idx)
                                    encodings_list.append(enc)
                                
                                avg_node = jnp.mean(jnp.stack([e[0] for e in encodings_list]), axis=0)
                                avg_edge = jnp.mean(jnp.stack([e[1] for e in encodings_list]), axis=0)
                                avg_enc = (avg_node, avg_edge, encodings_list[0][2], encodings_list[0][3], encodings_list[0][4])
                                
                                current_sampler = lambda k, t: split_sample_fn(k, avg_enc, random_decoding_order(k, len(native_seq))[0], temperature=t)
                                
                            else:
                                pass

                            for s in range(NUM_SAMPLES):
                                key, subkey_noise, subkey_sample = jax.random.split(key, 3)
                                
                                if ensemble_strategy == "feature_avg":
                                    sampled_seq = current_sampler(subkey_sample, temp)
                                    ppl_noised = float("nan")
                                else:
                                    if noise_method == "gaussian":
                                        noisy_full = apply_gaussian_noise(full_coords, noise_param, subkey_noise)
                                    elif noise_method == "md":
                                        noisy_full = apply_md_sampling(full_coords, params, noise_param/300.0, subkey_noise)
                                    else:
                                        noisy_full = full_coords
                                    
                                    noisy_bb = noisy_full[bb_indices]
                                    
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
    run_benchmark()
