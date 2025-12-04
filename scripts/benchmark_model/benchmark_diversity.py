"""Benchmark Sequence Diversity vs Recovery."""
import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import itertools
import argparse

# PrxteinMPNN imports
from prolix.physics import simulate, force_fields, jax_md_bridge, system
from priox.chem import residues as residue_constants
from priox.core.containers import Protein, ProteinTuple
from priox.io.weights import load_model
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.run.sampling import _sample_batch
from prxteinmpnn.utils.decoding_order import random_decoding_order

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Constants
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
QUICK_DEV_SET = ["1UAO"]
NUM_SAMPLES = 32 # Total sequences per method/param
MD_STEPS = 100
MD_THERM = 500

# --- Shared Helpers (Duplicated from benchmark_scrmsd.py) ---
def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception as e:
            print(f"Failed to fetch {pdb_id}: {e}")
            return None
    pdb_file = pdb.PDBFile.read(pdb_path)
    return pdb.get_structure(pdb_file, model=1)

def extract_system_from_biotite(atom_array):
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    chains = struc.get_chains(atom_array)
    if len(chains) > 0:
        atom_array = atom_array[atom_array.chain_id == chains[0]]
        
    res_names = []
    atom_names = []
    coords_list = []
    residue_indices = []
    aatypes = []
    
    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4: continue
            
        res_names.append(res_name)
        atom_names.extend(std_atoms)
        residue_indices.append(res[0].res_id)
        aatypes.append(residue_constants.restype_order.get(residue_constants.restype_3to1.get(res_name, 'X'), 20))
        
        res_coords = np.full((len(std_atoms), 3), np.nan)
        for i, atom_name in enumerate(std_atoms):
            mask = res.atom_name == atom_name
            if np.any(mask): res_coords[i] = res[mask][0].coord
            elif np.any(res.atom_name == "CA"): res_coords[i] = res[res.atom_name == "CA"][0].coord
            else: res_coords[i] = np.array([0., 0., 0.])
        coords_list.append(res_coords)
        
    if not coords_list: return None, None, None, None, None, None
    coords = np.vstack(coords_list)
    
    ca_coords_list = []
    for res_c, r_name in zip(coords_list, res_names):
        std_atoms = residue_constants.residue_atoms.get(r_name, [])
        res_bb = np.zeros((4, 3))
        for k, atom in enumerate(["N", "CA", "C", "O"]):
            if atom in std_atoms: res_bb[k] = res_c[std_atoms.index(atom)]
        ca_coords_list.append(res_bb)
    bb_coords = np.array(ca_coords_list)
    
    return coords, res_names, atom_names, bb_coords, np.array(residue_indices), np.array(aatypes)

def extract_bb_from_full(full_c, r_names):
    curr_idx = 0
    new_bb_list = []
    for r_name in r_names:
        std_atoms = residue_constants.residue_atoms.get(r_name, [])
        n_atoms = len(std_atoms)
        res_atoms = full_c[curr_idx : curr_idx + n_atoms]
        curr_idx += n_atoms
        res_bb = np.zeros((4, 3))
        for k, atom in enumerate(["N", "CA", "C", "O"]):
            if atom in std_atoms: res_bb[k] = res_atoms[std_atoms.index(atom)]
        new_bb_list.append(res_bb)
    return np.array(new_bb_list)

def coords_to_protein(bb_coords, aatypes, residue_index):
    num_res = bb_coords.shape[0]
    coords_37 = np.zeros((num_res, 37, 3))
    coords_37[:, 0, :] = bb_coords[:, 0, :] # N
    coords_37[:, 1, :] = bb_coords[:, 1, :] # CA
    coords_37[:, 2, :] = bb_coords[:, 2, :] # C
    coords_37[:, 4, :] = bb_coords[:, 3, :] # O
    pt = ProteinTuple(
        coordinates=coords_37,
        aatype=aatypes,
        atom_mask=np.ones((num_res, 37)),
        residue_index=residue_index,
        chain_index=np.zeros(num_res, dtype=int),
        dihedrals=None
    )
    p = Protein.from_tuple(pt)
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0) if x is not None else None, p)

def apply_gaussian_noise(coords, scale, key):
    return coords + jax.random.normal(key, coords.shape) * scale

def apply_md_sampling(coords, params, temperature, key):
    return simulate.run_simulation(
        params, coords, temperature=temperature * 300.0, min_steps=MD_STEPS, therm_steps=MD_THERM,
        implicit_solvent=True, solvent_dielectric=78.5, solute_dielectric=1.0, key=key
    )

# --- Diversity Metrics ---
def compute_metrics(sequences, native_seq, logits_list=None):
    """Compute diversity and recovery metrics."""
    n = len(sequences)
    if n == 0: return {}
    
    # Recovery
    recoveries = []
    for seq in sequences:
        matches = sum(1 for s, n in zip(seq, native_seq) if s == n)
        recoveries.append(matches / len(native_seq))
    avg_recovery = np.mean(recoveries)
    
    # Diversity (Pairwise Identity)
    pairwise_identities = []
    if n > 1:
        for s1, s2 in itertools.combinations(sequences, 2):
            matches = sum(1 for a, b in zip(s1, s2) if a == b)
            pairwise_identities.append(matches / len(s1))
        avg_pairwise_id = np.mean(pairwise_identities)
    else:
        avg_pairwise_id = 1.0
        
    # Unique Sequences
    unique_seqs = len(set(sequences))
    pct_unique = unique_seqs / n
    
    # Perplexity (if logits provided)
    avg_perplexity = np.nan
    if logits_list is not None:
        # logits: list of (seq_len, 21)
        # We need to compute perplexity of the *sampled* sequence
        # PPL = exp( -1/N * sum( log p(x_i) ) )
        ppls = []
        for seq, logits in zip(sequences, logits_list):
            # Convert seq string to indices
            seq_idx = [residue_constants.restype_order.get(aa, 20) for aa in seq]
            seq_idx = np.array(seq_idx)
            
            # Log Softmax
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            
            # Gather log probs of sampled tokens
            token_log_probs = np.array([log_probs[i, idx] for i, idx in enumerate(seq_idx) if idx < 20])
            
            if len(token_log_probs) > 0:
                ppl = np.exp(-np.mean(token_log_probs))
                ppls.append(ppl)
        if ppls:
            avg_perplexity = np.mean(ppls)
            
    return {
        "recovery": avg_recovery,
        "pairwise_identity": avg_pairwise_id,
        "unique_percent": pct_unique,
        "perplexity": avg_perplexity
    }

def run_benchmark(pdb_set=DEV_SET):
    print(f"Benchmarking Diversity vs Recovery on Dev Set: {pdb_set}")
    ff = force_fields.load_force_field_from_hub("ff14SB")
    model = load_model(model_version="v_48_020")
    
    # Sampler for Fixed Backbone (Temperature Scaling)
    sampler_temp = make_sample_sequences(model, random_decoding_order, "temperature")
    
    key = jax.random.PRNGKey(0)
    results = []
    
    for pdb_id in pdb_set:
        print(f"\nProcessing {pdb_id}...")
        atom_array = download_and_load_pdb(pdb_id)
        if atom_array is None: continue
        
        full_coords_np, res_names, atom_names, bb_coords_orig, res_idx, aatypes = extract_system_from_biotite(atom_array)
        if full_coords_np is None: continue
        
        native_seq = "".join([residue_constants.restypes[i] for i in aatypes if i < 20])
        params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
        coords = jnp.array(full_coords_np)
        
        # 1. Temperature Scaling (Fixed Backbone)
        print("  Method: Temperature Scaling...")
        prot_fixed = coords_to_protein(bb_coords_orig, aatypes, res_idx)
        
        # We want to sample at different temperatures
        TEMPS = [0.1, 0.5, 1.0, 1.5]
        for t in TEMPS:
            # Generate NUM_SAMPLES sequences
            spec = SamplingSpecification(inputs=["dummy"], num_samples=NUM_SAMPLES, temperature=t, batch_size=1)
            # Note: _sample_batch handles num_samples internally if we pass it correctly
            # But here we passed batch_size=1 protein.
            # _sample_batch returns (batch, num_samples, noise, temp, len)
            
            seqs_batch, logits_batch = _sample_batch(spec, prot_fixed, sampler_temp)
            
            # Extract sequences
            sequences = []
            logits_list = []
            for i in range(NUM_SAMPLES):
                seq_ints = seqs_batch[0, i, 0, 0] # batch 0, sample i, noise 0, temp 0 (since we passed scalar temp)
                seq_str = "".join([residue_constants.restypes[x] for x in seq_ints if x < 20])
                sequences.append(seq_str)
                logits_list.append(logits_batch[0, i, 0, 0])
                
            metrics = compute_metrics(sequences, native_seq, logits_list)
            results.append({"pdb": pdb_id, "method": "temperature", "param": t, **metrics})

        # 2. Gaussian Noise (Fixed Low Temp Sampling)
        print("  Method: Gaussian Noise...")
        SCALES = [0.1, 0.2, 0.3, 0.5]
        spec_low = SamplingSpecification(inputs=["dummy"], num_samples=1, temperature=0.1, batch_size=1)
        
        for scale in SCALES:
            sequences = []
            logits_list = []
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                noisy_coords = apply_gaussian_noise(coords, scale, subkey)
                bb_noisy = extract_bb_from_full(np.array(noisy_coords), res_names)
                prot = coords_to_protein(bb_noisy, aatypes, res_idx)
                
                seqs_batch, logits_batch = _sample_batch(spec_low, prot, sampler_temp)
                seq_ints = seqs_batch[0, 0, 0, 0]
                seq_str = "".join([residue_constants.restypes[x] for x in seq_ints if x < 20])
                sequences.append(seq_str)
                logits_list.append(logits_batch[0, 0, 0, 0])
                
            metrics = compute_metrics(sequences, native_seq, logits_list)
            results.append({"pdb": pdb_id, "method": "gaussian", "param": scale, **metrics})
            
        # 3. MD Sampling (Fixed Low Temp Sampling)
        print("  Method: MD Sampling...")
        MD_TEMPS = [300, 400, 500]
        for temp in MD_TEMPS:
            sequences = []
            logits_list = []
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                try:
                    md_coords = apply_md_sampling(coords, params, temp/300.0, subkey)
                    bb_md = extract_bb_from_full(np.array(md_coords), res_names)
                    prot = coords_to_protein(bb_md, aatypes, res_idx)
                    
                    seqs_batch, logits_batch = _sample_batch(spec_low, prot, sampler_temp)
                    seq_ints = seqs_batch[0, 0, 0, 0]
                    seq_str = "".join([residue_constants.restypes[x] for x in seq_ints if x < 20])
                    sequences.append(seq_str)
                    logits_list.append(logits_batch[0, 0, 0, 0])
                except Exception:
                    pass
            
            if sequences:
                metrics = compute_metrics(sequences, native_seq, logits_list)
                results.append({"pdb": pdb_id, "method": "md", "param": temp, **metrics})

    df = pd.DataFrame(results)
    df.to_csv("benchmark_diversity.csv", index=False)
    print("Saved results to benchmark_diversity.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diversity benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run on quick dev set (Chignolin).")
    args = parser.parse_args()
    
    target_set = QUICK_DEV_SET if args.quick else DEV_SET
    run_benchmark(target_set)
