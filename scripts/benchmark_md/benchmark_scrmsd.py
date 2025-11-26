"""Benchmark Designability (scRMSD) of generated sequences."""
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
from biotite.structure.io import save_structure

# PrxteinMPNN imports
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.run.sampling import _sample_batch
from prxteinmpnn.utils.decoding_order import random_decoding_order

# Note: We do NOT enable x64 globally here yet, because we need to load the model as f32 first.

# Constants
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
NUM_SAMPLES = 4  # Reduced for speed in this script, user can increase
MD_STEPS = 100
MD_THERM = 500

def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    """Download and load PDB using Biotite."""
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception as e:
            print(f"Failed to fetch {pdb_id}: {e}")
            return None
            
    pdb_file = pdb.PDBFile.read(pdb_path)
    atom_array = pdb.get_structure(pdb_file, model=1)
    return atom_array

def extract_system_from_biotite(atom_array):
    """Extract coordinates and topology from Biotite AtomArray."""
    # Filter for protein
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    
    # Get first chain
    chains = struc.get_chains(atom_array)
    if len(chains) > 0:
        atom_array = atom_array[atom_array.chain_id == chains[0]]
        
    res_names = []
    atom_names = []
    coords_list = []
    residue_indices = []
    chain_indices = []
    aatypes = []
    
    # Iterate over residues
    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1:
            continue
            
        # Standard atoms for this residue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        
        # Check backbone
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4:
            continue
            
        res_names.append(res_name)
        atom_names.extend(std_atoms)
        residue_indices.append(res[0].res_id)
        chain_indices.append(0) # Single chain
        aatypes.append(residue_constants.restype_order.get(residue_constants.restype_3to1.get(res_name, 'X'), 20))
        
        # Extract coords for all standard atoms
        res_coords = np.full((len(std_atoms), 3), np.nan)
        
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
                    
        coords_list.append(res_coords)
        
    if not coords_list:
        return None, None, None, None, None, None
        
    coords = np.vstack(coords_list)
    
    # Also need CA coords for Protein object (N, CA, C, O)
    # PrxteinMPNN expects (num_res, 4, 3) for N, CA, C, O usually, or (num_res, 37, 3)
    # Let's construct (num_res, 4, 3) for the Protein object
    ca_coords_list = []
    for res_c, r_name in zip(coords_list, res_names):
        std_atoms = residue_constants.residue_atoms.get(r_name, [])
        # Find indices of N, CA, C, O in std_atoms
        bb_indices = [std_atoms.index(a) for a in ["N", "CA", "C", "O"] if a in std_atoms]
        # This might fail if some are missing, but we checked backbone_mask < 4 above
        # However, std_atoms definition ensures they are present in the list
        
        # Map to standard 4 atoms
        res_bb = np.zeros((4, 3))
        for k, atom in enumerate(["N", "CA", "C", "O"]):
            if atom in std_atoms:
                idx = std_atoms.index(atom)
                res_bb[k] = res_c[idx]
            else:
                # Should not happen given check above
                pass
        ca_coords_list.append(res_bb)
        
    bb_coords = np.array(ca_coords_list)
    
    return coords, res_names, atom_names, bb_coords, np.array(residue_indices), np.array(aatypes)

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
        min_steps=MD_STEPS,
        therm_steps=MD_THERM,
        key=key
    )
    return r_final

def coords_to_protein(bb_coords, aatypes, residue_index):
    """Convert backbone coordinates to Protein object."""
    num_res = bb_coords.shape[0]
    
    # Pad to 37 atoms (standard for Protein object usually, or just 4 is fine if model supports it)
    # The model expects (N, 4, 3) or (N, 37, 3). Let's check data_structures.
    # It says "The atom types correspond to residue_constants.atom_types, i.e. the first three are N, CA, CB."
    # Wait, N, CA, CB, C, O?
    # residue_constants.atom_types usually has 37.
    # Let's pad to 37 to be safe.
    
    coords_37 = np.zeros((num_res, 37, 3))
    # N, CA, C, O are indices 0, 1, 2, 4 in standard AlphaFold/OpenFold ordering?
    # Let's check residue_constants.atom_order
    # N=0, CA=1, C=2, CB=3, O=4
    
    # bb_coords is N, CA, C, O
    coords_37[:, 0, :] = bb_coords[:, 0, :] # N
    coords_37[:, 1, :] = bb_coords[:, 1, :] # CA
    coords_37[:, 2, :] = bb_coords[:, 2, :] # C
    coords_37[:, 4, :] = bb_coords[:, 3, :] # O
    
    # Construct ProteinTuple
    pt = ProteinTuple(
        coordinates=coords_37,
        aatype=aatypes,
        atom_mask=np.ones((num_res, 37)), # Assume all present
        residue_index=residue_index,
        chain_index=np.zeros(num_res, dtype=int),
        dihedrals=None
    )
    
    # Convert to Protein (batch size 1)
    p = Protein.from_tuple(pt)
    
    # Add batch dimension
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0) if x is not None else None, p)

def run_folding(sequence, output_pdb):
    """Placeholder for running external folder (e.g. ESMFold)."""
    # In a real scenario, we would call:
    # subprocess.run(["esmfold", "-i", sequence, "-o", output_pdb])
    # For now, we just save the sequence to a FASTA
    with open(output_pdb.replace(".pdb", ".fasta"), "w") as f:
        f.write(f">seq\n{sequence}\n")
    return None # Return path to PDB if generated

def calculate_rmsd(native_coords, pred_coords):
    """Calculate RMSD between two sets of coordinates (CA only)."""
    # Placeholder
    return 0.0

def run_benchmark():
    print(f"Benchmarking Designability (scRMSD) on Dev Set: {DEV_SET}")
    
    # Load Force Field
    # Note: force_fields might need x64? Usually parameters are f64.
    # But we haven't enabled x64 yet.
    # If we enable x64 later, will these params be compatible?
    # Yes, JAX handles promotion usually.
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    # Load Model
    print("Loading PrxteinMPNN model...")
    # Load as float32 first (weights are float32)
    jax.config.update("jax_enable_x64", False)
    model = load_model(model_version="v_48_020") 
    
    # Enable x64 for physics
    jax.config.update("jax_enable_x64", True)
    # Cast model to float64
    import equinox as eqx
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, model)
    
    # Create Sampler
    sampler_fn = make_sample_sequences(
        model=model,
        decoding_order_fn=random_decoding_order,
        sampling_strategy="temperature" # Standard sampling
    )
    
    spec = SamplingSpecification(
        inputs=["dummy"], # Dummy input, not used by _sample_batch in this context
        num_samples=1, # 1 sequence per backbone sample
        temperature=0.1, # Low temp for design
        batch_size=1
    )
    
    key = jax.random.PRNGKey(0)
    results = []
    
    for pdb_id in DEV_SET:
        print(f"\nProcessing {pdb_id}...")
        atom_array = download_and_load_pdb(pdb_id)
        if atom_array is None: continue
        
        # Extract full coords for MD and BB coords for Protein object
        # Note: We need to map the perturbed full coords back to BB coords for PrxteinMPNN
        full_coords_np, res_names, atom_names, bb_coords_orig, res_idx, aatypes = extract_system_from_biotite(atom_array)
        
        if full_coords_np is None: continue
        
        params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
        coords = jnp.array(full_coords_np)
        
        # Helper to extract BB from full coords (assuming order matches)
        def extract_bb_from_full(full_c, r_names, a_names):
            # This is tricky because full_c is flat (N_atoms, 3)
            # We need to reconstruct residue structure
            # But we know the order from extract_system_from_biotite
            # Let's just use the atom names list
            
            curr_idx = 0
            new_bb_list = []
            for r_name in r_names:
                std_atoms = residue_constants.residue_atoms.get(r_name, [])
                n_atoms = len(std_atoms)
                res_atoms = full_c[curr_idx : curr_idx + n_atoms]
                curr_idx += n_atoms
                
                res_bb = np.zeros((4, 3))
                # N, CA, C, O
                for k, atom in enumerate(["N", "CA", "C", "O"]):
                    if atom in std_atoms:
                        idx = std_atoms.index(atom)
                        res_bb[k] = res_atoms[idx]
                new_bb_list.append(res_bb)
            return np.array(new_bb_list)

        # 1. Gaussian Noise
        print("  Generating Gaussian samples...")
        for scale in [0.1, 0.5]:
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                noisy_coords = apply_gaussian_noise(coords, scale, subkey)
                
                # Extract BB
                bb_noisy = extract_bb_from_full(np.array(noisy_coords), res_names, atom_names)
                
                # Create Protein
                prot = coords_to_protein(bb_noisy, aatypes, res_idx)
                
                # Sample Sequence
                # We need to shard if using mesh, but here we run single device probably
                seqs, logits = _sample_batch(spec, prot, sampler_fn)
                
                # Decode sequence
                seq_ints = seqs[0, 0, 0, 0] # batch, sample, noise, temp, len -> just take first
                seq_str = "".join([residue_constants.restypes[i] for i in seq_ints if i < 20])
                
                results.append({
                    "pdb": pdb_id,
                    "method": "gaussian",
                    "param": scale,
                    "sample": i,
                    "sequence": seq_str,
                    "length": len(seq_str)
                })
                
        # 2. MD Sampling
        print("  Generating MD samples...")
        for temp in [300, 400]:
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                try:
                    md_coords = apply_md_sampling(coords, params, temp/300.0, subkey)
                    
                    # Extract BB
                    bb_md = extract_bb_from_full(np.array(md_coords), res_names, atom_names)
                    
                    # Create Protein
                    prot = coords_to_protein(bb_md, aatypes, res_idx)
                    
                    # Sample Sequence
                    seqs, logits = _sample_batch(spec, prot, sampler_fn)
                    
                    # Decode sequence
                    seq_ints = seqs[0, 0, 0, 0]
                    seq_str = "".join([residue_constants.restypes[i] for i in seq_ints if i < 20])
                    
                    results.append({
                        "pdb": pdb_id,
                        "method": "md",
                        "param": temp,
                        "sample": i,
                        "sequence": seq_str,
                        "length": len(seq_str)
                    })
                except Exception as e:
                    print(f"    MD failed: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_scrmsd.csv", index=False)
    print("Saved sequences to benchmark_scrmsd.csv")
    
    # Note about folding
    print("\nTo compute scRMSD:")
    print("1. Run ColabFold/ESMFold on the generated sequences in benchmark_scrmsd.csv")
    print("2. Align predicted structures to original PDBs")
    print("3. Compute RMSD")

if __name__ == "__main__":
    run_benchmark()
