"""Benchmark geometric integrity of noise methods."""
import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from prxteinmpnn.utils import residue_constants
from jax_md import space

# Dev Set PDB IDs
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
NUM_SAMPLES = 32

from prxteinmpnn.io.process import frame_iterator_from_inputs
from prxteinmpnn.utils.data_structures import ProteinTuple

# Dev Set PDB IDs
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
NUM_SAMPLES = 32

def protein_tuple_to_jax_md_input(protein_tuple, ff):
    """Convert ProteinTuple to flat arrays for JAX MD."""
    # 1. Reconstruct sequence
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
        
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        
        count = 0
        for atom_name in std_atoms:
            atom_idx = residue_constants.atom_order.get(atom_name)
            if atom_idx is not None and protein_tuple.atom_mask[i, atom_idx] > 0.5:
                flat_coords.append(protein_tuple.coordinates[i, atom_idx])
                flat_atom_names.append(atom_name)
                count += 1
        atom_counts.append(count)
                
    # Parameterize system to get params
    params = jax_md_bridge.parameterize_system(ff, res_names, flat_atom_names, atom_counts=atom_counts)
    
    return jnp.array(flat_coords), params

def compute_geometry_metrics(coords, params):
    """Compute geometric metrics using JAX MD."""
    displacement_fn, _ = space.free()
    
    # 1. Bond lengths
    bonds = params["bonds"]
    bond_params = params["bond_params"]
    
    r1 = coords[bonds[:, 0]]
    r2 = coords[bonds[:, 1]]
    
    # Direct displacement (free space supports vectorization)
    # Fix: Use vmap for pairwise displacement
    dr = jax.vmap(displacement_fn)(r1, r2)
    d = jnp.linalg.norm(dr, axis=-1)
    
    bond_dev = jnp.abs(d - bond_params[:, 0])
    mean_bond_dev = jnp.mean(bond_dev)
    max_bond_dev = jnp.max(bond_dev)
    
    # 2. Total Energy
    energy_fn = system.make_energy_fn(displacement_fn, params)
    total_energy = energy_fn(coords)
    
    return {
        "mean_bond_dev": float(mean_bond_dev),
        "max_bond_dev": float(max_bond_dev),
        "total_energy": float(total_energy),
        "is_finite": bool(jnp.isfinite(total_energy))
    }

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

def run_benchmark(pdb_set=DEV_SET):
    """Run the geometric integrity benchmark."""
    print(f"Benchmarking Geometric Integrity on {pdb_set}")
    print(f"Samples per condition: {NUM_SAMPLES}")
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    results = []
    key = jax.random.PRNGKey(0)
    
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
                
            coords, params = jax_md_input
            
            proteins[pdb_id] = {
                "coords": coords,
                "params": params
            }
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    for pdb_id in pdb_set:
        if pdb_id not in proteins:
            print(f"Skipping {pdb_id} as it was not pre-loaded.")
            continue
            
        print(f"\nProcessing {pdb_id}...")
        try:
            data = proteins[pdb_id]
            coords = data["coords"]
            params = data["params"]
            
            # 1. Baseline (No Noise)
            print("  Evaluating Baseline...")
            metrics = compute_geometry_metrics(coords, params)
            results.append({"pdb": pdb_id, "method": "baseline", "param": 0, "sample": 0, **metrics})
            
            # 2. Gaussian Noise
            print("  Evaluating Gaussian Noise...")
            GAUSSIAN_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            for scale in GAUSSIAN_SCALES:
                for i in range(NUM_SAMPLES):
                    key, subkey = jax.random.split(key)
                    noisy_coords = apply_gaussian_noise(coords, scale, subkey)
                    metrics = compute_geometry_metrics(noisy_coords, params)
                    results.append({"pdb": pdb_id, "method": "gaussian", "param": scale, "sample": i, **metrics})
            
            # 3. MD Sampling
            print("  Evaluating MD Sampling...")
            MD_TEMPS = [270, 300, 330, 360, 390, 420, 450]
            for temp in MD_TEMPS:
                for i in range(NUM_SAMPLES):
                    key, subkey = jax.random.split(key)
                    try:
                        md_coords = apply_md_sampling(coords, params, temp/300.0, subkey)
                        metrics = compute_geometry_metrics(md_coords, params)
                        results.append({"pdb": pdb_id, "method": "md", "param": temp, "sample": i, **metrics})
                    except Exception as e:
                        print(f"    MD failed for T={temp}, sample={i}: {e}")
                        
        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            
    df = pd.DataFrame(results)
    print("\nResults Summary:")
    if not df.empty:
        print(df.groupby(["method", "param"])[["mean_bond_dev", "total_energy", "is_finite"]].mean())
    df.to_csv("benchmark_geometric_integrity.csv", index=False)
    print("Results saved to benchmark_geometric_integrity.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run geometric integrity benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run on quick dev set (Chignolin).")
    args = parser.parse_args()
    
    target_set = QUICK_DEV_SET if args.quick else DEV_SET
    run_benchmark(target_set)
