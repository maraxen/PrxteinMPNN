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
        
    # Remove hydrogens (we'll add them back via parameterization if needed, 
    # but for now we need to match the force field expectations)
    # Actually, parameterize_system expects standard residues.
    # We need to extract sequence and coordinates for heavy atoms.
    
    res_names = []
    atom_names = []
    coords_list = []
    
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
        
        # Extract coords for all standard atoms
        # Initialize with NaN
        res_coords = np.full((len(std_atoms), 3), np.nan)
        
        for i, atom_name in enumerate(std_atoms):
            mask = res.atom_name == atom_name
            if np.any(mask):
                res_coords[i] = res[mask][0].coord
            else:
                # Missing atom. 
                # For benchmark, we can't easily fix this without pdbfixer.
                # We'll place at CA (bad geometry) or skip.
                # Let's place at CA if available, else 0
                ca_mask = res.atom_name == "CA"
                if np.any(ca_mask):
                    res_coords[i] = res[ca_mask][0].coord
                else:
                    res_coords[i] = np.array([0., 0., 0.])
                    
        coords_list.append(res_coords)
        
    if not coords_list:
        return None, None, None
        
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names

def compute_geometry_metrics(coords, params):
    """Compute geometric metrics using JAX MD."""
    displacement_fn, _ = space.free()
    
    # 1. Bond lengths
    bonds = params["bonds"]
    bond_params = params["bond_params"]
    
    r1 = coords[bonds[:, 0]]
    r2 = coords[bonds[:, 1]]
    
    # Direct displacement (free space supports vectorization)
    dr = displacement_fn(r1, r2)
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

def run_benchmark():
    """Run the geometric integrity benchmark."""
    print(f"Benchmarking Geometric Integrity on Dev Set: {DEV_SET}")
    print(f"Samples per condition: {NUM_SAMPLES}")
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    results = []
    key = jax.random.PRNGKey(0)
    
    for pdb_id in DEV_SET:
        print(f"\nProcessing {pdb_id}...")
        try:
            atom_array = download_and_load_pdb(pdb_id)
            if atom_array is None:
                continue
                
            coords_np, res_names, atom_names = extract_system_from_biotite(atom_array)
            if coords_np is None:
                print(f"  Failed to extract system from {pdb_id}")
                continue
                
            params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
            
            if len(coords_np) != len(params["charges"]):
                print(f"  Mismatch: Coords {len(coords_np)} != Params {len(params['charges'])}")
                continue
                
            coords = jnp.array(coords_np)
            
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
    run_benchmark()
