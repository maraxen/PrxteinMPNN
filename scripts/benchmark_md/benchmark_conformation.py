"""Benchmark Conformational Validity (Ramachandran)."""
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

# PrxteinMPNN imports
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from prxteinmpnn.utils import residue_constants

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Constants
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
NUM_SAMPLES = 8
MD_STEPS = 100
MD_THERM = 500

# --- Shared Helpers ---
def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception:
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
    
    # We also need to track indices of N, CA, C for each residue for dihedral calc
    # Since we flatten everything, we need absolute indices into the coordinate array
    n_indices = []
    ca_indices = []
    c_indices = []
    
    current_atom_idx = 0
    
    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4: continue
            
        res_names.append(res_name)
        atom_names.extend(std_atoms)
        
        res_coords = np.full((len(std_atoms), 3), np.nan)
        
        # Find indices for this residue
        res_n_idx = -1
        res_ca_idx = -1
        res_c_idx = -1
        
        for i, atom_name in enumerate(std_atoms):
            mask = res.atom_name == atom_name
            if np.any(mask): 
                res_coords[i] = res[mask][0].coord
            elif np.any(res.atom_name == "CA"): 
                res_coords[i] = res[res.atom_name == "CA"][0].coord
            else: 
                res_coords[i] = np.array([0., 0., 0.])
                
            if atom_name == "N": res_n_idx = current_atom_idx + i
            if atom_name == "CA": res_ca_idx = current_atom_idx + i
            if atom_name == "C": res_c_idx = current_atom_idx + i
            
        coords_list.append(res_coords)
        
        if res_n_idx != -1 and res_ca_idx != -1 and res_c_idx != -1:
            n_indices.append(res_n_idx)
            ca_indices.append(res_ca_idx)
            c_indices.append(res_c_idx)
        else:
            # Should not happen given backbone check, but for safety
            pass

        current_atom_idx += len(std_atoms)
        
    if not coords_list: return None, None, None, None, None, None, None
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names, atom_array, np.array(n_indices), np.array(ca_indices), np.array(c_indices)

def apply_gaussian_noise(coords, scale, key):
    return coords + jax.random.normal(key, coords.shape) * scale

def compute_dihedrals_jax(coords, n_idx, ca_idx, c_idx):
    """Compute Phi and Psi angles using JAX.
    
    Args:
        coords: (N_atoms, 3)
        n_idx: (N_res,) indices of N atoms
        ca_idx: (N_res,) indices of CA atoms
        c_idx: (N_res,) indices of C atoms
        
    Returns:
        phi: (N_res,) in degrees
        psi: (N_res,) in degrees
    """
    def compute_dihedral(p1, p2, p3, p4):
        b0 = -1.0 * (p2 - p1)
        b1 = p3 - p2
        b2 = p4 - p3

        b1 /= jnp.linalg.norm(b1, axis=-1, keepdims=True)

        v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1

        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1, v) * w, axis=-1)

        return jnp.degrees(jnp.arctan2(y, x))

    # Phi
    # Valid for residues 1 to N-1
    c_prev = coords[c_idx[:-1]]
    n_curr = coords[n_idx[1:]]
    ca_curr = coords[ca_idx[1:]]
    c_curr_phi = coords[c_idx[1:]]
    
    phi_vals = compute_dihedral(c_prev, n_curr, ca_curr, c_curr_phi)
    # Pad first residue with 0.0
    phi = jnp.concatenate([jnp.array([0.0]), phi_vals])

    # Psi
    # Valid for residues 0 to N-2
    n_curr_psi = coords[n_idx[:-1]]
    ca_curr_psi = coords[ca_idx[:-1]]
    c_curr_psi = coords[c_idx[:-1]]
    n_next = coords[n_idx[1:]]
    
    psi_vals = compute_dihedral(n_curr_psi, ca_curr_psi, c_curr_psi, n_next)
    # Pad last residue
    psi = jnp.concatenate([psi_vals, jnp.array([0.0])])
    
    return phi, psi

def is_allowed_jax(phi, psi):
    """Check Ramachandran validity (JAX compatible)."""
    # Simple General Region Check (Broad)
    is_alpha = (phi > -160) & (phi < -20) & (psi > -100) & (psi < 50)
    is_beta = (phi > -180) & (phi < -20) & (psi > 50) & (psi < 180)
    is_left_alpha = (phi > 20) & (phi < 100) & (psi > 0) & (psi < 100)
    
    return is_alpha | is_beta | is_left_alpha

def run_benchmark():
    print("Benchmarking Conformational Validity (Optimized with VMAP)...")
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    results = []
    key = jax.random.PRNGKey(0)
    
    for pdb_id in DEV_SET:
        print(f"\nProcessing {pdb_id}...")
        atom_array = download_and_load_pdb(pdb_id)
        if atom_array is None: continue
        
        coords_np, res_names, atom_names, filtered_array, n_idx, ca_idx, c_idx = extract_system_from_biotite(atom_array)
        if coords_np is None: continue
        
        params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
        coords = jnp.array(coords_np)
        n_idx = jnp.array(n_idx)
        ca_idx = jnp.array(ca_idx)
        c_idx = jnp.array(c_idx)
        
        # 1. Baseline
        phi, psi = compute_dihedrals_jax(coords, n_idx, ca_idx, c_idx)
        valid = is_allowed_jax(phi, psi)
        if len(valid) > 2:
            pct_valid = jnp.mean(valid[1:-1]) * 100
        else:
            pct_valid = 0.0
        results.append({"pdb": pdb_id, "method": "baseline", "param": 0, "valid_pct": float(pct_valid)})
        
        # 2. Gaussian (VMAP)
        print("  Method: Gaussian...")
        
        @jax.jit
        def run_gaussian_batch(keys, scale):
            # vmap over keys
            def single_run(k):
                noisy = apply_gaussian_noise(coords, scale, k)
                phi, psi = compute_dihedrals_jax(noisy, n_idx, ca_idx, c_idx)
                valid = is_allowed_jax(phi, psi)
                return jnp.mean(valid[1:-1]) * 100
            
            return jax.vmap(single_run)(keys)

        for scale in [0.1, 0.2, 0.5]:
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, NUM_SAMPLES)
            scores = run_gaussian_batch(subkeys, scale)
            results.append({"pdb": pdb_id, "method": "gaussian", "param": scale, "valid_pct": float(jnp.mean(scores))})
            
        # 3. MD (VMAP)
        print("  Method: MD...")
        
        @jax.jit
        def run_md_batch(keys, temp_kelvin):
            # vmap over keys
            def single_run(k):
                # simulate.run_simulation is JIT-able
                md_coords = simulate.run_simulation(
                    params, coords, temperature=temp_kelvin, min_steps=MD_STEPS, therm_steps=MD_THERM,
                    implicit_solvent=True, solvent_dielectric=78.5, solute_dielectric=1.0, key=k
                )
                phi, psi = compute_dihedrals_jax(md_coords, n_idx, ca_idx, c_idx)
                valid = is_allowed_jax(phi, psi)
                return jnp.mean(valid[1:-1]) * 100
            
            return jax.vmap(single_run)(keys)

        for temp in [250, 298, 350, 450]:
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, NUM_SAMPLES)
            scores = run_md_batch(subkeys, float(temp))
            results.append({"pdb": pdb_id, "method": "md", "param": temp, "valid_pct": float(jnp.mean(scores))})

    df = pd.DataFrame(results)
    df.to_csv("benchmark_conformation.csv", index=False)
    print("Saved results to benchmark_conformation.csv")

if __name__ == "__main__":
    run_benchmark()
