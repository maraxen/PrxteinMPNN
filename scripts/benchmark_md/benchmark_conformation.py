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
import argparse

# PrxteinMPNN imports
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from prxteinmpnn.utils import residue_constants

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Constants
DEV_SET = ["1UBQ", "1CRN", "1BPTI", "2GB1", "1L2Y"]
QUICK_DEV_SET = ["1UAO"]
NUM_SAMPLES = 8
MD_STEPS = 1000
MD_THERM = 1000

from prxteinmpnn.io.parsing import biotite as parsing_biotite

# --- Shared Helpers ---
def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception:
            return None
    
    # Use our parsing utility which handles Hydride
    # Note: We load as AtomArray (model=1)
    try:
        atom_array = parsing_biotite.load_structure_with_hydride(
            pdb_path, model=1, add_hydrogens=True, remove_solvent=True
        )
        return atom_array
    except Exception as e:
        print(f"Failed to load {pdb_id}: {e}")
        return None

def extract_system_from_biotite(atom_array):
    # Filter for amino acids (already done by remove_solvent mostly, but good to be safe)
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    chains = struc.get_chains(atom_array)
    if len(chains) > 0:
        atom_array = atom_array[atom_array.chain_id == chains[0]]
        
    res_names = []
    atom_names = []
    coords_list = []
    
    # We also need to track indices of N, CA, C for each residue for dihedral calc
    n_indices = []
    ca_indices = []
    c_indices = []
    
    current_atom_idx = 0
    
    # Iterate residues
    # Note: We must preserve the atoms returned by Hydride/Biotite
    for i, res in enumerate(struc.residue_iter(atom_array)):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        
        # Check backbone
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4: continue
            
        res_names.append(res_name)
        
        # Extract all atoms in this residue
        res_atom_names = res.atom_name.tolist()
        res_coords = res.coord
        
        # Fix N-term H -> H1, H2, H3 for Amber (if present)
        if i == 0:
            # Check if we have H1, H2, H3 already
            has_h1 = "H1" in res_atom_names
            has_h2 = "H2" in res_atom_names
            has_h3 = "H3" in res_atom_names
            
            if not (has_h1 and has_h2 and has_h3):
                 # If we have H, rename it to H1 (legacy fallback)
                 for k, name in enumerate(res_atom_names):
                    if name == "H":
                        res_atom_names[k] = "H1"
        
        atom_names.extend(res_atom_names)
        coords_list.append(res_coords)
        
        # Find indices for dihedrals
        # We need relative index within residue
        try:
            n_local = res_atom_names.index("N")
            ca_local = res_atom_names.index("CA")
            c_local = res_atom_names.index("C")
            
            n_indices.append(current_atom_idx + n_local)
            ca_indices.append(current_atom_idx + ca_local)
            c_indices.append(current_atom_idx + c_local)
        except ValueError:
            pass # Should be caught by backbone check above

        current_atom_idx += len(res_atom_names)
        
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

def run_benchmark(pdb_set=DEV_SET, force_field_path="src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"):
    print(f"Benchmarking Conformational Validity (Optimized with VMAP) on {pdb_set}...")
    try:
        if os.path.exists(force_field_path):
            print(f"Loading local force field: {force_field_path}")
            ff = force_fields.load_force_field(force_field_path)
        else:
            raise FileNotFoundError
    except Exception:
        print(f"Force field {force_field_path} not found, falling back to ff14SB from Hub...")
        ff = force_fields.load_force_field_from_hub("ff14SB")
    
    results = []
    key = jax.random.PRNGKey(0)
    
    for pdb_id in pdb_set:
        print(f"\nProcessing {pdb_id}...")
        cache_path = os.path.join("data", "cache", f"{pdb_id}_system.npz")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            data = np.load(cache_path)
            coords_np = data["coords"]
            res_names = data["res_names"].tolist()
            atom_names = data["atom_names"].tolist()
            n_idx = data["n_idx"]
            ca_idx = data["ca_idx"]
            c_idx = data["c_idx"]
        else:
            atom_array = download_and_load_pdb(pdb_id)
            if atom_array is None: continue
            
            coords_np, res_names, atom_names, filtered_array, n_idx, ca_idx, c_idx = extract_system_from_biotite(atom_array)
            if coords_np is None: continue
            
            np.savez(
                cache_path, 
                coords=coords_np, 
                res_names=res_names, 
                atom_names=atom_names, 
                n_idx=n_idx, 
                ca_idx=ca_idx, 
                c_idx=c_idx
            )
        
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
    parser = argparse.ArgumentParser(description="Run conformational validity benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run on quick dev set (Chignolin).")
    parser.add_argument("--force_field", type=str, default="src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx", help="Path to force field file.")
    args = parser.parse_args()
    
    target_set = QUICK_DEV_SET if args.quick else DEV_SET
    run_benchmark(target_set, args.force_field)
