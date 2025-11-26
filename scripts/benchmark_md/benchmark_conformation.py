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
NUM_SAMPLES = 32
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
    
    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        backbone_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
        if np.sum(backbone_mask) < 4: continue
            
        res_names.append(res_name)
        atom_names.extend(std_atoms)
        
        res_coords = np.full((len(std_atoms), 3), np.nan)
        for i, atom_name in enumerate(std_atoms):
            mask = res.atom_name == atom_name
            if np.any(mask): res_coords[i] = res[mask][0].coord
            elif np.any(res.atom_name == "CA"): res_coords[i] = res[res.atom_name == "CA"][0].coord
            else: res_coords[i] = np.array([0., 0., 0.])
        coords_list.append(res_coords)
        
    if not coords_list: return None, None, None, None
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names, atom_array

def apply_gaussian_noise(coords, scale, key):
    return coords + jax.random.normal(key, coords.shape) * scale

def apply_md_sampling(coords, params, temperature, key):
    return simulate.run_simulation(params, coords, temperature=temperature * 300.0, min_steps=MD_STEPS, therm_steps=MD_THERM, key=key)

def check_ramachandran(atom_array, coords):
    """Check Ramachandran validity."""
    # Update coords
    # We need to map the flat coords back to atom_array
    # This assumes atom_array has same atoms as extracted
    # But extract_system filters atoms.
    # So we should create a new atom_array or update carefully.
    
    # Let's just create a new AtomArray from scratch using the extracted info
    # Or better, just update the filtered atom_array returned by extract_system
    
    # We need to ensure the atom count matches.
    # extract_system returns coords for *standard atoms only*.
    # The atom_array passed in might have more (hydrogens, OXT, etc).
    # We need to filter atom_array to match coords.
    
    # Let's filter atom_array to only include the atoms we extracted
    # This is tricky.
    # Instead, let's just use the fact that we know the residue structure.
    # We can reconstruct a minimal AtomArray for dihedral calculation.
    # Biotite needs N, CA, C atoms for Phi/Psi.
    
    # We extracted N, CA, C, O for every residue (checked backbone_mask).
    # So we can just pull them out from coords.
    
    # Reconstruct simple array
    # We need to know which rows in coords correspond to N, CA, C.
    # We have res_names and atom_names.
    
    phi_psi_list = []
    
    # We can use biotite.structure.dihedral_backbone if we have an AtomArray.
    # Let's build a minimal one.
    
    # Create an empty AtomArray
    # We need to know total atoms.
    n_atoms = len(coords)
    new_array = struc.AtomArray(n_atoms)
    new_array.coord = coords
    new_array.res_id = np.zeros(n_atoms, dtype=int) # Dummy
    new_array.chain_id = np.full(n_atoms, "A", dtype="U1")
    new_array.atom_name = np.full(n_atoms, "   ", dtype="U4")
    
    # Fill in details
    curr_idx = 0
    res_id = 1
    for r_name in atom_array.res_name: # This iterates over atoms, not residues? No, atom_array.res_name is array.
        pass
    
    # Let's iterate over the res_names/atom_names we extracted
    # But we need to pass res_names and atom_names to this function?
    # No, we can just do it in the loop.
    pass

def compute_dihedrals_from_coords(coords, res_names):
    """Compute Phi/Psi angles manually or via Biotite."""
    # Construct a minimal AtomArray
    # We need N, CA, C for each residue.
    
    # Flattened list of atoms
    atoms = []
    curr_idx = 0
    
    # We need to assign res_id to group them
    res_ids = []
    atom_names_list = []
    
    for i, r_name in enumerate(res_names):
        std_atoms = residue_constants.residue_atoms.get(r_name, [])
        n_atoms = len(std_atoms)
        
        for j, atom in enumerate(std_atoms):
            atom_names_list.append(atom)
            res_ids.append(i + 1)
            
        curr_idx += n_atoms
        
    new_array = struc.AtomArray(len(coords))
    new_array.coord = coords
    new_array.res_id = np.array(res_ids)
    new_array.atom_name = np.array(atom_names_list)
    new_array.chain_id = np.full(len(coords), "A")
    
    # Calculate dihedrals
    # biotite.structure.dihedral_backbone returns (phi, psi, omega)
    # It returns shape (N_res, 3)
    try:
        phi, psi, omega = struc.dihedral_backbone(new_array)
    except Exception:
        return np.array([]), np.array([])
        
    # Convert to degrees
    phi = np.degrees(phi)
    psi = np.degrees(psi)
    
    return phi, psi

def is_allowed(phi, psi):
    """Check if Phi/Psi is in allowed region (General)."""
    # Simple General Region Check
    # Alpha: -180 < phi < -20, -100 < psi < 45 (Broad)
    # Beta: -180 < phi < -20, 45 < psi < 180 (Broad)
    # Plus some allowed regions for Gly/Pro?
    # Let's use a very broad "Not Forbidden" check.
    # Forbidden: phi > 0 (except Gly), or steric clash regions.
    # Let's just check if it falls in the "Favored" or "Allowed" regions of a standard plot.
    # For benchmark, we compare MD vs Gaussian.
    # Gaussian often puts atoms in random places, so phi/psi will be uniform or weird.
    # MD should keep them in valid regions.
    
    # Let's use a simple box for Alpha/Beta
    is_alpha = (phi > -160) & (phi < -20) & (psi > -100) & (psi < 50)
    is_beta = (phi > -180) & (phi < -20) & (psi > 50) & (psi < 180)
    is_left_alpha = (phi > 20) & (phi < 100) & (psi > 0) & (psi < 100) # Rare but possible (Gly)
    
    return is_alpha | is_beta | is_left_alpha

def run_benchmark():
    print("Benchmarking Conformational Validity...")
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    results = []
    key = jax.random.PRNGKey(0)
    
    for pdb_id in DEV_SET:
        print(f"\nProcessing {pdb_id}...")
        atom_array = download_and_load_pdb(pdb_id)
        if atom_array is None: continue
        
        coords_np, res_names, atom_names, filtered_array = extract_system_from_biotite(atom_array)
        if coords_np is None: continue
        
        params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
        coords = jnp.array(coords_np)
        
        # 1. Baseline
        phi, psi = compute_dihedrals_from_coords(coords_np, res_names)
        valid = is_allowed(phi, psi)
        pct_valid = np.mean(valid) * 100 if len(valid) > 0 else 0
        results.append({"pdb": pdb_id, "method": "baseline", "param": 0, "valid_pct": pct_valid})
        
        # 2. Gaussian
        print("  Method: Gaussian...")
        for scale in [0.1, 0.2, 0.5]:
            valid_pcts = []
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                noisy = apply_gaussian_noise(coords, scale, subkey)
                phi, psi = compute_dihedrals_from_coords(np.array(noisy), res_names)
                valid = is_allowed(phi, psi)
                valid_pcts.append(np.mean(valid) * 100 if len(valid) > 0 else 0)
            results.append({"pdb": pdb_id, "method": "gaussian", "param": scale, "valid_pct": np.mean(valid_pcts)})
            
        # 3. MD
        print("  Method: MD...")
        for temp in [300, 400, 500]:
            valid_pcts = []
            for i in range(NUM_SAMPLES):
                key, subkey = jax.random.split(key)
                try:
                    md_coords = apply_md_sampling(coords, params, temp/300.0, subkey)
                    phi, psi = compute_dihedrals_from_coords(np.array(md_coords), res_names)
                    valid = is_allowed(phi, psi)
                    valid_pcts.append(np.mean(valid) * 100 if len(valid) > 0 else 0)
                except Exception:
                    pass
            if valid_pcts:
                results.append({"pdb": pdb_id, "method": "md", "param": temp, "valid_pct": np.mean(valid_pcts)})

    df = pd.DataFrame(results)
    df.to_csv("benchmark_conformation.csv", index=False)
    print("Saved results to benchmark_conformation.csv")

if __name__ == "__main__":
    run_benchmark()
