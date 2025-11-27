"""Stress Test Stability (Long-Time Horizon)."""
print("DEBUG: Script started")
import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from prxteinmpnn.utils import residue_constants
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
from jax_md import space, simulate as jax_simulate, quantity

# PrxteinMPNN imports
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system, constants

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Constants
PDB_ID = "1UBQ"
TOTAL_STEPS = 10000
LOG_INTERVAL = 100
RMSD_THRESHOLD = 8.0 # Angstroms

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
        
    if not coords_list: return None, None, None
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names

def run_stress_test():
    print(f"Running Stability Stress Test on {PDB_ID}...")
    print(f"Total Steps: {TOTAL_STEPS}, Log Interval: {LOG_INTERVAL}")
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    atom_array = download_and_load_pdb(PDB_ID)
    if atom_array is None: return
    
    coords_np, res_names, atom_names = extract_system_from_biotite(atom_array)
    if coords_np is None: return
    
    params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
    coords = jnp.array(coords_np)
    
    # Setup System
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, params)
    
    # 1. Minimize
    print("Minimizing...")
    r_min = simulate.run_minimization(energy_fn, coords, steps=500)
    
    # 2. Setup NVT
    temperature = 300.0
    dt = 2e-3
    gamma = 0.1
    kT = constants.BOLTZMANN_KCAL * temperature
    
    init_fn, apply_fn = jax_simulate.nvt_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        gamma=gamma
    )
    
    key = jax.random.PRNGKey(0)
    state = init_fn(key, r_min)
    
    # JIT the step function (runs LOG_INTERVAL steps)
    @jax.jit
    def step_block(state):
        def body_fn(i, s):
            return apply_fn(s)
        return jax.lax.fori_loop(0, LOG_INTERVAL, body_fn, state)
    
    # Run Loop
    results = []
    start_coords = r_min
    
    print("Starting Simulation Loop...")
    t0 = time.time()
    
    num_blocks = TOTAL_STEPS // LOG_INTERVAL
    
    for block in range(num_blocks):
        state = step_block(state)
        
        # Compute Metrics
        curr_coords = state.position
        
        # Energy
        e = energy_fn(curr_coords)
        
        # RMSD (align first? usually RMSD is after alignment. jax_md.quantity.rmsd does alignment if not specified? No, it just computes dist. We need alignment.)
        # For stability, we usually align to start structure.
        # jax_md doesn't have built-in alignment rmsd easily accessible?
        # quantity.rmsd(p1, p2) computes sqrt(mean((p1-p2)^2)).
        # If protein rotates, this will be large.
        # We need to align.
        # For now, let's assume Langevin dynamics without momentum conservation might drift?
        # Actually, `space.free` implies no box.
        # We can use `quantity.align_and_compute_rmsd` if available? No.
        # Let's just compute simple RMSD and note if it drifts. Or center it.
        # Better: use `biotite.structure.superimpose` on CPU for logging (slow but accurate).
        
        # Move to CPU for logging
        curr_coords_np = np.array(curr_coords)
        start_coords_np = np.array(start_coords)
        
        # Superimpose
        # We need AtomArray for Biotite superimpose? Or just coords.
        # biotite.structure.superimpose(fixed, mobile)
        # It expects AtomArray or ndarray.
        # If ndarray, it assumes matching atoms.
        
        # Center both
        curr_centered = curr_coords_np - np.mean(curr_coords_np, axis=0)
        start_centered = start_coords_np - np.mean(start_coords_np, axis=0)
        
        # Rotation
        fitted, transformation = struc.superimpose(start_centered, curr_centered)
        rmsd = struc.rmsd(start_centered, fitted)
        
        step_num = (block + 1) * LOG_INTERVAL
        
        print(f"Step {step_num}: Energy={e:.2f}, RMSD={rmsd:.2f} A")
        
        results.append({
            "step": step_num,
            "energy": float(e),
            "rmsd": float(rmsd)
        })
        
        if not np.isfinite(e):
            print("Energy exploded (NaN/Inf). Test Failed.")
            break
            
        if rmsd > RMSD_THRESHOLD:
            print(f"RMSD exceeded threshold ({RMSD_THRESHOLD} A). Unfolding detected.")
            # Don't break, just note it? Or break to save time.
            # User said "Failure Condition: If ... unfolds ... parameters need tuning".
            # We should probably continue to see if it recovers (unlikely) or just stop.
            # Let's stop.
            break
            
    df = pd.DataFrame(results)
    df.to_csv("stress_test_stability.csv", index=False)
    print("Saved results to stress_test_stability.csv")
    
    # Plotting (optional, but good for user)
    # visualize_benchmarks.py will handle it if we add it.

if __name__ == "__main__":
    run_stress_test()
