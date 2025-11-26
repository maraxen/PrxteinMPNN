"""Benchmark Computational Scaling (Latency & Memory)."""
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
PDB_ID = "1UBQ"
LENGTHS = [100, 250, 500, 1000, 1500, 2000]
STEPS = 100

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

def create_dummy_system(base_atom_array, target_length):
    """Create a dummy system by replicating the base system."""
    # Filter for protein
    base_atom_array = base_atom_array[struc.filter_amino_acids(base_atom_array)]
    chains = struc.get_chains(base_atom_array)
    if len(chains) > 0:
        base_atom_array = base_atom_array[base_atom_array.chain_id == chains[0]]
        
    base_len = struc.get_residue_count(base_atom_array)
    num_repeats = int(np.ceil(target_length / base_len))
    
    # Replicate
    atoms_list = []
    offset = np.array([50.0, 0.0, 0.0]) # Shift by 50A to avoid overlap
    
    current_res_id_offset = 0
    
    for i in range(num_repeats):
        new_atoms = base_atom_array.copy()
        new_atoms.coord += i * offset
        new_atoms.res_id += current_res_id_offset
        atoms_list.append(new_atoms)
        current_res_id_offset += base_len
        
    combined_atoms = atoms_list[0]
    for i in range(1, len(atoms_list)):
        combined_atoms += atoms_list[i]
        
    # Trim to target length
    # This is a bit tricky with atom array, easier to just take residues
    # But for MD parameterization, we need whole residues.
    # Let's just keep it as is, it might be slightly larger than target_length
    # Or we can slice.
    
    # Extract system info for parameterization
    res_names = []
    atom_names = []
    coords_list = []
    
    count = 0
    for res in struc.residue_iter(combined_atoms):
        if count >= target_length: break
        
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        std_atoms = residue_constants.residue_atoms.get(res_name, [])
        
        # Check backbone
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
        count += 1
        
    if not coords_list: return None, None, None
    
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names

def run_benchmark():
    print("Benchmarking Computational Scaling...")
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    atom_array = download_and_load_pdb(PDB_ID)
    if atom_array is None:
        print(f"Failed to load {PDB_ID}")
        return

    results = []
    key = jax.random.PRNGKey(0)
    
    for length in LENGTHS:
        print(f"\nTarget Length: {length}")
        
        # Create System
        t0 = time.time()
        coords_np, res_names, atom_names = create_dummy_system(atom_array, length)
        if coords_np is None: continue
        
        try:
            params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
            coords = jnp.array(coords_np)
            setup_time = time.time() - t0
            print(f"  Setup Time: {setup_time:.2f}s")
            print(f"  System Size: {len(coords)} atoms")
            
            # Run MD
            # We want to measure compilation time vs execution time
            # simulate.run_simulation does everything.
            # To separate, we might need to call lower level, but run_simulation is what users use.
            # So let's just measure the first call (compile + run) and maybe a second call?
            # But run_simulation runs the whole loop inside JIT.
            
            # Let's measure total time for N steps.
            t_start = time.time()
            
            # We use a short run
            final_coords = simulate.run_simulation(
                params,
                coords,
                temperature=300.0,
                min_steps=10,
                therm_steps=STEPS,
                key=key
            )
            # Block until done
            final_coords.block_until_ready()
            t_end = time.time()
            total_time = t_end - t_start
            
            # Estimate latency per step (ignoring compilation overhead for now, or assuming it dominates for short runs?)
            # For scaling, we care about the trend.
            # Ideally we would run twice to subtract compilation.
            
            print(f"  Total Time ({STEPS} steps): {total_time:.2f}s")
            
            # Try to get memory usage?
            # JAX doesn't easily expose peak memory from python.
            # We can just record time.
            
            results.append({
                "length": length,
                "num_atoms": len(coords),
                "total_time": total_time,
                "setup_time": setup_time,
                "steps": STEPS
            })
            
        except Exception as e:
            print(f"  Failed: {e}")
            # OOM might happen here
            results.append({
                "length": length,
                "error": str(e)
            })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_scaling.csv", index=False)
    print("Saved results to benchmark_scaling.csv")

if __name__ == "__main__":
    run_benchmark()
