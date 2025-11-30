"""Validate MD engine robustness across different force fields."""
import os
import sys
import time
import numpy as np
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
            if np.any(mask): 
                res_coords[i] = res[mask][0].coord
            elif np.any(res.atom_name == "CA"): 
                res_coords[i] = res[res.atom_name == "CA"][0].coord
            else: 
                res_coords[i] = np.array([0., 0., 0.])
                
        coords_list.append(res_coords)
        
    if not coords_list: return None, None, None
    coords = np.vstack(coords_list)
    return coords, res_names, atom_names

def validate_force_field(ff_path, pdb_id="1UAO"):
    print(f"\nTesting Force Field: {ff_path}")
    
    if not os.path.exists(ff_path):
        print(f"  Error: File not found: {ff_path}")
        return False

    try:
        ff = force_fields.load_force_field(ff_path)
        print("  Force field loaded successfully.")
    except Exception as e:
        print(f"  Error loading force field: {e}")
        return False

    # Load System
    atom_array = download_and_load_pdb(pdb_id)
    if atom_array is None:
        print(f"  Error: Could not load PDB {pdb_id}")
        return False
        
    coords_np, res_names, atom_names = extract_system_from_biotite(atom_array)
    if coords_np is None:
        print("  Error: Could not extract system.")
        return False
        
    print(f"  System loaded: {len(res_names)} residues, {len(atom_names)} atoms.")

    # Parameterize
    try:
        params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
        print("  System parameterized successfully.")
    except Exception as e:
        print(f"  Error parameterizing system: {e}")
        return False

    # Run Simulation
    coords = jnp.array(coords_np)
    key = jax.random.PRNGKey(0)
    
    try:
        print("  Running Minimization...")
        # Short minimization
        start_time = time.time()
        final_coords = simulate.run_simulation(
            params, coords, 
            temperature=300.0, 
            min_steps=100, 
            therm_steps=100,
            implicit_solvent=True,
            key=key
        )
        end_time = time.time()
        
        if jnp.any(jnp.isnan(final_coords)):
            print("  FAILURE: NaNs detected in final coordinates.")
            return False
            
        print(f"  SUCCESS: Simulation completed in {end_time - start_time:.2f}s without NaNs.")
        return True
        
    except Exception as e:
        print(f"  FAILURE: Simulation crashed: {e}")
        return False

if __name__ == "__main__":
    ff_dir = "src/prxteinmpnn/physics/force_fields/eqx"
    force_fields_to_test = [
        os.path.join(ff_dir, "protein19SB.eqx"),
        os.path.join(ff_dir, "amber14-all.eqx")
    ]
    
    results = {}
    for ff_path in force_fields_to_test:
        success = validate_force_field(ff_path)
        results[os.path.basename(ff_path)] = "PASS" if success else "FAIL"
        
    print("\nSummary:")
    for ff, result in results.items():
        print(f"{ff}: {result}")
