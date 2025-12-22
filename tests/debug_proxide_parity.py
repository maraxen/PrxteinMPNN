
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.io.parsing import parse_structure
from prxteinmpnn.utils.residue_constants import atom_order

def debug_parity():
    # Load structure
    pdb_path = "tests/data/1ubq.pdb"
    # prxteinmpnn.io.parsing.parse_structure returns a Protein directly (not generator, per wrapper)
    # Wait, dispatch.py says: return _parse_structure(...) which returns Protein.
    # But parse_input yields.
    # Let's check dispatch.py line 51 again. Yes, returns Protein directly.
    protein = parse_structure(pdb_path)
    
    # 1. Inspect Proxide Output
    print(f"Proxide RBF: {protein.rbf_features.shape}")
    if protein.neighbor_indices is None:
        print("Proxide Neighbors: None (Legacy binary detected)")
    else:
        print(f"Proxide Neighbors: {protein.neighbor_indices.shape}")
    
    # 2. Compute Python Features (Standard / Correct for PrxteinMPNN)
    coords = jnp.array(protein.coordinates)
    
    backbone_coords_python = compute_backbone_coordinates(coords) 
    print(f"Python Backbone Order (N,CA,C,O,CB): {backbone_coords_python.shape}")
    
    distances = compute_backbone_distance(backbone_coords_python)
    mask = jnp.ones((coords.shape[0],))
    dist_masked = jnp.where((mask[:,None]*mask[None,:]).astype(bool), distances, jnp.inf)
    
    k = 48 # standard K_NEIGHBORS
    _, neighbors_python = jax.lax.top_k(-dist_masked, k)
    
    # Use Python neighbors for comparison since Rust ones are missing
    neighbors_to_use = neighbors_python
    
    rbf_python = compute_radial_basis(backbone_coords_python, neighbors_to_use)
    
    # 3. Compute Python Features with "Rust Order" (N, CA, C, CB, O) hypothesis
    nitrogen = coords[:, atom_order["N"], :]
    alpha_carbon = coords[:, atom_order["CA"], :]
    carbon = coords[:, atom_order["C"], :]
    beta_carbon_raw = coords[:, atom_order["CB"], :] # Note: using raw CB here to check if Rust uses raw
    oxygen = coords[:, atom_order["O"], :]
    
    # NOTE: compute_backbone_coordinates computes CB. 
    # If Rust uses raw CB, that's another difference.
    # Let's try constructing "Rust Order" using the exact same atoms python uses (computed CB) first.
    
    # computed CB from python pipeline
    # python[4] is CB, python[3] is O.
    cb_computed = backbone_coords_python[:, 4]
    
    backbone_coords_rust_order = jnp.stack([
        backbone_coords_python[:, 0], # N
        backbone_coords_python[:, 1], # CA
        backbone_coords_python[:, 2], # C
        cb_computed,                  # CB (index 3 in Rust order)
        backbone_coords_python[:, 3], # O (index 4 in Rust order)
    ], axis=1)
    
    rbf_python_order = compute_radial_basis(backbone_coords_python, neighbors_to_use)
    rbf_rust_order = compute_radial_basis(backbone_coords_rust_order, neighbors_to_use)
    
    
    # Comparisons
    print("\n--- neighbor_indices Check ---")
    if protein.neighbor_indices is not None:
         neighbors_match = jnp.mean(jnp.array(protein.neighbor_indices) == neighbors_python)
         print(f"Neighbors Match %: {neighbors_match*100:.2f}%")
    else:
         print("Skipping neighbor comparison (proxide neighbors missing)")
    
    print("\n--- RBF Comparison (Proxide vs Python Order (N,CA,C,O,CB)) ---")
    diff_python = jnp.abs(protein.rbf_features - rbf_python_order)
    print(f"Mean Diff: {jnp.mean(diff_python):.6f}")
    
    print("\n--- RBF Comparison (Proxide vs Rust Order (N,CA,C,CB,O)) ---")
    diff_rust = jnp.abs(protein.rbf_features - rbf_rust_order)
    print(f"Mean Diff: {jnp.mean(diff_rust):.6f}")

    if jnp.mean(diff_rust) < jnp.mean(diff_python):
        print("\nCONCLUSION: Proxide likely uses (N, CA, C, CB, O) order (Standard Atom37).")
        print("ACTION: We need to reorder RBF features from Proxide.")
    else:
        print("\nCONCLUSION: Proxide does not match (N, CA, C, CB, O) order either.")



if __name__ == "__main__":
    debug_parity()
