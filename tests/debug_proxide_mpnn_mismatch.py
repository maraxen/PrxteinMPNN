
import jax
import jax.numpy as jnp
import numpy as np
from proxide import OutputSpec, parse_structure as parse_rust
from proxide.core.containers import Protein

from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.model.features import top_k

def run_debug(pdb_path: str):
    print(f"DEBUG: Analyzing {pdb_path}")
    k_neighbors = 48

    # 1. Rust/Proxide Implementation (Target)
    spec_rust = OutputSpec()
    spec_rust.compute_rbf = True
    spec_rust.rbf_num_neighbors = k_neighbors
    spec_rust.output_format_target = "mpnn"
    spec_rust.remove_solvent = True
    
    print("Parsing with Proxide (MPNN Target)...")
    rust_dict = parse_rust(pdb_path, spec=spec_rust)
    protein_rust = Protein.from_rust_dict(rust_dict)
    
    rust_rbf = np.array(protein_rust.rbf_features)
    rust_neighbors = np.array(protein_rust.neighbor_indices)

    # 2. Python Implementation (Reference)
    spec_py = OutputSpec()
    spec_py.compute_rbf = False # We will compute in Py
    spec_py.output_format_target = "general" # Atom37
    spec_py.remove_solvent = True

    print("Parsing with Proxide (Atom37 Baseline)...")
    py_dict = parse_rust(pdb_path, spec=spec_py)
    protein_py = Protein.from_rust_dict(py_dict)
    
    # Python Feature Calculation Logic (lifted from ProteinFeatures)
    structure_coordinates = jnp.array(protein_py.coordinates)
    
    # Emulate features.py logic
    # compute_backbone_coordinates extracts (N, CA, C, O, CB)
    backbone_atom_coordinates = compute_backbone_coordinates(structure_coordinates)
    distances = compute_backbone_distance(backbone_atom_coordinates)
    
    # Masking logic (simplified for debug - assuming no mask needed for 1ubq)
    # k-NN
    k = min(k_neighbors, structure_coordinates.shape[0])
    _, py_neighbors = top_k(-distances, k)
    py_neighbors = np.array(py_neighbors, dtype=np.int32)
    
    # RBF
    py_rbf = np.array(compute_radial_basis(backbone_atom_coordinates, py_neighbors))

    # 3. Comparison
    print("\n--- COMPARISON ---")
    print(f"Shapes: Rust RBF {rust_rbf.shape}, Py RBF {py_rbf.shape}")
    print(f"Shapes: Rust Neighbors {rust_neighbors.shape}, Py Neighbors {py_neighbors.shape}")

    # Neighbor check
    neighbor_match = np.allclose(rust_neighbors, py_neighbors)
    print(f"Neighbor Indices Match: {neighbor_match}")
    if not neighbor_match:
        diff_mask = rust_neighbors != py_neighbors
        n_diff = np.sum(diff_mask)
        print(f"  Mismatched Neighbor Indices Count: {n_diff} / {rust_neighbors.size}")
        print("  Sample Mismatches (Idx, Rust, Py):")
        indices = np.where(diff_mask)
        for i in range(min(5, len(indices[0]))):
            r, c = indices[0][i], indices[1][i]
            print(f"    ({r}, {c}): {rust_neighbors[r,c]} vs {py_neighbors[r,c]}")
            
    # RBF check
    rbf_match = np.allclose(rust_rbf, py_rbf, atol=1e-4) # Relaxed tolerance
    print(f"RBF Features Match (tol=1e-4): {rbf_match}")
    
    if not rbf_match:
        diff = np.abs(rust_rbf - py_rbf)
        print(f"  Max Diff: {np.max(diff)}")
        print(f"  Mean Diff: {np.mean(diff)}")
        
        # Check if it's just a permutation?
        # A common issue is the flattening order of the 5x5 pair matrix.
        # Python: (Sender Atom x Receiver Atom)? or (Receiver x Sender)?
        # Let's inspect slice 0
        print("  Checking alignment of first feature vector...")
        print(f"  Rust[0,0,:5]: {rust_rbf[0,0,:5]}")
        print(f"  Py[0,0,:5]:   {py_rbf[0,0,:5]}")

if __name__ == "__main__":
    run_debug("tests/data/1ubq.pdb")
