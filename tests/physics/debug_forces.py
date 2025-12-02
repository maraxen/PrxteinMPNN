import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, energy
from termcolor import colored

# PrxteinMPNN Imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import force_fields, jax_md_bridge, system, bonded, generalized_born, cmap
from prxteinmpnn.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb

PDB_PATH = "data/pdb/1UBQ.pdb"
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"

# Enable x64
jax.config.update("jax_enable_x64", True)

def run_force_debug():

    print(colored("===========================================================", "cyan"))
    print(colored("   Debug Forces: Component-Wise Finite Difference", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Setup System
    print(colored("\n[1] Setting up JAX MD System...", "yellow"))
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    # Get topology info
    residues = []
    atom_names = []
    atom_counts = []
    
    # We need to manually parse atom_array to get residues/names in correct format
    # Biotite atom_array has res_name and atom_name
    
    curr_res_id = -1
    curr_res_name = ""
    curr_count = 0
    
    # Iterate atoms
    # Note: Biotite arrays are 0-indexed
    res_ids = atom_array.res_id
    res_names = atom_array.res_name
    a_names = atom_array.atom_name
    
    # Group by residue
    # We assume sorted by residue
    unique_res_ids = np.unique(res_ids)
    
    # This is a bit manual, let's use the same logic as validation script if possible.
    # But here we just need lists.
    
    # Simple loop
    last_id = -999
    for i in range(len(atom_array)):
        rid = res_ids[i]
        rname = res_names[i]
        aname = a_names[i]
        
        if rid != last_id:
            if last_id != -999:
                atom_counts.append(curr_count)
                residues.append(curr_res_name)
            curr_count = 0
            curr_res_name = rname
            last_id = rid
            
        # Fix H -> H1 for N-term
        if i == 0 and aname == "H": aname = "H1"
        
        atom_names.append(aname)
        curr_count += 1
        
    # Append last
    atom_counts.append(curr_count)
    residues.append(curr_res_name)
    
    print(f"Loaded {len(residues)} residues, {len(atom_names)} atoms.")
    
    # Load FF
    ff = force_fields.load_force_field(FF_EQX_PATH)
    system_params = jax_md_bridge.parameterize_system(ff, residues, atom_names, atom_counts)
    
    # Positions
    positions = jnp.array(atom_array.coord, dtype=jnp.float64) # Use x64
    
    # Perturb linear atoms (Hack for Hydride issue)
    # We know 1215 (HH22) is linear.
    # Let's find all NH2 hydrogens and perturb them slightly.
    print(colored("\n[1b] Perturbing NH2 Hydrogens to avoid singularity...", "yellow"))
    
    # Find indices of NH2 hydrogens (HH21, HH22, etc.)
    # In 1UBQ, Arg is residue.
    # Atom names in Arg: ... NH1, NH2, HH11, HH12, HH21, HH22
    # We want to perturb HH22 specifically, or just all HH* atoms attached to NH2.
    
    # Simple perturbation: Add random noise to ALL atoms?
    # Or just specific ones.
    # Let's add small noise to everything to be safe.
    # 1e-3 A noise.
    
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, positions.shape) * 0.01 # 0.01 A noise
    positions = positions + noise
    
    print("Added 0.01 A noise to all positions.")

    
    # Displacement
    displacement_fn, shift_fn = space.free()
    
    # 2. Define Component Energy Functions
    print(colored("\n[2] Defining Component Energy Functions...", "yellow"))
    
    # Bond
    e_bond_fn = bonded.make_bond_energy_fn(displacement_fn, system_params['bonds'], system_params['bond_params'])
    
    # Angle
    e_angle_fn = bonded.make_angle_energy_fn(displacement_fn, system_params['angles'], system_params['angle_params'])
    
    # Torsion (Dihedral + Improper)
    e_dih_fn = bonded.make_dihedral_energy_fn(displacement_fn, system_params['dihedrals'], system_params['dihedral_params'])
    e_imp_fn = bonded.make_dihedral_energy_fn(displacement_fn, system_params['impropers'], system_params['improper_params'])
    
    def e_torsion_total(r):
        return e_dih_fn(r) + e_imp_fn(r)
        
    def e_torsion_proper(r):
        return e_dih_fn(r)
        
    def e_torsion_improper(r):
        return e_imp_fn(r)

        
    # CMAP
    def e_cmap_fn(r):
        cmap_torsions = system_params["cmap_torsions"]
        cmap_indices = system_params["cmap_indices"]
        cmap_coeffs = system_params["cmap_coeffs"]
        
        phi_indices = cmap_torsions[:, 0:4]
        psi_indices = cmap_torsions[:, 1:5]
        
        phi = system.compute_dihedral_angles(r, phi_indices, displacement_fn)
        psi = system.compute_dihedral_angles(r, psi_indices, displacement_fn)
        
        # Use (psi, phi) ordering
        return cmap.compute_cmap_energy(psi, phi, cmap_indices, cmap_coeffs)

    # GBSA (Implicit Solvent)
    # We need to setup GBSA function with correct masks
    # Replicate system.py logic
    
    charges = system_params["charges"]
    sigmas = system_params["sigmas"]
    gb_radii = system_params["gb_radii"]
    scaled_radii = system_params["scaled_radii"]
    scale_matrix_vdw = system_params.get("scale_matrix_vdw")
    scale_matrix_elec = system_params.get("scale_matrix_elec")
    
    # GBSA Masks
    gb_mask = jnp.ones_like(scale_matrix_vdw)
    mask_12_13 = scale_matrix_elec == 0.0
    mask_14 = (scale_matrix_elec > 0.0) & (scale_matrix_elec < 0.9)
    gb_energy_mask = jnp.where(mask_12_13, 1.0, scale_matrix_elec)
    gb_energy_mask = jnp.where(mask_14, 0.0, gb_energy_mask)
    
    def e_gbsa_fn(r):
        e_gb, born_radii = generalized_born.compute_gb_energy(
            r, charges, gb_radii, 
            solvent_dielectric=78.5, solute_dielectric=1.0, dielectric_offset=0.09,
            mask=gb_mask, energy_mask=gb_energy_mask, scaled_radii=scaled_radii
        )
        return e_gb
        
    # LJ and Coulomb
    epsilons = system_params["epsilons"]
    def e_lj_fn(r):
        # Replicate system.py compute_lj
        dr = space.map_product(displacement_fn)(r, r)
        dist = space.distance(dr)
        sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
        eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
        e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
        if scale_matrix_vdw is not None:
            e_lj = e_lj * scale_matrix_vdw
        return 0.5 * jnp.sum(e_lj)
        
    def e_coul_fn(r):
        # Replicate system.py compute_electrostatics (direct part)
        dr = space.map_product(displacement_fn)(r, r)
        dist = space.distance(dr)
        q_ij = charges[:, None] * charges[None, :]
        dist_safe = dist + 1e-6
        e_coul = 332.0637 * (q_ij / dist_safe)
        if scale_matrix_elec is not None:
            e_coul = e_coul * scale_matrix_elec
        return 0.5 * jnp.sum(e_coul)

        
    # 3. Finite Difference Check
    print(colored("\n[3] Running Finite Difference Checks...", "yellow"))
    
    def check_component(name, energy_fn, atom_idx=None):
        print(f"\nChecking {name}...")
        
        # Analytical Force
        # F = -grad(U)
        grad_fn = jax.grad(energy_fn)
        
        # Print Energy
        e_val = energy_fn(positions)
        print(f"  Total Energy ({name}): {e_val}")
        
        forces_analytical = -grad_fn(positions)

        
        # Finite Difference
        epsilon = 1e-4
        
        # If atom_idx is provided, only check that atom
        # Otherwise check a few random atoms + max force atom
        
        indices_to_check = []
        if atom_idx is not None:
            indices_to_check = [atom_idx]
        else:
            # Pick max force atom
            f_mag = jnp.linalg.norm(forces_analytical, axis=1)
            max_idx = jnp.argmax(f_mag)
            indices_to_check.append(int(max_idx))
            # And a few random
            indices_to_check.extend([0, 10, 100])
            
        max_rel_error = 0.0
        
        for idx in indices_to_check:
            # Check x, y, z
            f_ana = forces_analytical[idx]
            f_fd = np.zeros(3)
            
            for axis in range(3):
                # Perturb +eps
                pos_p = positions.at[idx, axis].add(epsilon)
                e_p = energy_fn(pos_p)
                
                # Perturb -eps
                pos_m = positions.at[idx, axis].add(-epsilon)
                e_m = energy_fn(pos_m)
                
                # Central difference: F = -(E_p - E_m) / (2*eps)
                f_fd[axis] = -(e_p - e_m) / (2 * epsilon)
                
            # Compare
            diff = np.abs(f_ana - f_fd)
            rel_error = np.linalg.norm(diff) / (np.linalg.norm(f_fd) + 1e-6)
            
            if rel_error > max_rel_error:
                max_rel_error = rel_error
                
            print(f"  Atom {idx} ({atom_names[idx]}):")
            print(f"    Ana: {f_ana}")
            print(f"    FD:  {f_fd}")
            print(f"    Diff: {diff} (Rel: {rel_error:.6f})")
            
            if rel_error > 1e-3:
                print(colored(f"    FAIL: Large discrepancy!", "red"))
                
                # If Torsion, print details
                if "Torsion" in name:
                    # Find torsions involving this atom
                    print(f"    Inspecting Torsions for Atom {idx}...")
                    dihedrals = system_params['dihedrals']
                    d_params = system_params['dihedral_params']
                    
                    found = False
                    for i, d in enumerate(dihedrals):
                        if idx in d:
                            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                            dp = d_params[i]
                            print(f"      Torsion: {p1}-{p2}-{p3}-{p4} Params: {dp}")
                            
                            # Check bond lengths involved
                            # p1-p2, p2-p3, p3-p4
                            r1 = positions[p1]
                            r2 = positions[p2]
                            r3 = positions[p3]
                            r4 = positions[p4]
                            
                            d12 = jnp.linalg.norm(displacement_fn(r1, r2))
                            d23 = jnp.linalg.norm(displacement_fn(r2, r3))
                            d34 = jnp.linalg.norm(displacement_fn(r3, r4))
                            
                            print(f"      Distances: {d12:.4f}, {d23:.4f}, {d34:.4f}")
                            
                            if d12 < 0.1 or d23 < 0.1 or d34 < 0.1:
                                print(colored("      WARNING: Short bond length!", "red"))
                                
                            found = True
                    if not found:
                        print("      No torsions found for this atom (maybe Improper?)")
                        
                    # Check Impropers
                    impropers = system_params['impropers']
                    i_params = system_params['improper_params']
                    for i, d in enumerate(impropers):
                        if idx in d:
                            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                            ip = i_params[i]
                            print(f"      Improper: {p1}-{p2}-{p3}-{p4} Params: {ip}")
                            
                            r1 = positions[p1]
                            r2 = positions[p2]
                            r3 = positions[p3]
                            r4 = positions[p4]
                            
                            # Improper angle usually involves planes.
                            # Check distances anyway.
                            d12 = jnp.linalg.norm(displacement_fn(r1, r2))
                            d23 = jnp.linalg.norm(displacement_fn(r2, r3))
                            d34 = jnp.linalg.norm(displacement_fn(r3, r4))
                            print(f"      Distances: {d12:.4f}, {d23:.4f}, {d34:.4f}")
                            
                            # Calculate Bond Angles
                            # i-j-k
                            v_ji = displacement_fn(r1, r2)
                            v_jk = displacement_fn(r3, r2) # k - j
                            cos_theta1 = jnp.dot(v_ji, v_jk) / (jnp.linalg.norm(v_ji) * jnp.linalg.norm(v_jk))
                            theta1 = jnp.degrees(jnp.arccos(jnp.clip(cos_theta1, -1.0, 1.0)))
                            
                            # j-k-l
                            v_kj = displacement_fn(r2, r3) # j - k
                            v_kl = displacement_fn(r4, r3) # l - k
                            cos_theta2 = jnp.dot(v_kj, v_kl) / (jnp.linalg.norm(v_kj) * jnp.linalg.norm(v_kl))
                            theta2 = jnp.degrees(jnp.arccos(jnp.clip(cos_theta2, -1.0, 1.0)))
                            
                            print(f"      Angles: {theta1:.2f}, {theta2:.2f}")
                            
                            # Calculate v, w norms (singularity check)
                            b0 = displacement_fn(r1, r2)
                            b1 = displacement_fn(r3, r2) # k - j
                            b2 = displacement_fn(r4, r3) # l - k
                            
                            # Use bonded.py logic
                            # b0 = i - j
                            # b1 = k - j
                            # b2 = l - k
                            # But bonded.py uses:
                            # b0 = r_i - r_j
                            # b1 = r_k - r_j
                            # b2 = r_l - r_k
                            
                            b0 = displacement_fn(r1, r2)
                            b1 = displacement_fn(r3, r2)
                            b2 = displacement_fn(r4, r3)
                            
                            b1_norm = jnp.linalg.norm(b1) + 1e-8
                            b1_unit = b1 / b1_norm
                            
                            v = b0 - jnp.dot(b0, b1_unit) * b1_unit
                            w = b2 - jnp.dot(b2, b1_unit) * b1_unit
                            
                            v_norm = jnp.linalg.norm(v)
                            w_norm = jnp.linalg.norm(w)
                            
                            print(f"      Proj Norms: v={v_norm:.6f}, w={w_norm:.6f}")
                            
                            if v_norm < 1e-3 or w_norm < 1e-3:
                                print(colored("      WARNING: Near Singularity (v or w ~ 0)!", "red"))


            else:
                print(colored(f"    PASS", "green"))

                
        return max_rel_error

    # Check Components
    check_component("Bond", e_bond_fn)
    check_component("Angle", e_angle_fn)
    check_component("Torsion_Total", e_torsion_total)
    check_component("Torsion_Proper", e_torsion_proper)
    check_component("Torsion_Improper", e_torsion_improper)
    check_component("CMAP", e_cmap_fn)
    check_component("GBSA", e_gbsa_fn)
    check_component("LJ", e_lj_fn)
    check_component("Coulomb", e_coul_fn)


    
    # Check Atom 1159 (NH2) specifically if requested
    # But indices might differ slightly if Hydride added atoms differently?
    # Let's find NH2
    nh2_indices = [i for i, n in enumerate(atom_names) if n == "NH2"]
    if nh2_indices:
        print(f"\nChecking specific NH2 atoms: {nh2_indices}")
        for idx in nh2_indices:
            # check_component(f"Torsion_Proper (Atom {idx})", e_torsion_proper, atom_idx=idx)
            # check_component(f"Torsion_Improper (Atom {idx})", e_torsion_improper, atom_idx=idx)
            
            # Isolate single torsions
            if idx == 1202: # Focus on the problematic one
                print(colored(f"\nIsolating Torsions for Atom {idx}...", "yellow"))
                dihedrals = system_params['dihedrals']
                d_params = system_params['dihedral_params']
                
                for i, d in enumerate(dihedrals):
                    if idx in d:
                        # Create a single-torsion energy function
                        # We need to slice params to keep shapes correct (1, 4) and (1, 3)
                        single_d_idx = dihedrals[i:i+1]
                        single_d_param = d_params[i:i+1]
                        
                        e_single = bonded.make_dihedral_energy_fn(displacement_fn, single_d_idx, single_d_param)
                        
                        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                        print(f"\nChecking Single Torsion: {p1}-{p2}-{p3}-{p4}")
                        
                        # Calculate Angles
                        r1 = positions[p1]
                        r2 = positions[p2]
                        r3 = positions[p3]
                        r4 = positions[p4]
                        
                        v_ji = displacement_fn(r1, r2)
                        v_jk = displacement_fn(r3, r2)
                        cos_theta1 = jnp.dot(v_ji, v_jk) / (jnp.linalg.norm(v_ji) * jnp.linalg.norm(v_jk))
                        theta1 = jnp.degrees(jnp.arccos(jnp.clip(cos_theta1, -1.0, 1.0)))
                        
                        v_kj = displacement_fn(r2, r3)
                        v_kl = displacement_fn(r4, r3)
                        cos_theta2 = jnp.dot(v_kj, v_kl) / (jnp.linalg.norm(v_kj) * jnp.linalg.norm(v_kl))
                        theta2 = jnp.degrees(jnp.arccos(jnp.clip(cos_theta2, -1.0, 1.0)))
                        
                        print(f"  Angles: {theta1:.2f}, {theta2:.2f}")
                        
                        if abs(theta1 - 180.0) < 1.0 or abs(theta2 - 180.0) < 1.0:
                            print(colored("  WARNING: Linear Angle Detected!", "red"))
                            print(f"  Coords:")
                            print(f"    r1 ({atom_names[p1]}): {r1}")
                            print(f"    r2 ({atom_names[p2]}): {r2}")
                            print(f"    r3 ({atom_names[p3]}): {r3}")
                            print(f"    r4 ({atom_names[p4]}): {r4}")
                        
                        check_component(f"Torsion_{i}", e_single, atom_idx=idx)





if __name__ == "__main__":
    run_force_debug()
