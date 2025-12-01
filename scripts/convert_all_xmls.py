# File: scripts/convert_all_xmls.py
import os
import glob
import json
import xml.etree.ElementTree as ET
import jax.numpy as jnp
import equinox as eqx
from prxteinmpnn.physics.force_fields import FullForceField, save_force_field

# Units
KJ_TO_KCAL = 0.239005736
NM_TO_ANGSTROM = 10.0

def parse_xml_to_eqx(xml_path: str, output_dir: str):
    ff_name = os.path.basename(xml_path).replace('.xml', '').replace('.ff', '')
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Validate it's a force field file
        if root.tag != 'ForceField':
            return # Skip non-forcefield files silently

        print(f"--- Processing {ff_name} ---")

        # --- 1. Basic Parameter Parsing ---
        type_to_class = {t.attrib['name']: t.attrib['class'] for t in root.findall('AtomTypes/Type')}
        
        class_to_lj = {}
        for a in root.findall('NonbondedForce/Atom'):
            key = a.attrib.get('class', a.attrib.get('type'))
            if key:
                class_to_lj[key] = (float(a.attrib.get('sigma', 1.0)), float(a.attrib.get('epsilon', 0.0)))

        # Initialize Data containers
        charges, sigmas, epsilons = [0.0], [1.0], [0.0]
        hyperparams = {
            "atom_key_to_id": {("UNK", "UNK"): 0}, "id_to_atom_key": [("UNK", "UNK")],
            "atom_class_map": {}, "atom_type_map": {}, "source_files": [os.path.basename(xml_path)],
            "bonds": [], "angles": [], "propers": [], "impropers": [], "cmap_torsions": [],
            "residue_templates": {}
        }
        current_id = 1

        # Parse Residues & Atoms
        for res in root.findall('Residues/Residue'):
            res_name = res.attrib['name']
            
            # Track atoms in this residue for bond mapping (index -> name)
            res_atom_names = []
            
            for atom in res.findall('Atom'):
                atom_name, atom_type = atom.attrib['name'], atom.attrib['type']
                res_atom_names.append(atom_name)
                
                key = (res_name, atom_name)
                
                if key not in hyperparams["atom_key_to_id"]:
                    atom_cls = type_to_class.get(atom_type, atom_type)
                    sigma, epsilon = class_to_lj.get(atom_cls, (1.0, 0.0))
                    
                    hyperparams["atom_key_to_id"][key] = current_id
                    hyperparams["id_to_atom_key"].append(key)
                    hyperparams["atom_class_map"][f"{res_name}_{atom_name}"] = atom_cls
                    hyperparams["atom_type_map"][f"{res_name}_{atom_name}"] = atom_type
                    charges.append(float(atom.attrib.get('charge', 0.0)))
                    sigmas.append(sigma * NM_TO_ANGSTROM)
                    epsilons.append(epsilon * KJ_TO_KCAL)
                    current_id += 1
            
            # Extract Residue Internal Bonds (Templates)
            res_bonds = []
            for bond in res.findall('Bond'):
                if 'from' in bond.attrib and 'to' in bond.attrib:
                    idx1 = int(bond.attrib['from'])
                    idx2 = int(bond.attrib['to'])
                    if idx1 < len(res_atom_names) and idx2 < len(res_atom_names):
                        name1 = res_atom_names[idx1]
                        name2 = res_atom_names[idx2]
                        res_bonds.append((name1, name2))
                elif 'atomName1' in bond.attrib and 'atomName2' in bond.attrib:
                    name1 = bond.attrib['atomName1']
                    name2 = bond.attrib['atomName2']
                    res_bonds.append((name1, name2))
            
            if res_bonds:
                hyperparams["residue_templates"][res_name] = res_bonds
        
        # Parse Standard Terms
        for b in root.findall('HarmonicBondForce/Bond'):
             c1 = b.attrib.get('class1', b.attrib.get('type1'))
             c2 = b.attrib.get('class2', b.attrib.get('type2'))
             if c1 and c2:
                 hyperparams["bonds"].append((c1, c2, float(b.attrib['length']), float(b.attrib['k'])))

        for a in root.findall('HarmonicAngleForce/Angle'):
            c1, c2, c3 = [a.attrib.get(f'class{i}') or a.attrib.get(f'type{i}') for i in range(1, 4)]
            if c1 and c2 and c3:
                hyperparams["angles"].append((c1, c2, c3, float(a.attrib['angle']), float(a.attrib['k'])))
        
        # Helper for torsions
        def parse_tor(tag_name):
            data = []
            for t in root.findall(f'PeriodicTorsionForce/{tag_name}'):
                classes = []
                for i in range(1, 5):
                    c = t.attrib.get(f'class{i}')
                    if c is None:
                        c = t.attrib.get(f'type{i}')
                    classes.append(c)
                
                if any(c is None for c in classes): continue
                terms = []
                for i in range(1, 7):
                    if f'periodicity{i}' in t.attrib:
                        terms.append((int(t.attrib[f'periodicity{i}']), float(t.attrib[f'phase{i}']), float(t.attrib[f'k{i}'])))
                if terms: data.append({'classes': tuple(classes), 'terms': terms})
            return data

        hyperparams['propers'] = parse_tor('Proper')
        hyperparams['impropers'] = parse_tor('Improper')

        # --- 2. CMAP Parsing ---
        cmap_grids = []
        for cmap_force in root.findall('CMAPTorsionForce'):
            # Parse Maps
            for map_tag in cmap_force.findall('Map'):
                energy_elem = map_tag.find('Energy')
                if energy_elem is not None:
                    energy_str = energy_elem.text.strip()
                else:
                    energy_str = map_tag.text.strip()
                    
                values = [float(x) for x in energy_str.split()]
                size = int(len(values)**0.5)
                if size * size != len(values):
                    print(f"  WARNING: Invalid CMAP grid size in {ff_name}, skipping map.")
                    continue
                    
                # Convert kJ/mol -> kcal/mol
                grid = jnp.array(values, dtype=jnp.float32).reshape(size, size) * KJ_TO_KCAL
                cmap_grids.append(grid)
            
            # Parse Torsions (Class mappings)
            for t in cmap_force.findall('Torsion'):
                classes = [t.attrib.get(f'class{i}') or t.attrib.get(f'type{i}') for i in range(1, 6)]
                if all(classes):
                    hyperparams["cmap_torsions"].append({
                        'classes': tuple(classes),
                        'map_index': int(t.attrib['map'])
                    })

        if cmap_grids:
            cmap_energy_grids = jnp.stack(cmap_grids)
        else:
            # Default empty grid if no CMAP
            cmap_energy_grids = jnp.zeros((0, 24, 24), dtype=jnp.float32)

        # Create and Save
        model = FullForceField(
            charges_by_id=jnp.asarray(charges),
            sigmas_by_id=jnp.asarray(sigmas),
            epsilons_by_id=jnp.asarray(epsilons),
            cmap_energy_grids=cmap_energy_grids,
            **hyperparams
        )
        
        # Save to disk
        output_path = os.path.join(output_dir, f"{ff_name}.eqx")
        save_force_field(output_path, model)
        print(f"✅ Saved {output_path}")

    except Exception as e:
        print(f"🛑 Error processing {ff_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("openmmforcefields"):
        print("Cloning repo...")
        os.system("git clone --depth 1 https://github.com/openmm/openmmforcefields.git")
    
    output_dir = "src/prxteinmpnn/physics/force_fields/eqx"
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursive search for ANY xml file
    xml_files = glob.glob("openmmforcefields/**/*.xml", recursive=True)
    print(f"Found {len(xml_files)} potential XML files.")
    
    for xml in xml_files:
        parse_xml_to_eqx(xml, output_dir)
