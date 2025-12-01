import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import force_fields

FF_PATH = "src/prxteinmpnn/physics/force_fields/eqx/amber14-all.eqx"

try:
    ff = force_fields.load_force_field(FF_PATH)
    print(f"Loaded {FF_PATH}")
    print(f"Residues in atom_key_to_id: {len(set(r for r, a in ff.atom_key_to_id.keys()))}")
    print(f"Sample Residues: {list(set(r for r, a in ff.atom_key_to_id.keys()))[:10]}")
    
    if hasattr(ff, 'residue_templates'):
        print(f"Residue Templates: {len(ff.residue_templates)}")
        print(f"Sample Templates: {list(ff.residue_templates.keys())[:5]}")
    else:
        print("No residue_templates found.")
        
    print(f"Propers: {len(ff.propers)}")
    print(f"Impropers: {len(ff.impropers)}")
    
except Exception as e:
    print(f"Failed to load: {e}")
