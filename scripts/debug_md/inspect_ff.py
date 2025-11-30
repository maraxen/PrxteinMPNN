import equinox as eqx
from prxteinmpnn.physics.force_fields import load_force_field

ff = load_force_field("src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx")
print(f"Residue Templates: {len(ff.residue_templates)}")
if "ALA" in ff.residue_templates:
    print(f"ALA bonds: {ff.residue_templates['ALA']}")
else:
    print("ALA template missing")
