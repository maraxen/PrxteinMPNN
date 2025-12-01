import os
import sys
# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.convert_all_xmls import parse_xml_to_eqx

# Path to XML
xml_path = os.path.abspath("openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml")
output_dir = "src/prxteinmpnn/physics/force_fields/eqx"

if not os.path.exists(xml_path):
    print(f"Error: XML not found at {xml_path}")
    sys.exit(1)

print(f"Regenerating protein19SB.eqx from {xml_path}...")
parse_xml_to_eqx(xml_path, output_dir)
print("Done.")
