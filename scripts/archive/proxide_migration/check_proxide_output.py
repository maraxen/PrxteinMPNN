import proxide
from proxide import parse_structure, OutputSpec
import os

print("Checking imports...")
try:
  from proxide.types import Protein

  print("Found proxide.types.Protein")
except ImportError:
  print("No proxide.types.Protein")

try:
  from proxide.protein import Protein

  print("Found proxide.protein.Protein")
except ImportError:
  print("No proxide.protein.Protein")

try:
  import proxide.constants

  print("Found proxide.constants")
  print(dir(proxide.constants))
except ImportError:
  print("No proxide.constants")

print("\nChecking parse_structure output...")
# create a dummy pdb file
pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   2.500   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -0.500   1.000  1.00  0.00           C
"""
with open("test.pdb", "w") as f:
  f.write(pdb_content)

try:
  spec = OutputSpec()
  result = parse_structure("test.pdb", spec)
  print("Result type:", type(result))
  if hasattr(result, "keys"):
    print("Keys:", list(result.keys()))
  else:
    print("Dir:", dir(result))

  if "aatype" in result:
    print("aatype shape:", result["aatype"].shape)

except Exception as e:
  print("Parse failed:", e)
finally:
  if os.path.exists("test.pdb"):
    os.remove("test.pdb")
