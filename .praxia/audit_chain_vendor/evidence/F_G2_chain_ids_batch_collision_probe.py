"""Differential probe: does proxide's batched Protein.chain_ids correctly track each
structure's OWN chain letters, or does it silently collapse to structure 0's vocabulary?

Discovered while scoping G2 (chains_to_design) in
.praxia/docs/specs/260826_chain-selection-gap-closure.md -- chains_to_design requires
resolving each residue's chain LETTER from (chain_index, chain_ids), so this is the
foundational data this feature would be built on.

Two arms:
  ARM 1 (heterogeneous chain COUNT):  two_chain_ab.pdb (chains A,B) + tests/data/1ubq.pdb
                                       (chain A only) -- expect a crash or a clean signal,
                                       not silent success.
  ARM 2 (same chain COUNT, different LETTERS): two_chain_ab.pdb (A,B) + two_chain_cd.pdb
                                       (C,D), both 2 chains -- this is the dangerous case:
                                       stacking could succeed while silently mislabeling.

Run: uv run python .praxia/audit_chain_vendor/evidence/F_G2_chain_ids_batch_collision_probe.py
"""

import json
from pathlib import Path

from proxide.ops.dataset import create_protein_dataset

HERE = Path(__file__).resolve().parent
AB = str(HERE / "F_G2_chain_ids_batch_two_chain_ab.pdb")
CD = str(HERE / "F_G2_chain_ids_batch_two_chain_cd.pdb")
SINGLE_CHAIN = str(HERE.parents[2] / "tests" / "data" / "1ubq.pdb")

report: dict[str, object] = {"probe": "F_G2_chain_ids_batch_collision", "arms": {}}

# ARM 1: heterogeneous chain count (2 chains vs 1 chain) in one batch.
try:
  ds = create_protein_dataset([AB, SINGLE_CHAIN], batch_size=2, parse_kwargs={})
  list(ds)
  report["arms"]["heterogeneous_chain_count"] = {"outcome": "no_error", "verdict": "UNEXPECTED-PASS"}
except ValueError as exc:
  report["arms"]["heterogeneous_chain_count"] = {
    "outcome": "ValueError",
    "message": str(exc),
    "verdict": "CRASHES-LOUDLY (at least not silent)",
  }

# ARM 2: same chain count (2 and 2), different letters -- the dangerous silent case.
ds = create_protein_dataset([AB, CD], batch_size=2, parse_kwargs={})
batches = list(ds)
p = batches[0]
chain_ids = list(p.chain_ids)
row0_chain0_expected = "A"  # AB.pdb's first chain
row1_chain0_expected = "C"  # CD.pdb's first chain, DIFFERENT from AB.pdb
row1_chain0_resolved = chain_ids[int(p.chain_index[1][0])]

report["arms"]["same_count_different_letters"] = {
  "batched_chain_ids": chain_ids,
  "row0_expected_first_chain_letter": row0_chain0_expected,
  "row1_expected_first_chain_letter": row1_chain0_expected,
  "row1_resolved_first_chain_letter": row1_chain0_resolved,
  "verdict": (
    "SILENT-CORRUPTION-CONFIRMED"
    if row1_chain0_resolved != row1_chain0_expected
    else "UNEXPECTED-CORRECT"
  ),
}

print(json.dumps(report, indent=2))
