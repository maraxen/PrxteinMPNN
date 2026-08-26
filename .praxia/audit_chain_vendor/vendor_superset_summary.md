# Axis B round 1 -- vendor superset coverage matrix

Formalizes the Phase 1 manual read into a reproducible AST-based checker
(`checkers/vendor_superset_checker.py`). Extracted programmatically, not assumed:

- 49 argparse flags from `dauparas/LigandMPNN@26ec57ac976ade5379920dbd43c7f97a91cf82de`'s `run.py`
- 124 dataclass fields across every `*Specification` class in `src/aminx/run/specs.py`
- Checked against a 9-entry hand-curated chain-selection alias map (`ALIAS_MAP` in the checker)

Full machine-readable output: `vendor_superset_coverage.json`.

## Present (6/9) -- aminx has a mapped field

| vendor flag | aminx field(s) | note |
|---|---|---|
| `fixed_residues` | `fixed_mask` | direct equivalent |
| `fixed_residues_multi` | `fixed_mask` | **batch-ergonomics gap**: aminx's field is per-call, not a json-mapped-by-pdb-path batch structure |
| `redesigned_residues` | `fixed_mask` | covered by complement, not a real gap |
| `redesigned_residues_multi` | `fixed_mask` | same batch-ergonomics gap as `fixed_residues_multi` |
| `symmetry_residues` | `tie_group_map`, `tied_positions` | conceptual match only -- reachability across all 4 runner surfaces still untraced |
| `parse_these_chains_only` | `chain_id` | consumed at `host/prep.py:96`, upstream of every runner surface |

## Absent (3/9) -- candidate real gaps

| vendor flag | vendor semantics | status |
|---|---|---|
| `chains_to_design` | chain-letter-level design/fix split, e.g. `"A,B,C,F"` | **no aminx field at all** -- orphaned `chain_mask_fixed` (#1881) is residue-index-only |
| `symmetry_weights` | per-tie-group weighting for symmetric design | no aminx field found -- needs confirming once `tie_group_map`'s real semantics are traced |
| `homo_oligomer` | convenience preset that auto-sets `symmetry_residues`+`symmetry_weights` | no aminx convenience preset |

## Deliberately out of scope for this axis

37 vendor flags are unmapped (checkpoint/model-variant selection, bias/omit-AA biasing,
transmembrane-model labels, side-chain packing family, I/O plumbing) -- not chain-selection
related, not audited here. Full list in `vendor_superset_coverage.json`'s
`coverage.unmapped_vendor_flag`.

## What this axis does NOT answer

Whether a *present* field (`fixed_mask`, `chain_id`, `tie_group_map`) is actually wired into
every runner surface is Axis A's question, answered by differential bit-identity probes, not by
AST field-presence. See `findings.jsonl` (FA1 PASS, FA2/FA3 FAIL-then-fixed) and
`seed_findings.md`'s still-untraced `tie_group_map` reachability question.

## Next steps (not done here)

1. Trace `tie_group_map`/`tied_positions`/`pass_mode` reachability across sample/score/jacobian/
   inspect the same way FA1-FA3 did for `fixed_mask` -- needed to confirm the `symmetry_residues`
   "present" mapping is actually meaningful, not just field-exists.
2. Owner decision: is `chains_to_design`'s convenience worth adding as a real aminx feature
   (chain-letter -> residue-index resolution helper), or is "construct fixed_mask yourself" an
   acceptable permanent answer?
3. `fixed_residues_multi`/`redesigned_residues_multi`'s batch-ergonomics gap -- confirm whether
   any existing aminx multi-structure batch path already covers this before treating it as a gap.
