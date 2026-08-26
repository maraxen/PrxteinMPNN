"""Axis B (task 260826_chain-selection-vendor-superset-audit): vendor superset checker.

AST-extracts LigandMPNN's argparse flag surface and aminx's RunSpecification-family field
surface, applies a hand-curated alias map, and produces a coverage matrix. This answers "does
the concept exist in aminx at all" -- a structural/surface question. Whether a present field is
actually WIRED into every runner surface is Axis A's question (differential bit-identity probes,
see F_A1/F_A2 probes) -- deliberately NOT re-derived here to keep the two axes independent.

Run with:
    uv run python .praxia/audit_chain_vendor/checkers/vendor_superset_checker.py \
        --vendor-run-py /path/to/LigandMPNN/run.py \
        --aminx-specs-py src/aminx/run/specs.py
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

# Hand-curated: vendor argparse flag -> {aminx_fields, category, notes}. Only chain-selection
# relevant flags are covered -- this is not a full-surface audit (see vendor_flags.json's note
# for the excluded categories: bias_AA*, omit_AA*, transmembrane_*, pack_side_chains family,
# model_type/checkpoint_* family).
ALIAS_MAP: dict[str, dict] = {
    "chains_to_design": {
        "aminx_fields": [],
        "category": "chain-level design/fix split",
        "notes": (
            "No aminx field expresses this directly. chain_mask_fixed (orphaned PR #1881, "
            "never merged) is a raw residue-index array, not a chain-letter selector -- a "
            "caller still has to hand-build the mapping themselves."
        ),
    },
    "parse_these_chains_only": {
        "aminx_fields": ["chain_id"],
        "category": "structure-level chain filter",
        "notes": "aminx's chain_id (RunSpecification base) is consumed at host/prep.py:96, upstream of all runner surfaces.",
    },
    "fixed_residues": {
        "aminx_fields": ["fixed_mask"],
        "category": "residue-level fix",
        "notes": "fixed_mask is the aminx equivalent; residue letters vs a boolean array is an ergonomics difference, not a coverage gap.",
    },
    "fixed_residues_multi": {
        "aminx_fields": ["fixed_mask"],
        "category": "residue-level fix (batch)",
        "notes": "aminx's fixed_mask is per-call, not a json-mapped-by-pdb-path batch structure -- a real ergonomics gap for multi-structure batches, distinct from the single-structure case.",
    },
    "redesigned_residues": {
        "aminx_fields": ["fixed_mask"],
        "category": "residue-level allowlist (inverse framing)",
        "notes": "Covered by fixed_mask's complement; no dedicated 'redesign-only-these' field, but semantically equivalent -- not a real gap.",
    },
    "redesigned_residues_multi": {
        "aminx_fields": ["fixed_mask"],
        "category": "residue-level allowlist (batch)",
        "notes": "Same batch-ergonomics gap as fixed_residues_multi.",
    },
    "symmetry_residues": {
        "aminx_fields": ["tie_group_map", "tied_positions"],
        "category": "cross-chain symmetric-tying",
        "notes": "Conceptual match via specs.py's 'tied-position logit averaging' family -- reachability across all 4 runner surfaces NOT YET traced (deferred from Phase 1 seed_findings.md).",
    },
    "symmetry_weights": {
        "aminx_fields": [],
        "category": "cross-chain symmetric-tying (per-tie-group weighting)",
        "notes": "No aminx field found for per-symmetry-group WEIGHT (as opposed to just which positions are tied) -- candidate real gap, needs confirmation once tie_group_map's actual semantics are traced.",
    },
    "homo_oligomer": {
        "aminx_fields": [],
        "category": "cross-chain symmetric-tying (convenience preset)",
        "notes": "No aminx convenience preset found; would need to be hand-constructed via tie_group_map if that field supports the same shape at all.",
    },
}


@dataclass
class VendorFlag:
    flag: str
    type_: str | None
    default: str | None
    help_: str | None
    source_line: int


@dataclass
class AminxField:
    name: str
    declaring_class: str
    annotation: str | None
    default: str | None
    source_line: int


def extract_vendor_flags(run_py_path: Path) -> list[VendorFlag]:
    """AST-walk run.py for every `<argparser-like>.add_argument(...)` call."""
    tree = ast.parse(run_py_path.read_text(encoding="utf-8"), filename=str(run_py_path))
    flags: list[VendorFlag] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        first_arg_value = node.args[0].value
        if not isinstance(first_arg_value, str):
            continue
        flag_name = first_arg_value.lstrip("-")
        type_txt = default_txt = help_txt = None
        for kw in node.keywords:
            if kw.arg == "type" and isinstance(kw.value, ast.Name):
                type_txt = kw.value.id
            elif kw.arg == "default":
                try:
                    default_txt = ast.unparse(kw.value)
                except Exception:  # noqa: BLE001
                    default_txt = "<unparseable>"
            elif kw.arg == "help" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                help_txt = kw.value.value
        flags.append(
            VendorFlag(
                flag=flag_name, type_=type_txt, default=default_txt, help_=help_txt,
                source_line=node.lineno,
            ),
        )
    return flags


def extract_aminx_fields(specs_py_path: Path) -> list[AminxField]:
    """AST-walk specs.py for dataclass field declarations on RunSpecification and subclasses."""
    tree = ast.parse(specs_py_path.read_text(encoding="utf-8"), filename=str(specs_py_path))
    fields: list[AminxField] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Heuristic: only classes that look like RunSpecification or a subclass thereof --
        # i.e. named *Specification, or explicitly named RunSpecification/RunSpec.
        if not (node.name.endswith("Specification") or node.name in {"RunSpecification", "RunSpec"}):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                try:
                    annotation_txt = ast.unparse(item.annotation)
                except Exception:  # noqa: BLE001
                    annotation_txt = "<unparseable>"
                default_txt = None
                if item.value is not None:
                    try:
                        default_txt = ast.unparse(item.value)
                    except Exception:  # noqa: BLE001
                        default_txt = "<unparseable>"
                fields.append(
                    AminxField(
                        name=item.target.id, declaring_class=node.name,
                        annotation=annotation_txt, default=default_txt,
                        source_line=item.lineno,
                    ),
                )
    return fields


def build_coverage_matrix(vendor_flags: list[VendorFlag], aminx_fields: list[AminxField]) -> dict:
    aminx_field_names = {f.name for f in aminx_fields}
    matrix = {"present": [], "absent": [], "unmapped_vendor_flag": []}
    for vf in vendor_flags:
        alias = ALIAS_MAP.get(vf.flag)
        if alias is None:
            matrix["unmapped_vendor_flag"].append(vf.flag)
            continue
        mapped_fields = alias["aminx_fields"]
        present_fields = [f for f in mapped_fields if f in aminx_field_names]
        entry = {
            "vendor_flag": vf.flag,
            "vendor_help": vf.help_,
            "vendor_source_line": vf.source_line,
            "category": alias["category"],
            "expected_aminx_fields": mapped_fields,
            "present_aminx_fields": present_fields,
            "notes": alias["notes"],
        }
        if mapped_fields and present_fields:
            matrix["present"].append(entry)
        else:
            matrix["absent"].append(entry)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-run-py", type=Path, required=True)
    parser.add_argument("--aminx-specs-py", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        default=Path(".praxia/audit_chain_vendor/vendor_superset_coverage.json"),
    )
    parser.add_argument(
        "--vendor-commit", type=str, default=None,
        help="Vendor repo commit SHA to stamp into the report for reproducibility "
        "(the vendor_run_py path itself is often an ephemeral clone location).",
    )
    args = parser.parse_args()

    vendor_flags = extract_vendor_flags(args.vendor_run_py)
    aminx_fields = extract_aminx_fields(args.aminx_specs_py)
    matrix = build_coverage_matrix(vendor_flags, aminx_fields)

    report = {
        "vendor_run_py": str(args.vendor_run_py),
        "vendor_commit": args.vendor_commit,
        "aminx_specs_py": str(args.aminx_specs_py),
        "vendor_flags_extracted_total": len(vendor_flags),
        "aminx_fields_extracted_total": len(aminx_fields),
        "chain_selection_flags_checked": len(ALIAS_MAP),
        "coverage": matrix,
        "summary": {
            "present": len(matrix["present"]),
            "absent": len(matrix["absent"]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))  # noqa: T201 -- CLI tool, this is the actual output


if __name__ == "__main__":
    main()
