"""Emit the aminx control-knob matrix: spec field x entry point x hop.

Produced for task_id 260715_aminx-campaign-control-knob-audit.

Every cell is derived by introspecting the *installed* aminx (dataclass fields, Typer
command params) or by AST-parsing aminx source -- never by reading prose or trusting a
summary. The audit this feeds exists because seven control knobs were silently dropped
between their declaration site and the model, and every one was found incidentally.

Run under a venv with the audited aminx importable; the ref is reported in the output so
a matrix can never be silently attributed to the wrong version.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Entry points are DISCOVERED by walking the click tree, never hand-listed. An earlier
# version of this script hardcoded seven command paths and silently audited 3 of the 6
# campaign subcommands -- reproducing, in the audit tool itself, the exact hand-maintained-
# list-drifts-from-reality bug it exists to find. Enumerate, don't enumerate-by-hand.
MAX_DEPTH = 2


def _git_ref(repo: Path) -> dict[str, str]:
  """Record exactly which ref this matrix describes."""
  def _run(*args: str) -> str:
    try:
      return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
      ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
      return "unknown"

  return {"head": _run("rev-parse", "HEAD"), "describe": _run("describe", "--always", "--dirty")}


def _spec_fields() -> dict[str, dict[str, Any]]:
  from aminx.run.specs import RunSpecification, SamplingSpecification

  base = {f.name for f in fields(RunSpecification)}
  out: dict[str, dict[str, Any]] = {}
  for f in fields(SamplingSpecification):
    out[f.name] = {
      "type": str(f.type),
      "init": f.init,
      "origin": "RunSpecification" if f.name in base else "SamplingSpecification",
    }
  return out


def _cli_params() -> dict[str, set[str]]:
  """Map every discovered command to the spec-field names it can set.

  Uses click's own param objects rather than parsing help text, so a flag cannot appear
  present here unless it is genuinely registered on the command. Groups (`run`, `spec`,
  `campaign`) are reported too: their shared model flags live on the group callback, and a
  subcommand inherits them only when invoked as `aminx run --flag sample`.
  """
  import typer.main

  from aminx.cli import app

  out: dict[str, set[str]] = {}

  def walk(cmd: Any, path: list[str], depth: int) -> None:
    label = " ".join(path) if path else "<root>"
    if path:
      out[label] = {p.name for p in cmd.params if p.name}
    subs = getattr(cmd, "commands", None)
    if not subs or depth >= MAX_DEPTH:
      return
    for name in sorted(subs):
      walk(subs[name], [*path, name], depth + 1)

  walk(typer.main.get_command(app), [], 0)
  logger.info("discovered %d CLI commands: %s", len(out), ", ".join(sorted(out)))
  return out


def _dict_literal_keys(src: Path, target: str) -> set[str]:
  """AST-extract the keys of the `row["sampling_spec"] = {...}` literal.

  AST rather than regex: the point of this audit is that a hand-maintained literal drifts
  from the dataclass, so the extraction of that literal must itself be exact.
  """
  tree = ast.parse(src.read_text())
  found: set[str] = set()

  for node in ast.walk(tree):
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
      continue
    for tgt in node.targets:
      if (
        isinstance(tgt, ast.Subscript)
        and isinstance(tgt.slice, ast.Constant)
        and tgt.slice.value == target
      ):
        for k in node.value.keys:
          if isinstance(k, ast.Constant) and isinstance(k.value, str):
            found.add(k.value)
  return found


def _named_set_literal(src: Path, names: tuple[str, ...]) -> set[str]:
  """Extract string members of a module-level frozenset/set assignment."""
  tree = ast.parse(src.read_text())
  out: set[str] = set()
  for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
      continue
    tgt_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
    if not tgt_names & set(names):
      continue
    for sub in ast.walk(node.value):
      if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
        out.add(sub.value)
  return out


def _coercion_whitelist(src: Path) -> set[str]:
  """Extract the array-coercion field names from `_coerce_field_value`.

  These are the fields whose JSON lists become arrays on decode; a field absent here that
  holds an array arrives as a plain list, silently.
  """
  tree = ast.parse(src.read_text())
  out: set[str] = set()
  for node in ast.walk(tree):
    if not (isinstance(node, ast.FunctionDef) and node.name == "_coerce_field_value"):
      continue
    for sub in ast.walk(node):
      # match: field_name in {"a", "b", ...}
      if isinstance(sub, ast.Compare) and isinstance(sub.ops[0], ast.In):
        for cmp in sub.comparators:
          if isinstance(cmp, ast.Set):
            for elt in cmp.elts:
              if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.add(elt.value)
  return out


def _consumer_reads(src_root: Path, field_names: set[str]) -> dict[str, list[str]]:
  """Find which fields are read as attributes anywhere under host/.

  Heuristic and deliberately over-inclusive: it answers "is this field mentioned by any
  consumer at all", so a field with an empty list here is a strong silent-no-op candidate.
  It is a triage signal, never a verdict -- Phase 2's differential harness is the verdict.
  """
  hits: dict[str, set[str]] = {n: set() for n in field_names}
  for py in sorted((src_root / "host").rglob("*.py")):
    try:
      tree = ast.parse(py.read_text())
    except SyntaxError:
      logger.warning("skipping unparseable file: %s", py)
      continue
    rel = str(py.relative_to(src_root))
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute) and node.attr in hits:
        hits[node.attr].add(rel)
  return {k: sorted(v) for k, v in hits.items()}


def build_matrix(repo: Path) -> dict[str, Any]:
  src_root = repo / "src" / "aminx"
  campaign_py = src_root / "host" / "campaign.py"
  spec_json_py = src_root / "run" / "spec_json.py"

  spec = _spec_fields()
  cli = _cli_params()
  manifest_keys = _dict_literal_keys(campaign_py, "sampling_spec")
  non_json = _named_set_literal(spec_json_py, ("_NON_JSON_ROOT_FIELDS",))
  coercion = _coercion_whitelist(spec_json_py)
  reads = _consumer_reads(src_root, set(spec))

  rows = []
  for name, meta in sorted(spec.items()):
    row: dict[str, Any] = {
      "field": name,
      "origin": meta["origin"],
      "type": meta["type"],
      "in_manifest_dict": name in manifest_keys,
      "spec_json_encodable": name not in non_json and meta["init"],
      "array_coerced_on_decode": name in coercion,
      "read_under_host": reads.get(name, []),
    }
    for label in sorted(cli):
      row[f"cli::{label}"] = name in cli[label]
    rows.append(row)

  return {
    "task_id": "260715_aminx-campaign-control-knob-audit",
    "aminx_ref": _git_ref(repo),
    "cli_commands_discovered": sorted(cli),
    "field_count": len(rows),
    "manifest_dict_key_count": len(manifest_keys),
    # Keys written into the manifest that are not spec fields at all: these would raise
    # TypeError at SamplingSpecification(**payload) unless deprecated-stripped.
    "manifest_keys_not_spec_fields": sorted(manifest_keys - set(spec)),
    "rows": rows,
  }


def summarize(matrix: dict[str, Any]) -> str:
  rows = matrix["rows"]
  # Subcommands only -- the bare "campaign" group carries no spec flags of its own.
  campaign_cols = [c for c in matrix["cli_commands_discovered"] if c.startswith("campaign ")]
  reachable_via_plan = [r for r in rows if r.get("cli::campaign plan")]
  in_manifest = [r for r in rows if r["in_manifest_dict"]]
  # The core gap: a field that NO campaign entry point can set and that the manifest never
  # carries is unreachable in campaign mode, whatever its default claims.
  unreachable = [
    r for r in rows
    if not r["in_manifest_dict"]
    and not any(r.get(f"cli::{c}") for c in campaign_cols)
  ]
  never_read = [r for r in rows if not r["read_under_host"]]

  lines = [
    f"aminx ref:            {matrix['aminx_ref']['describe']} ({matrix['aminx_ref']['head'][:10]})",
    f"CLI commands found:   {len(matrix['cli_commands_discovered'])}"
    f" ({len(campaign_cols)} campaign: {', '.join(c.split(' ', 1)[1] for c in campaign_cols)})",
    f"spec fields:          {matrix['field_count']}",
    f"manifest dict keys:   {matrix['manifest_dict_key_count']}",
    f"settable via plan:    {len(reachable_via_plan)}",
    f"carried in manifest:  {len(in_manifest)}",
    "",
    f"UNREACHABLE in campaign mode (no campaign flag, not in manifest): {len(unreachable)}",
    *[f"    {r['field']}" for r in unreachable],
    "",
    f"NEVER read anywhere under host/: {len(never_read)}",
    *[f"    {r['field']}" for r in never_read],
  ]
  if matrix["manifest_keys_not_spec_fields"]:
    lines += ["", "manifest keys that are NOT spec fields:", *[f"    {k}" for k in matrix["manifest_keys_not_spec_fields"]]]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--repo", type=Path, default=Path(__file__).resolve().parents[2],
    help="aminx repo root to AST-parse (must match the importable aminx).",
  )
  parser.add_argument("--out", type=Path, default=None, help="Write full matrix JSON here.")
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )

  matrix = build_matrix(args.repo)
  print(summarize(matrix))

  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    logger.info("wrote matrix: %s", args.out)


if __name__ == "__main__":
  main()
