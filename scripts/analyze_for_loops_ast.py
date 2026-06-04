#!/usr/bin/env python3
"""AST scan: list ``for``-loops in Aminx JAX model/sampling code.

Run from repo: ``python aminx/scripts/analyze_for_loops_ast.py``

Heuristic: loops are annotated as
  - static_unroll — ``for i in range(`` (compile-time length; often over layers / modules)
  - other — not matched; inspect manually
  - nested_function — body contains ``def ``
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoopInfo:
  file: Path
  lineno: int
  col_offset: int
  kind: str
  line_preview: str
  has_nested_def: bool


def _line_preview(path: Path, lineno: int) -> str:
  lines = path.read_text().splitlines()
  if 1 <= lineno <= len(lines):
    return lines[lineno - 1].strip()[: 120]
  return ""


def visit_file(path: Path) -> list[LoopInfo]:
  out: list[LoopInfo] = []
  try:
    tree = ast.parse(path.read_text(), filename=str(path))
  except SyntaxError as e:
    return [
      LoopInfo(path, e.lineno or 0, 0, "parse_error", str(e), False),
    ]

  for node in ast.walk(tree):
    if not isinstance(node, ast.For):
      continue
    it = node.iter
    k = "other"
    if (
      isinstance(it, ast.Call)
      and isinstance(it.func, ast.Name)
      and it.func.id == "range"
    ):
      k = "static_unroll_range"
    elif (
      isinstance(it, ast.Call)
      and isinstance(it.func, ast.Name)
      and it.func.id in ("enumerate", "zip", "reversed", "iter")
    ):
      k = f"for_{it.func.id}"
    elif isinstance(it, ast.Name):
      k = "for_iter_var"

    has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(node))
    out.append(
      LoopInfo(
        file=path,
        lineno=node.lineno,
        col_offset=node.col_offset,
        kind=k,
        line_preview=_line_preview(path, node.lineno or 0),
        has_nested_def=has_def,
      ),
    )
  return out


def main() -> int:
  here = Path(__file__).resolve()
  root = here.parent.parent / "src" / "aminx"
  if not root.is_dir():
    print("Expected", root, file=sys.stderr)
    return 1

  targets = {root / "model", root / "sampling", root / "utils"}
  files: list[Path] = []
  for t in targets:
    if t.is_dir():
      files.extend(sorted(t.rglob("*.py")))

  all_info: list[LoopInfo] = []
  for f in files:
    if "__pycache__" in f.parts or f.name == "analyze_for_loops_ast.py":
      continue
    all_info.extend(visit_file(f))

  by_kind: dict[str, list[LoopInfo]] = {}
  for li in all_info:
    by_kind.setdefault(li.kind, []).append(li)

  print("=== for-loops in aminx/model, sampling, utils (AST) ===\n")
  for kind in sorted(by_kind, key=str):
    items = by_kind[kind]
    print(f"## {kind} ({len(items)})\n")
    for li in items[: 200]:  # cap
      n = " * " if li.has_nested_def else "   "
      rel = li.file.relative_to(root.parent)
      print(f"  {n}{rel}:{li.lineno}  {li.line_preview}")
    if len(items) > 200:
      print(f"  ... and {len(items) - 200} more\n")
    print()
  print(
    "Notes:\n"
    "  - 'static_unroll_range' = ``for i in range(...)`` — usually OK for JAX; ``range``\n"
    "    is Python at trace time, so the loop unrolls. Replacing with lax.scan only helps\n"
    "    if the *body* is a single tracable function over a *stack* of parameters, not a\n"
    "    tuple of different modules indexed by i.\n"
    "  - 'for_iter_var' / 'for_in_name' = loop over a JAX array — often must become\n"
    "    lax.map / vmap / scan, or the iterator is a Python list (rare in hot path).\n",
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
