"""Read/write helper fields on pytest JSON reports (pytest-json-report format).

Usage:
  python .praxia/pytest_error_tool.py show PATH [NODEID]
  python .praxia/pytest_error_tool.py set PATH NODEID --strategy S [--diff D] [--reasoning R]
  python .praxia/pytest_error_tool.py list-failed PATH

Fields stored on each matching ``tests[]`` entry (for failed tests):
  resolution_strategy, diff, reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

META_FIELDS = frozenset({"resolution_strategy", "diff", "reasoning"})


def load_report(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as f:
    return json.load(f)


def save_report(path: Path, data: dict[str, Any]) -> None:
  with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")


def iter_failed_tests(report: dict[str, Any]) -> list[dict[str, Any]]:
  return [t for t in report.get("tests", []) if t.get("outcome") == "failed"]


def find_test(report: dict[str, Any], nodeid: str) -> dict[str, Any]:
  for t in report.get("tests", []):
    if t.get("nodeid") == nodeid:
      return t
  msg = f"No test with nodeid={nodeid!r}"
  raise KeyError(msg)


def set_failure_fields(
  report: dict[str, Any],
  *,
  nodeid: str,
  resolution_strategy: str | None = None,
  diff: str | None = None,
  reasoning: str | None = None,
) -> dict[str, Any]:
  t = find_test(report, nodeid)
  if t.get("outcome") != "failed":
    msg = f"Test {nodeid!r} is not failed (outcome={t.get('outcome')!r})"
    raise ValueError(msg)
  if resolution_strategy is not None:
    t["resolution_strategy"] = resolution_strategy
  if diff is not None:
    t["diff"] = diff
  if reasoning is not None:
    t["reasoning"] = reasoning
  return t


def get_failure_fields(test: dict[str, Any]) -> dict[str, Any]:
  return {k: test[k] for k in META_FIELDS if k in test}


def _cmd_show(path: Path, nodeid: str | None) -> int:
  report = load_report(path)
  if nodeid is None:
    for t in iter_failed_tests(report):
      print(t["nodeid"])
    return 0
  t = find_test(report, nodeid)
  print(json.dumps(t, indent=2, ensure_ascii=False))
  return 0


def _cmd_list_failed(path: Path) -> int:
  report = load_report(path)
  for t in iter_failed_tests(report):
    extra = get_failure_fields(t)
    print(t["nodeid"])
    if extra:
      print(json.dumps(extra, indent=2, ensure_ascii=False))
  return 0


def _cmd_set(
  path: Path,
  nodeid: str,
  *,
  strategy: str | None,
  diff: str | None,
  reasoning: str | None,
) -> int:
  report = load_report(path)
  set_failure_fields(
    report,
    nodeid=nodeid,
    resolution_strategy=strategy,
    diff=diff,
    reasoning=reasoning,
  )
  save_report(path, report)
  return 0


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Annotate pytest-json-report output.")
  sub = p.add_subparsers(dest="cmd", required=True)

  p_show = sub.add_parser("show", help="List failed nodeids or dump one test record")
  p_show.add_argument("path", type=Path)
  p_show.add_argument("nodeid", nargs="?", default=None)
  p_show.set_defaults(func=lambda a: _cmd_show(a.path, a.nodeid))

  p_list = sub.add_parser("list-failed", help="Print failed tests and any annotation fields")
  p_list.add_argument("path", type=Path)
  p_list.set_defaults(func=lambda a: _cmd_list_failed(a.path))

  p_set = sub.add_parser("set", help="Set annotation fields on a failed test (saves file)")
  p_set.add_argument("path", type=Path)
  p_set.add_argument("nodeid")
  p_set.add_argument("--strategy", dest="strategy", default=None)
  p_set.add_argument("--diff", dest="diff", default=None)
  p_set.add_argument("--reasoning", dest="reasoning", default=None)
  p_set.set_defaults(
    func=lambda a: _cmd_set(a.path, a.nodeid, strategy=a.strategy, diff=a.diff, reasoning=a.reasoning),
  )

  args = p.parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  sys.exit(main())
