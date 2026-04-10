"""Generate parity graphics and Markdown/HTML reports from manifests and evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


@dataclass(frozen=True, slots=True)
class TierResult:
  """JUnit summary for one parity tier."""

  name: str
  tests: int
  passed: int
  failed: int
  skipped: int
  missing: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
  """Aggregated parity evidence metrics for one path/condition lane."""

  path_id: str
  case_count: int
  pearson_mean: float | None
  pearson_min: float | None
  pearson_pass_rate: float | None
  mae_mean: float | None
  rmse_mean: float | None
  token_agreement_mean: float | None
  condition: str | None = None


@dataclass(frozen=True, slots=True)
class PathPearsonPoint:
  """Per-case Pearson point used for swarm plotting."""

  path_id: str
  case_id: str
  case_kind: str
  seed: int
  value: float
  condition: str | None = None


@dataclass(frozen=True, slots=True)
class CaseInventoryEntry:
  """Case-level metadata extracted from parity evidence rows."""

  case_id: str
  case_kind: str
  sequence_lengths: tuple[int, ...]
  seeds: tuple[int, ...]
  backbone_ids: tuple[str, ...]
  checkpoint_ids: tuple[str, ...]
  tiers: tuple[str, ...]
  path_count: int


@dataclass(frozen=True, slots=True)
class IntrinsicSummary:
  """Baseline-aware intrinsic-noise summary for one parity path/condition lane."""

  path_id: str
  condition: str | None
  case_count: int
  ratio_mean: float | None
  ratio_max: float | None
  pass_95_rate: float | None
  pass_99_rate: float | None


@dataclass(frozen=True, slots=True)
class MacroSummary:
  """Macro-distribution parity summary for one path/condition pair."""

  path_id: str
  condition: str
  case_count: int
  identity_wasserstein_mean: float | None
  entropy_wasserstein_mean: float | None
  composition_js_mean: float | None
  identity_ks_pvalue_median: float | None
  note: str | None = None


@dataclass(frozen=True, slots=True)
class SidechainGateSummary:
  """Aggregated side-chain-conditioned integration-gate status."""

  path_id: str
  condition: str
  case_count: int
  pass_count: int
  warn_count: int
  fail_count: int
  status: str
  reason: str


@dataclass(frozen=True, slots=True)
class CoverageEntry:
  """Coverage/exclusion annotation for one parity path."""

  path_id: str
  tier: str
  status: str
  reason: str


@dataclass(frozen=True, slots=True)
class TiedLaneDescriptor:
  """One configured tied/multistate apples-to-apples comparison lane."""

  condition: str
  comparison_api: str
  reference_combiner: str
  jax_multi_state_strategy: str
  token_comparison_enabled: bool
  is_primary: bool


@dataclass(frozen=True, slots=True)
class ExecutiveSnapshot:
  """Top-level parity status summary rendered near the report header."""

  status: str
  outcomes: tuple[str, ...]
  caveats: tuple[str, ...]


def _parse_junit(path: Path, tier_name: str) -> TierResult:
  """Parse a JUnit XML file into pass/fail/skip counts."""
  if not path.exists():
    return TierResult(
      name=tier_name,
      tests=0,
      passed=0,
      failed=0,
      skipped=0,
      missing=True,
    )

  root = ElementTree.fromstring(path.read_text(encoding="utf-8"))  # noqa: S314
  if root.tag == "testsuite":
    suites = [root]
  elif root.tag == "testsuites":
    suites = list(root.findall("testsuite"))
  else:
    suites = []

  tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
  failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
  errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
  skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
  failed = failures + errors
  passed = max(tests - failed - skipped, 0)
  return TierResult(
    name=tier_name,
    tests=tests,
    passed=passed,
    failed=failed,
    skipped=skipped,
    missing=False,
  )


def _write_checkpoint_checksums(
  *,
  project_root: Path,
  assets: list[dict[str, object]],
  output_path: Path,
) -> tuple[int, int]:
  """Write converted-checkpoint checksums and return (available, expected)."""
  lines: list[str] = []
  expected = 0
  available = 0
  for asset in assets:
    if asset.get("asset_kind") != "converted_checkpoint":
      continue
    required_for = asset.get("required_for")
    if not isinstance(required_for, list) or "parity_audit" not in required_for:
      continue
    rel_path = asset.get("path")
    asset_id = asset.get("id", "unknown")
    if not isinstance(rel_path, str):
      continue
    expected += 1
    target = project_root / rel_path
    if not target.exists():
      lines.append(f"MISSING  {rel_path}  ({asset_id})")
      continue
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    lines.append(f"{digest}  {rel_path}  ({asset_id})")
    available += 1

  output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
  return available, expected


def _load_evidence_metrics(path: Path) -> list[dict[str, Any]]:
  """Load evidence metric rows from CSV."""
  if not path.exists():
    return []
  rows: list[dict[str, Any]] = []
  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      row["seed"] = int(row.get("seed", "0"))
      row["sequence_length"] = int(row.get("sequence_length", "0"))
      row["metric_value"] = float(row.get("metric_value", "0.0"))
      threshold = row.get("threshold")
      row["threshold"] = float(threshold) if threshold not in (None, "") else None
      row["metric_group"] = row.get("metric_group") or None
      row["condition"] = row.get("condition") or None
      row["note"] = row.get("note") or None
      passed = row.get("passed")
      if passed in ("True", "true", "1"):
        row["passed"] = True
      elif passed in ("False", "false", "0"):
        row["passed"] = False
      else:
        row["passed"] = None
      rows.append(row)
  return rows


def _load_evidence_points(path: Path) -> list[dict[str, Any]]:
  """Load evidence point rows from CSV."""
  if not path.exists():
    return []
  rows: list[dict[str, Any]] = []
  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      row["seed"] = int(row.get("seed", "0"))
      row["sequence_length"] = int(row.get("sequence_length", "0"))
      row["reference_value"] = float(row.get("reference_value", "0.0"))
      row["observed_value"] = float(row.get("observed_value", "0.0"))
      row["point_kind"] = row.get("point_kind") or None
      row["condition"] = row.get("condition") or None
      rows.append(row)
  return rows


def _extract_tied_lane_descriptors(matrix_paths: list[dict[str, Any]]) -> list[TiedLaneDescriptor]:
  """Extract tied/multistate lane metadata from the parity matrix manifest."""
  tied_path = next(
    (
      path
      for path in matrix_paths
      if isinstance(path, dict) and path.get("id") == "tied-positions-and-multi-state"
    ),
    None,
  )
  if tied_path is None:
    return []

  acceptance = tied_path.get("acceptance")
  if not isinstance(acceptance, dict):
    msg = "tied-positions-and-multi-state acceptance payload must be an object."
    raise TypeError(msg)
  lane_payloads = acceptance.get("comparison_lanes")
  if lane_payloads is None:
    return []
  if not isinstance(lane_payloads, list):
    msg = "tied-positions-and-multi-state.acceptance.comparison_lanes must be a list."
    raise TypeError(msg)

  lanes: list[TiedLaneDescriptor] = []
  seen_conditions: set[str] = set()
  for payload in lane_payloads:
    if not isinstance(payload, dict):
      msg = "Each tied comparison lane must be an object."
      raise TypeError(msg)
    condition = payload.get("condition")
    if not isinstance(condition, str) or not condition:
      msg = "Each tied comparison lane must define a non-empty condition."
      raise TypeError(msg)
    if condition in seen_conditions:
      msg = f"Duplicate tied comparison condition {condition!r} in parity matrix."
      raise ValueError(msg)
    seen_conditions.add(condition)

    comparison_api = payload.get("comparison_api", "sampling")
    if comparison_api not in {"sampling", "scoring"}:
      msg = f"{condition}: invalid comparison_api {comparison_api!r}."
      raise TypeError(msg)
    reference_combiner = payload.get("reference_combiner")
    if reference_combiner not in {"weighted_sum", "arithmetic_mean", "geometric_mean"}:
      msg = f"{condition}: invalid reference_combiner {reference_combiner!r}."
      raise TypeError(msg)
    jax_strategy = payload.get("jax_multi_state_strategy")
    if jax_strategy not in {"arithmetic_mean", "geometric_mean", "product"}:
      msg = f"{condition}: invalid jax_multi_state_strategy {jax_strategy!r}."
      raise TypeError(msg)
    token_comparison = payload.get("token_comparison")
    if token_comparison not in {"enabled", "disabled"}:
      msg = f"{condition}: token_comparison must be 'enabled' or 'disabled'."
      raise TypeError(msg)
    is_primary = payload.get("is_primary")
    if not isinstance(is_primary, bool):
      msg = f"{condition}: is_primary must be boolean."
      raise TypeError(msg)

    lanes.append(
      TiedLaneDescriptor(
        condition=condition,
        comparison_api=str(comparison_api),
        reference_combiner=str(reference_combiner),
        jax_multi_state_strategy=str(jax_strategy),
        token_comparison_enabled=token_comparison == "enabled",
        is_primary=is_primary,
      ),
    )

  if sum(int(lane.is_primary) for lane in lanes) != 1:
    msg = "Exactly one tied/multistate comparison lane must be marked primary."
    raise ValueError(msg)

  return sorted(lanes, key=lambda lane: (not lane.is_primary, lane.condition))


def _aggregate_evidence(
  matrix_paths: list[dict[str, Any]],
  metric_rows: list[dict[str, Any]],
) -> list[EvidenceSummary]:
  """Aggregate path-level evidence summaries from scalar metric rows."""
  by_path: dict[str, list[dict[str, Any]]] = {}
  for row in metric_rows:
    path_id = row.get("path_id")
    if isinstance(path_id, str):
      by_path.setdefault(path_id, []).append(row)

  summaries: list[EvidenceSummary] = []
  for item in matrix_paths:
    path_id = item.get("id")
    if not isinstance(path_id, str):
      continue
    path_rows = by_path.get(path_id, [])
    if not path_rows:
      summaries.append(
        EvidenceSummary(
          path_id=path_id,
          case_count=0,
          pearson_mean=None,
          pearson_min=None,
          pearson_pass_rate=None,
          mae_mean=None,
          rmse_mean=None,
          token_agreement_mean=None,
          condition=None,
        ),
      )
      continue

    by_condition: dict[str | None, list[dict[str, Any]]] = {}
    for row in path_rows:
      condition = row.get("condition")
      condition_key = condition if isinstance(condition, str) else None
      by_condition.setdefault(condition_key, []).append(row)

    for condition, rows in sorted(by_condition.items(), key=lambda item: (item[0] is not None, item[0] or "")):
      case_count = len({(row["case_id"], row["case_kind"]) for row in rows}) if rows else 0
      pearsons = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "pearson_correlation"]
      pearson_passes = [row.get("passed") for row in rows if row.get("metric_name") == "pearson_correlation"]
      maes = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "mae"]
      rmses = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "rmse"]
      token_agreements = [
        float(row["metric_value"])
        for row in rows
        if row.get("metric_name") == "token_agreement"
      ]

      pass_values = [value for value in pearson_passes if isinstance(value, bool)]
      pass_rate = float(np.mean(pass_values)) if pass_values else None
      summaries.append(
        EvidenceSummary(
          path_id=path_id,
          case_count=case_count,
          pearson_mean=float(np.mean(pearsons)) if pearsons else None,
          pearson_min=float(np.min(pearsons)) if pearsons else None,
          pearson_pass_rate=pass_rate,
          mae_mean=float(np.mean(maes)) if maes else None,
          rmse_mean=float(np.mean(rmses)) if rmses else None,
          token_agreement_mean=float(np.mean(token_agreements)) if token_agreements else None,
          condition=condition,
        ),
      )
  return summaries


def _collect_case_pearson_points(metric_rows: list[dict[str, Any]]) -> list[PathPearsonPoint]:
  """Collect one Pearson value per path/case/seed tuple."""
  grouped: dict[tuple[str, str | None, str, str, int], list[float]] = {}
  for row in metric_rows:
    if row.get("metric_name") != "pearson_correlation":
      continue
    path_id = row.get("path_id")
    condition = row.get("condition")
    case_id = row.get("case_id")
    case_kind = row.get("case_kind")
    seed = row.get("seed")
    if (
      not isinstance(path_id, str)
      or not isinstance(case_id, str)
      or not isinstance(case_kind, str)
      or not isinstance(seed, int)
    ):
      continue
    condition_key = condition if isinstance(condition, str) else None
    key = (path_id, condition_key, case_id, case_kind, seed)
    grouped.setdefault(key, []).append(float(row["metric_value"]))

  points: list[PathPearsonPoint] = []
  for (path_id, condition, case_id, case_kind, seed), values in sorted(grouped.items()):
    points.append(
      PathPearsonPoint(
        path_id=path_id,
        case_id=case_id,
        case_kind=case_kind,
        seed=seed,
        value=float(np.mean(values)),
        condition=condition,
      ),
    )
  return points


def _aggregate_case_inventory(
  metric_rows: list[dict[str, Any]],
  point_rows: list[dict[str, Any]],
) -> list[CaseInventoryEntry]:
  """Aggregate evaluated case metadata across metric and point evidence rows."""
  grouped: dict[tuple[str, str], dict[str, set[int] | set[str]]] = {}
  for row in [*metric_rows, *point_rows]:
    case_id = row.get("case_id")
    case_kind = row.get("case_kind")
    if not isinstance(case_id, str) or not isinstance(case_kind, str):
      continue
    key = (case_id, case_kind)
    if key not in grouped:
      grouped[key] = {
        "sequence_lengths": set(),
        "seeds": set(),
        "backbone_ids": set(),
        "checkpoint_ids": set(),
        "tiers": set(),
        "path_ids": set(),
      }
    entry = grouped[key]

    sequence_length = row.get("sequence_length")
    if isinstance(sequence_length, int) and sequence_length > 0:
      length_values = entry["sequence_lengths"]
      if isinstance(length_values, set):
        length_values.add(sequence_length)

    seed = row.get("seed")
    if isinstance(seed, int) and seed > 0:
      seed_values = entry["seeds"]
      if isinstance(seed_values, set):
        seed_values.add(seed)

    backbone_id = row.get("backbone_id")
    if isinstance(backbone_id, str) and backbone_id:
      backbone_values = entry["backbone_ids"]
      if isinstance(backbone_values, set):
        backbone_values.add(backbone_id)

    checkpoint_id = row.get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id:
      checkpoint_values = entry["checkpoint_ids"]
      if isinstance(checkpoint_values, set):
        checkpoint_values.add(checkpoint_id)

    tier = row.get("tier")
    if isinstance(tier, str) and tier:
      tier_values = entry["tiers"]
      if isinstance(tier_values, set):
        tier_values.add(tier)

    path_id = row.get("path_id")
    if isinstance(path_id, str) and path_id:
      path_values = entry["path_ids"]
      if isinstance(path_values, set):
        path_values.add(path_id)

  inventory_entries: list[CaseInventoryEntry] = []
  for (case_id, case_kind), entry in sorted(grouped.items()):
    sequence_lengths = tuple(sorted(int(value) for value in entry["sequence_lengths"]))
    seeds = tuple(sorted(int(value) for value in entry["seeds"]))
    backbone_ids = tuple(sorted(str(value) for value in entry["backbone_ids"]))
    checkpoint_ids = tuple(sorted(str(value) for value in entry["checkpoint_ids"]))
    tiers = tuple(sorted(str(value) for value in entry["tiers"]))
    path_count = len(entry["path_ids"])
    inventory_entries.append(
      CaseInventoryEntry(
        case_id=case_id,
        case_kind=case_kind,
        sequence_lengths=sequence_lengths,
        seeds=seeds,
        backbone_ids=backbone_ids,
        checkpoint_ids=checkpoint_ids,
        tiers=tiers,
        path_count=path_count,
      ),
    )
  return inventory_entries


def _aggregate_intrinsic(metric_rows: list[dict[str, Any]]) -> list[IntrinsicSummary]:
  by_lane: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
  for row in metric_rows:
    if row.get("metric_group") != "intrinsic":
      continue
    path_id = row.get("path_id")
    if not isinstance(path_id, str):
      continue
    condition = row.get("condition")
    condition_key = condition if isinstance(condition, str) else None
    by_lane.setdefault((path_id, condition_key), []).append(row)

  summaries: list[IntrinsicSummary] = []
  for (path_id, condition), rows in sorted(by_lane.items()):
    case_count = len({(row.get("case_id"), row.get("case_kind")) for row in rows})
    ratios = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "parity_to_intrinsic_ratio"]
    pass_95 = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "intrinsic_pass_95"]
    pass_99 = [float(row["metric_value"]) for row in rows if row.get("metric_name") == "intrinsic_pass_99"]
    summaries.append(
      IntrinsicSummary(
        path_id=path_id,
        condition=condition,
        case_count=case_count,
        ratio_mean=float(np.mean(ratios)) if ratios else None,
        ratio_max=float(np.max(ratios)) if ratios else None,
        pass_95_rate=float(np.mean(pass_95)) if pass_95 else None,
        pass_99_rate=float(np.mean(pass_99)) if pass_99 else None,
      ),
    )
  return summaries


def _aggregate_macro(metric_rows: list[dict[str, Any]]) -> list[MacroSummary]:
  by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in metric_rows:
    if row.get("metric_group") != "macro":
      continue
    path_id = row.get("path_id")
    condition = row.get("condition")
    if not isinstance(path_id, str) or not isinstance(condition, str):
      continue
    by_key.setdefault((path_id, condition), []).append(row)

  summaries: list[MacroSummary] = []
  for (path_id, condition), rows in sorted(by_key.items()):
    case_count = len({(row.get("case_id"), row.get("case_kind")) for row in rows})
    identity_w = [
      float(row["metric_value"])
      for row in rows
      if row.get("metric_name") == "macro_identity_wasserstein"
    ]
    entropy_w = [
      float(row["metric_value"])
      for row in rows
      if row.get("metric_name") == "macro_entropy_wasserstein"
    ]
    composition_js = [
      float(row["metric_value"])
      for row in rows
      if row.get("metric_name") == "macro_composition_js_distance"
    ]
    identity_p = [
      float(row["metric_value"])
      for row in rows
      if row.get("metric_name") == "macro_identity_ks_pvalue"
    ]
    note = next((str(row["note"]) for row in rows if isinstance(row.get("note"), str) and row.get("note")), None)
    summaries.append(
      MacroSummary(
        path_id=path_id,
        condition=condition,
        case_count=case_count,
        identity_wasserstein_mean=float(np.mean(identity_w)) if identity_w else None,
        entropy_wasserstein_mean=float(np.mean(entropy_w)) if entropy_w else None,
        composition_js_mean=float(np.mean(composition_js)) if composition_js else None,
        identity_ks_pvalue_median=float(np.median(identity_p)) if identity_p else None,
        note=note,
      ),
    )
  return summaries


def _decode_sidechain_gate_status(value: float) -> str:
  rounded = int(round(value))
  if rounded <= 0:
    return "pass"
  if rounded == 1:
    return "warn"
  return "fail"


def _aggregate_sidechain_gate(metric_rows: list[dict[str, Any]]) -> list[SidechainGateSummary]:
  by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in metric_rows:
    if row.get("metric_group") != "macro":
      continue
    if row.get("condition") != "side_chain_conditioned":
      continue
    if row.get("metric_name") != "macro_sidechain_conditioned_gate_status_code":
      continue
    path_id = row.get("path_id")
    condition = row.get("condition")
    if not isinstance(path_id, str) or not isinstance(condition, str):
      continue
    by_key.setdefault((path_id, condition), []).append(row)

  summaries: list[SidechainGateSummary] = []
  for (path_id, condition), rows in sorted(by_key.items()):
    statuses = [_decode_sidechain_gate_status(float(row.get("metric_value", 1.0))) for row in rows]
    pass_count = sum(1 for status in statuses if status == "pass")
    warn_count = sum(1 for status in statuses if status == "warn")
    fail_count = sum(1 for status in statuses if status == "fail")
    case_count = len({(row.get("case_id"), row.get("case_kind")) for row in rows})
    status = "fail" if fail_count else ("warn" if warn_count else "pass")
    reasons = sorted({
      str(row.get("note"))
      for row in rows
      if isinstance(row.get("note"), str) and row.get("note")
    })
    summaries.append(
      SidechainGateSummary(
        path_id=path_id,
        condition=condition,
        case_count=case_count,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        status=status,
        reason="; ".join(reasons) if reasons else "",
      ),
    )
  return summaries


def _coverage_reason(
  path_id: str,
  *,
  case_count: int,
  has_any_metrics: bool,
) -> tuple[str, str]:
  if case_count == 0:
    pending = {
      "logits-helper-branches": "Pending fast-tier instrumentation.",
      "averaged-encoding": "Pending averaging-path instrumentation.",
      "end-to-end-run-apis": "Pending run-API instrumentation.",
    }
    if path_id in pending:
      return ("pending", pending[path_id])
    if path_id == "checkpoint-family-load":
      return ("excluded", "Audit path uses categorical family-load metrics.")
    return ("pending", "No evidence rows collected for this path.")
  if path_id == "checkpoint-family-load":
    return (
      "instrumented",
      "Audit path is categorical; correlation and error columns are intentionally n/a.",
    )
  if has_any_metrics:
    return ("instrumented", "Path has collected evidence rows.")
  return ("pending", "Path has rows but no scalar metrics.")


def _build_coverage_entries(
  matrix_paths: list[dict[str, Any]],
  evidence_summaries: list[EvidenceSummary],
) -> list[CoverageEntry]:
  summary_by_path: dict[str, list[EvidenceSummary]] = {}
  for summary in evidence_summaries:
    summary_by_path.setdefault(summary.path_id, []).append(summary)

  entries: list[CoverageEntry] = []
  for path in matrix_paths:
    path_id = path.get("id")
    if not isinstance(path_id, str):
      continue
    summaries = summary_by_path.get(path_id, [])
    if not summaries:
      status, reason = _coverage_reason(path_id, case_count=0, has_any_metrics=False)
      entries.append(
        CoverageEntry(
          path_id=path_id,
          tier=str(path.get("tier", "unknown")),
          status=status,
          reason=reason,
        ),
      )
      continue
    case_count = max(summary.case_count for summary in summaries)
    has_any_metrics = any(
      any(
        value is not None
        for value in (
          summary.pearson_mean,
          summary.mae_mean,
          summary.rmse_mean,
          summary.token_agreement_mean,
        )
      )
      for summary in summaries
    )
    status, reason = _coverage_reason(
      path_id,
      case_count=case_count,
      has_any_metrics=has_any_metrics,
    )
    entries.append(
      CoverageEntry(
        path_id=path_id,
        tier=str(path.get("tier", "unknown")),
        status=status,
        reason=reason,
      ),
    )
  return entries


def _plot_no_data(title: str, output_path: Path) -> None:
  fig, ax = plt.subplots(figsize=(8, 3))
  ax.axis("off")
  ax.text(0.5, 0.5, f"No data available for {title}.", ha="center", va="center")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_path_counts(path_counts: dict[str, int], output_path: Path) -> None:
  tiers = ["parity_fast", "parity_heavy", "parity_audit"]
  values = [path_counts.get(tier, 0) for tier in tiers]
  fig, ax = plt.subplots(figsize=(8, 4))
  bars = ax.bar(tiers, values, color=["#4C78A8", "#F58518", "#54A24B"])
  ax.set_title("Parity Paths by Tier")
  ax.set_ylabel("Path count")
  ax.set_ylim(0, max(values + [1]) * 1.2)
  for bar, value in zip(bars, values, strict=False):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(value), ha="center")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_test_status(results: list[TierResult], output_path: Path) -> None:
  tiers = [result.name for result in results]
  passed = [result.passed for result in results]
  skipped = [result.skipped for result in results]
  failed = [result.failed for result in results]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.bar(tiers, passed, label="passed", color="#54A24B")
  ax.bar(tiers, skipped, bottom=passed, label="skipped", color="#9D9DA1")
  stacked_base = [a + b for a, b in zip(passed, skipped, strict=False)]
  ax.bar(tiers, failed, bottom=stacked_base, label="failed", color="#E45756")
  ax.set_title("Parity Test Outcomes by Tier")
  ax.set_ylabel("Test count")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_family_coverage(expected: dict[str, int], available: dict[str, int], output_path: Path) -> None:
  families = sorted(expected)
  expected_values = [expected[family] for family in families]
  available_values = [available.get(family, 0) for family in families]

  x_positions = list(range(len(families)))
  width = 0.38
  fig, ax = plt.subplots(figsize=(9, 4))
  ax.bar([x - width / 2 for x in x_positions], expected_values, width=width, label="expected")
  ax.bar([x + width / 2 for x in x_positions], available_values, width=width, label="available")
  ax.set_xticks(x_positions)
  ax.set_xticklabels(families)
  ax.set_ylabel("Checkpoint count")
  ax.set_title("Converted Checkpoint Family Coverage (parity_audit)")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_correlation_by_path(points: list[PathPearsonPoint], output_path: Path) -> None:
  if not points:
    _plot_no_data("path correlation swarm", output_path)
    return

  lane_labels = sorted({_lane_label(point.path_id, point.condition) for point in points})
  if not lane_labels:
    _plot_no_data("path correlation swarm", output_path)
    return

  lane_index = {lane_label: index for index, lane_label in enumerate(lane_labels)}
  by_case_kind: dict[str, list[PathPearsonPoint]] = {}
  values_by_lane: dict[str, list[float]] = {lane_label: [] for lane_label in lane_labels}
  for point in points:
    by_case_kind.setdefault(point.case_kind, []).append(point)
    lane_label = _lane_label(point.path_id, point.condition)
    values_by_lane[lane_label].append(point.value)

  palette_cycle = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
  palette = {
    case_kind: palette_cycle[index % len(palette_cycle)]
    for index, case_kind in enumerate(sorted(by_case_kind))
  }
  rng = np.random.default_rng(0)

  fig, ax = plt.subplots(figsize=(12, 5.2))
  for case_kind, case_points in sorted(by_case_kind.items()):
    x_values = []
    y_values = []
    for point in case_points:
      center = float(lane_index[_lane_label(point.path_id, point.condition)])
      x_values.append(float(rng.normal(loc=center, scale=0.07)))
      y_values.append(point.value)
    ax.scatter(
      x_values,
      y_values,
      s=24,
      alpha=0.75,
      color=palette[case_kind],
      edgecolors="none",
      label=case_kind,
    )

  for lane_label, index in lane_index.items():
    lane_values = values_by_lane[lane_label]
    if not lane_values:
      continue
    median_value = float(np.median(np.asarray(lane_values, dtype=np.float64)))
    ax.hlines(median_value, index - 0.24, index + 0.24, color="#222222", linewidth=1.6, zorder=3)

  ax.axhline(0.95, color="#9D9DA1", linestyle="--", linewidth=1.0)
  ax.set_xticks(np.arange(len(lane_labels)))
  ax.set_xticklabels(lane_labels, rotation=35, ha="right")
  ax.set_ylim(0.0, 1.02)
  ax.set_ylabel("Pearson r (per case)")
  ax.set_title("Pearson Correlation by Parity Path/Condition (Swarm)")

  legend_handles, legend_labels = ax.get_legend_handles_labels()
  legend_handles.extend(
    [
      plt.Line2D([0], [0], color="#222222", lw=1.6, label="median per path"),
      plt.Line2D([0], [0], color="#9D9DA1", lw=1.0, linestyle="--", label="threshold r=0.95"),
    ],
  )
  legend_labels.extend(["median per path", "threshold r=0.95"])
  ax.legend(legend_handles, legend_labels, loc="lower left", ncols=2)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_error_by_path(summaries: list[EvidenceSummary], output_path: Path) -> None:
  with_mae = [summary for summary in summaries if summary.mae_mean is not None]
  if not with_mae:
    _plot_no_data("path error summary", output_path)
    return
  labels = [_lane_label(summary.path_id, summary.condition) for summary in with_mae]
  maes = [summary.mae_mean or 0.0 for summary in with_mae]
  rmses = [summary.rmse_mean or 0.0 for summary in with_mae]
  x_pos = np.arange(len(labels))
  width = 0.38
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.bar(x_pos - width / 2, maes, width=width, label="MAE")
  ax.bar(x_pos + width / 2, rmses, width=width, label="RMSE")
  ax.set_xticks(x_pos)
  ax.set_xticklabels(labels, rotation=35, ha="right")
  ax.set_ylabel("Error")
  ax.set_title("Error Metrics by Parity Path/Condition")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_intrinsic_ratio_by_path(summaries: list[IntrinsicSummary], output_path: Path) -> None:
  if not summaries:
    _plot_no_data("intrinsic baseline ratio summary", output_path)
    return
  with_ratio = [summary for summary in summaries if summary.ratio_mean is not None]
  if not with_ratio:
    _plot_no_data("intrinsic baseline ratio summary", output_path)
    return
  labels = [_lane_label(summary.path_id, summary.condition) for summary in with_ratio]
  ratio_mean = np.asarray([float(summary.ratio_mean or 0.0) for summary in with_ratio], dtype=np.float64)
  ratio_max = np.asarray(
    [float(summary.ratio_max if summary.ratio_max is not None else summary.ratio_mean or 0.0) for summary in with_ratio],
    dtype=np.float64,
  )
  positive_values = np.concatenate([ratio_mean[ratio_mean > 0], ratio_max[ratio_max > 0]])
  if positive_values.size == 0:
    _plot_no_data("intrinsic baseline ratio summary", output_path)
    return
  safe_floor = max(float(np.min(positive_values) * 0.5), 1e-6)
  plotted_mean = np.maximum(ratio_mean, safe_floor)
  plotted_max = np.maximum(ratio_max, safe_floor)
  non_positive_count = int(np.sum(ratio_mean <= 0.0) + np.sum(ratio_max <= 0.0))
  x_pos = np.arange(len(labels))
  fig, ax = plt.subplots(figsize=(11, 4.8))
  ax.bar(
    x_pos,
    plotted_mean - safe_floor,
    bottom=safe_floor,
    color="#72B7B2",
    label="mean cross/intrinsic ratio",
  )
  ax.scatter(x_pos, plotted_max, color="#E45756", zorder=3, label="max ratio")
  upper = float(max(np.max(plotted_mean), np.max(plotted_max)))
  y_upper = upper * 1.35 if upper > safe_floor else safe_floor * 10.0
  ax.set_yscale("log")
  ax.set_ylim(safe_floor, y_upper)
  if safe_floor <= 1.0 <= y_upper:
    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1, label="ratio=1 baseline")
  ax.set_xticks(x_pos)
  ax.set_xticklabels(labels, rotation=30, ha="right")
  ax.set_ylabel("Cross-model MAE / pooled intrinsic MAE (log scale)")
  ax.set_title("Baseline-aware parity ratio by path/condition (log y-axis)")
  if non_positive_count:
    ax.text(
      0.01,
      0.01,
      f"{non_positive_count} non-positive ratio value(s) clipped to plotting floor.",
      transform=ax.transAxes,
      fontsize=8,
    )
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_macro_swarm(points: list[dict[str, Any]], output_path: Path) -> None:
  relevant = [
    row
    for row in points
    if row.get("point_kind") in {"macro_sequence_identity", "macro_entropy"}
    and isinstance(row.get("condition"), str)
  ]
  if not relevant:
    _plot_no_data("macro distribution swarm", output_path)
    return

  categories = sorted({(str(row["point_kind"]), str(row["condition"])) for row in relevant})
  x_positions: list[float] = []
  y_values: list[float] = []
  colors: list[str] = []
  tick_positions: list[float] = []
  tick_labels: list[str] = []
  rng = np.random.default_rng(0)
  palette = {"reference": "#4C78A8", "observed": "#F58518"}

  offset = 0.0
  for point_kind, condition in categories:
    rows = [row for row in relevant if row.get("point_kind") == point_kind and row.get("condition") == condition]
    reference_values = np.asarray([float(row["reference_value"]) for row in rows], dtype=np.float64)
    observed_values = np.asarray([float(row["observed_value"]) for row in rows], dtype=np.float64)
    if reference_values.size == 0 or observed_values.size == 0:
      continue

    left = offset
    right = offset + 0.6
    ref_jitter = rng.normal(loc=left, scale=0.04, size=reference_values.size)
    obs_jitter = rng.normal(loc=right, scale=0.04, size=observed_values.size)
    x_positions.extend(ref_jitter.tolist())
    y_values.extend(reference_values.tolist())
    colors.extend([palette["reference"]] * reference_values.size)
    x_positions.extend(obs_jitter.tolist())
    y_values.extend(observed_values.tolist())
    colors.extend([palette["observed"]] * observed_values.size)
    tick_positions.extend([left, right])
    metric_label = "identity" if point_kind == "macro_sequence_identity" else "entropy"
    tick_labels.extend([f"{metric_label}/{condition}\nref", f"{metric_label}/{condition}\njax"])
    offset += 1.8

  if not y_values:
    _plot_no_data("macro distribution swarm", output_path)
    return
  fig, ax = plt.subplots(figsize=(max(10, len(tick_labels) * 0.75), 5.2))
  ax.scatter(x_positions, y_values, s=12, alpha=0.55, c=colors, edgecolors="none")
  ax.set_xticks(tick_positions)
  ax.set_xticklabels(tick_labels, rotation=20, ha="right")
  ax.set_ylabel("Metric value")
  ax.set_title("Macro distribution swarm diagnostics")
  legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["reference"], label="reference", markersize=6),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["observed"], label="JAX", markersize=6),
  ]
  ax.legend(handles=legend_handles, loc="upper right")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_scatter_grid(points: list[dict[str, Any]], output_path: Path) -> None:
  key_paths = [
    "decoder-conditional-scoring",
    "autoregressive-sampling",
    "ligand-conditioning-context",
    "side-chain-packer",
  ]
  selected = [path_id for path_id in key_paths if any(row.get("path_id") == path_id for row in points)]
  if not selected:
    _plot_no_data("scatter diagnostics", output_path)
    return

  ncols = 2
  nrows = int(np.ceil(len(selected) / ncols))
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
  axes_array = np.array(axes, ndmin=1).ravel()
  for axis in axes_array:
    axis.set_visible(False)

  for index, path_id in enumerate(selected):
    axis = axes_array[index]
    axis.set_visible(True)
    rows = [row for row in points if row.get("path_id") == path_id]
    ref_values = np.asarray([float(row["reference_value"]) for row in rows], dtype=np.float64)
    obs_values = np.asarray([float(row["observed_value"]) for row in rows], dtype=np.float64)
    if ref_values.size > 5000:
      rng = np.random.default_rng(0)
      sample_idx = rng.choice(ref_values.size, size=5000, replace=False)
      ref_values = ref_values[sample_idx]
      obs_values = obs_values[sample_idx]
    axis.scatter(ref_values, obs_values, s=4, alpha=0.3, color="#4C78A8")
    low = min(float(np.min(ref_values)), float(np.min(obs_values)))
    high = max(float(np.max(ref_values)), float(np.max(obs_values)))
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#E45756")
    axis.set_title(path_id)
    axis.set_xlabel("Reference")
    axis.set_ylabel("Observed (JAX)")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_bland_altman(points: list[dict[str, Any]], output_path: Path) -> None:
  rows = [row for row in points if row.get("path_id") == "decoder-conditional-scoring"]
  if not rows:
    _plot_no_data("Bland-Altman diagnostics", output_path)
    return
  ref_values = np.asarray([float(row["reference_value"]) for row in rows], dtype=np.float64)
  obs_values = np.asarray([float(row["observed_value"]) for row in rows], dtype=np.float64)
  means = (ref_values + obs_values) / 2.0
  diffs = obs_values - ref_values
  mean_diff = float(np.mean(diffs))
  std_diff = float(np.std(diffs))
  upper = mean_diff + 1.96 * std_diff
  lower = mean_diff - 1.96 * std_diff

  if means.size > 6000:
    rng = np.random.default_rng(1)
    idx = rng.choice(means.size, size=6000, replace=False)
    means = means[idx]
    diffs = diffs[idx]

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.scatter(means, diffs, s=4, alpha=0.3, color="#4C78A8")
  ax.axhline(mean_diff, color="#E45756", linestyle="--", label=f"mean diff={mean_diff:.4f}")
  ax.axhline(upper, color="#9D9DA1", linestyle=":", label=f"+1.96 SD={upper:.4f}")
  ax.axhline(lower, color="#9D9DA1", linestyle=":", label=f"-1.96 SD={lower:.4f}")
  ax.set_xlabel("Mean(reference, observed)")
  ax.set_ylabel("Observed - Reference")
  ax.set_title("Bland-Altman: decoder-conditional-scoring")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_case_coverage_heatmap(
  matrix_paths: list[dict[str, Any]],
  metric_rows: list[dict[str, Any]],
  output_path: Path,
) -> None:
  if not metric_rows:
    _plot_no_data("case coverage heatmap", output_path)
    return
  path_ids = [path["id"] for path in matrix_paths if isinstance(path.get("id"), str)]
  case_ids = sorted({str(row.get("case_id", "")) for row in metric_rows})
  if not path_ids or not case_ids:
    _plot_no_data("case coverage heatmap", output_path)
    return
  matrix = np.zeros((len(path_ids), len(case_ids)), dtype=np.int32)
  path_index = {path_id: idx for idx, path_id in enumerate(path_ids)}
  case_index = {case_id: idx for idx, case_id in enumerate(case_ids)}
  for row in metric_rows:
    path_id = row.get("path_id")
    case_id = row.get("case_id")
    if not isinstance(path_id, str) or not isinstance(case_id, str):
      continue
    if path_id in path_index and case_id in case_index:
      matrix[path_index[path_id], case_index[case_id]] = 1
  fig, ax = plt.subplots(figsize=(12, max(4, len(path_ids) * 0.45)))
  cmap = ListedColormap(["#E5E7EB", "#2F855A"])
  norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
  ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
  ax.set_xticks(np.arange(len(case_ids)))
  ax.set_xticklabels(case_ids, rotation=40, ha="right")
  ax.set_yticks(np.arange(len(path_ids)))
  ax.set_yticklabels(path_ids)
  ax.set_title("Binary evidence coverage: path x case (1=present, 0=absent)")
  ax.set_xlabel("Case")
  ax.set_ylabel("Parity path")
  ax.legend(
    handles=[
      Patch(facecolor="#2F855A", edgecolor="#1F2937", label="Present (1): evidence row exists"),
      Patch(facecolor="#E5E7EB", edgecolor="#1F2937", label="Absent (0): no evidence row"),
    ],
    loc="upper right",
  )
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _fmt_optional(value: float | None, digits: int = 4, *, reason: str | None = None) -> str:
  if value is not None:
    return f"{value:.{digits}f}"
  return f"n/a({reason})" if reason else "n/a"


def _fmt_joined_ints(values: tuple[int, ...]) -> str:
  if not values:
    return "n/a"
  return ", ".join(str(value) for value in values)


def _fmt_joined_strings(values: tuple[str, ...]) -> str:
  if not values:
    return "n/a"
  return ", ".join(values)


def _lane_label(path_id: str, condition: str | None) -> str:
  if condition is None:
    return path_id
  return f"{path_id} [{condition}]"


def _fmt_condition(condition: str | None) -> str:
  return condition if condition is not None else "default"


def _markdown_figure(*, alt: str, src: str, caption: str) -> list[str]:
  return [
    f"![{alt}]({src})",
    f"_Figure: {caption}_",
    "",
  ]


def _html_figure(*, alt: str, src: str, caption: str) -> str:
  return (
    "<figure>\n"
    f"  <img alt=\"{alt}\" src=\"{src}\">\n"
    f"  <figcaption>{caption}</figcaption>\n"
    "</figure>\n"
  )


def _summary_na_reason(
  summary: EvidenceSummary,
  coverage_by_path: dict[str, CoverageEntry],
) -> str | None:
  if summary.path_id == "checkpoint-family-load":
    return "categorical"
  if summary.case_count == 0:
    entry = coverage_by_path.get(summary.path_id)
    if entry is not None:
      return entry.status
  return None


def _has_scalar_metrics(summary: EvidenceSummary) -> bool:
  return any(
    value is not None
    for value in (
      summary.pearson_mean,
      summary.mae_mean,
      summary.rmse_mean,
      summary.token_agreement_mean,
    )
  )


def _build_executive_snapshot(
  *,
  tier_results: list[TierResult],
  evidence_summaries: list[EvidenceSummary],
  intrinsic_summaries: list[IntrinsicSummary],
  sidechain_gate_summaries: list[SidechainGateSummary],
  coverage_entries: list[CoverageEntry],
  evidence_metrics_present: bool,
) -> ExecutiveSnapshot:
  total_tests = sum(result.tests for result in tier_results)
  total_passed = sum(result.passed for result in tier_results)
  total_failed = sum(result.failed for result in tier_results)
  total_skipped = sum(result.skipped for result in tier_results)
  failed_tiers = tuple(result.name for result in tier_results if result.failed > 0)
  missing_tiers = tuple(result.name for result in tier_results if result.missing)

  scalar_lanes = [
    summary for summary in evidence_summaries if summary.case_count > 0 and _has_scalar_metrics(summary)
  ]
  pearson_lanes = [
    summary for summary in scalar_lanes if summary.case_count > 0 and summary.pearson_mean is not None
  ]
  pearson_alert_lanes = [
    summary
    for summary in pearson_lanes
    if (
      summary.pearson_min is not None
      and summary.pearson_min < 0.95
      or summary.pearson_pass_rate is not None
      and summary.pearson_pass_rate < 1.0
    )
  ]

  coverage_counts: dict[str, int] = {"instrumented": 0, "pending": 0, "excluded": 0}
  for entry in coverage_entries:
    if entry.status in coverage_counts:
      coverage_counts[entry.status] += 1

  sidechain_warn = sum(summary.warn_count for summary in sidechain_gate_summaries)
  sidechain_fail = sum(summary.fail_count for summary in sidechain_gate_summaries)
  intrinsic_alert_lanes = tuple(
    _lane_label(summary.path_id, summary.condition)
    for summary in intrinsic_summaries
    if summary.ratio_max is not None and summary.ratio_max >= 100.0
  )

  if failed_tiers or missing_tiers:
    status = "Attention required: at least one tier failed or lacks JUnit evidence."
  elif not evidence_metrics_present:
    status = "Tier checks are green, but lane-level evidence CSVs are missing."
  elif pearson_alert_lanes or sidechain_fail or coverage_counts["pending"] > 0:
    status = "Tier checks pass; review caveats before declaring full parity closure."
  else:
    status = "Tier checks and collected lane evidence currently support parity on this corpus."

  outcomes = [
    (
      f"Tier outcomes: {total_passed}/{total_tests} passed, {total_failed} failed,"
      f" {total_skipped} skipped."
    ),
    (
      f"Coverage map: {coverage_counts['instrumented']} instrumented,"
      f" {coverage_counts['pending']} pending, {coverage_counts['excluded']} excluded."
    ),
  ]
  if missing_tiers:
    outcomes.append(f"Missing JUnit artifacts: {', '.join(missing_tiers)}.")
  if evidence_metrics_present:
    outcomes.append(
      (
        f"Scalar evidence spans {len(scalar_lanes)} path/condition lanes;"
        f" Pearson metrics are present in {len(pearson_lanes)} lanes."
      ),
    )
    if pearson_alert_lanes:
      outcomes.append(
        (
          f"{len(pearson_alert_lanes)} lane(s) dip below the Pearson policy floor"
          " (min r or pass-rate checks)."
        ),
      )
  else:
    outcomes.append("Lane-level metrics unavailable until `collect_parity_evidence.py` is rerun.")

  caveats: list[str] = []
  if intrinsic_alert_lanes:
    top_lanes = ", ".join(intrinsic_alert_lanes[:2])
    suffix = " (and more)." if len(intrinsic_alert_lanes) > 2 else "."
    caveats.append(
      (
        "Intrinsic ratios use pooled intrinsic MAE in the denominator; near-zero baselines can"
        f" inflate ratios for {top_lanes}{suffix}"
      ),
    )
  if sidechain_fail:
    caveats.append(
      f"Side-chain-conditioned integration gate reports {sidechain_fail} fail verdict(s).",
    )
  elif sidechain_warn:
    caveats.append(
      (
        "Side-chain-conditioned integration gate reports"
        f" {sidechain_warn} warning verdict(s), driven by excluded corpus lanes."
      ),
    )
  if not caveats:
    caveats.append("No blocking caveats are currently visible in collected evidence rows.")

  return ExecutiveSnapshot(
    status=status,
    outcomes=tuple(outcomes),
    caveats=tuple(caveats),
  )


def _terminology_legend_rows() -> tuple[tuple[str, str], ...]:
  return (
    (
      "Mean r / Min r",
      "Pearson correlation summaries per path/condition lane; values nearer 1 indicate tighter parity.",
    ),
    (
      "Pearson pass rate",
      "Fraction of per-case Pearson checks passing policy thresholds for that lane.",
    ),
    (
      "n/a(categorical | pending | excluded)",
      "Metric intentionally absent because the path is categorical, not instrumented, or excluded.",
    ),
    (
      "Cross/intrinsic ratio",
      "Cross-model MAE divided by pooled intrinsic MAE; near-zero denominators can inflate ratios.",
    ),
    (
      "Coverage heatmap values",
      "Binary semantics only: 1 means at least one evidence row is present; 0 means absent.",
    ),
  )


def _render_markdown(
  *,
  generated_at: str,
  path_counts: dict[str, int],
  tier_results: list[TierResult],
  family_expected: dict[str, int],
  family_available: dict[str, int],
  checksum_rel_path: str,
  matrix_paths: list[dict[str, Any]],
  evidence_summaries: list[EvidenceSummary],
  case_inventory_entries: list[CaseInventoryEntry],
  intrinsic_summaries: list[IntrinsicSummary],
  macro_summaries: list[MacroSummary],
  sidechain_gate_summaries: list[SidechainGateSummary],
  coverage_entries: list[CoverageEntry],
  evidence_metrics_rel_path: str,
  evidence_points_rel_path: str,
  evidence_metrics_present: bool,
  tied_lane_descriptors: list[TiedLaneDescriptor] | None = None,
) -> str:
  tied_lanes = tied_lane_descriptors or []
  snapshot = _build_executive_snapshot(
    tier_results=tier_results,
    evidence_summaries=evidence_summaries,
    intrinsic_summaries=intrinsic_summaries,
    sidechain_gate_summaries=sidechain_gate_summaries,
    coverage_entries=coverage_entries,
    evidence_metrics_present=evidence_metrics_present,
  )
  legend_rows = _terminology_legend_rows()
  pearson_sample_count = sum(
    summary.case_count
    for summary in evidence_summaries
    if summary.case_count > 0 and summary.pearson_mean is not None
  )
  scalar_lane_count = sum(
    1 for summary in evidence_summaries if summary.case_count > 0 and _has_scalar_metrics(summary)
  )
  intrinsic_lane_count = sum(1 for summary in intrinsic_summaries if summary.ratio_mean is not None)
  coverage_case_count = len(case_inventory_entries)
  coverage_path_count = len(coverage_entries)

  lines = [
    "# Parity Assessment Report",
    "",
    f"Generated: **{generated_at}** (UTC)",
    "",
    "## Executive parity snapshot",
    "",
    f"**Status:** {snapshot.status}",
    "",
    "### Key outcomes",
    "",
  ]
  lines.extend(f"- {outcome}" for outcome in snapshot.outcomes)
  lines.extend(["", "### Caveats", ""])
  lines.extend(f"- {caveat}" for caveat in snapshot.caveats)
  lines.extend(
    [
      "",
      "## How parity is assessed",
      "",
      "1. **`parity_fast`**: deterministic fixture and helper-branch parity checks (no upstream checkout required).",
      "2. **`parity_heavy`**: reference-backed numerical parity checks against pinned LigandMPNN.",
      "3. **`parity_audit`**: converted checkpoint-family load/audit checks across model families.",
      "",
      "## Terminology / interpretation legend",
      "",
      "| Term | Interpretation |",
      "| --- | --- |",
    ],
  )
  for term, interpretation in legend_rows:
    lines.append(f"| {term} | {interpretation} |")

  lines.extend(
    [
      "",
      "## Tier test outcomes (from JUnit)",
      "",
      "| Tier | Tests | Passed | Skipped | Failed | Notes |",
      "| --- | ---: | ---: | ---: | ---: | --- |",
    ],
  )
  for result in tier_results:
    notes = "missing junit" if result.missing else ""
    lines.append(
      f"| `{result.name}` | {result.tests} | {result.passed} | {result.skipped} | {result.failed} | {notes} |",
    )
  lines.append("")
  lines.extend(
    _markdown_figure(
      alt="Tier-level JUnit pass skip fail outcomes",
      src="assets/parity_test_outcomes_by_tier.png",
      caption=(
        "Stacked JUnit pass/skip/fail totals per tier; this is the first release gate for triage."
      ),
    ),
  )

  lines.extend(
    [
      "## Converted family checkpoint coverage",
      "",
      "| Family | Expected converted | Available locally |",
      "| --- | ---: | ---: |",
    ],
  )
  for family in sorted(family_expected):
    lines.append(f"| `{family}` | {family_expected[family]} | {family_available.get(family, 0)} |")
  lines.append("")
  lines.extend(
    _markdown_figure(
      alt="Converted checkpoint availability by family",
      src="assets/parity_family_checkpoint_coverage.png",
      caption=(
        "Expected vs locally present converted checkpoints for `parity_audit`; missing assets block"
        " audit evidence collection."
      ),
    ),
  )

  lines.extend(
    [
      "## Expanded evidence summary",
      "",
      "### Evaluated protein systems and cases",
      "",
      "Case metadata below is extracted directly from collected evidence rows.",
      "",
      "| Case ID | Kind | Length(s) | Seed(s) | Backbone ID(s) | Checkpoint ID(s) | Tier(s) | Paths |",
      "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ],
  )
  if case_inventory_entries:
    for entry in case_inventory_entries:
      lines.append(
        "| "
        f"`{entry.case_id}` | `{entry.case_kind}` | {_fmt_joined_ints(entry.sequence_lengths)} | "
        f"{_fmt_joined_ints(entry.seeds)} | {_fmt_joined_strings(entry.backbone_ids)} | "
        f"{_fmt_joined_strings(entry.checkpoint_ids)} | {_fmt_joined_strings(entry.tiers)} | "
        f"{entry.path_count} |"
      )
  else:
    lines.append("| *(none)* | n/a | n/a | n/a | n/a | n/a | n/a | 0 |")
  lines.append("")

  if tied_lanes:
    lines.extend(
      [
        "### Tied/multistate apples-to-apples lanes",
        "",
        "Configured tied/multistate lanes are listed beside evidence tables so unlike strategies never merge.",
        "",
        "| Condition | Comparison API | Reference combiner | JAX strategy | Token comparison | Primary |",
        "| --- | --- | --- | --- | --- | --- |",
      ],
    )
    for lane in tied_lanes:
      lines.append(
        "| "
        f"`{lane.condition}` | `{lane.comparison_api}` | `{lane.reference_combiner}` | "
        f"`{lane.jax_multi_state_strategy}` | "
        f"`{'enabled' if lane.token_comparison_enabled else 'disabled'}` | "
        f"`{'yes' if lane.is_primary else 'no'}` |"
      )
    lines.append("")

  if evidence_metrics_present:
    coverage_by_path = {entry.path_id: entry for entry in coverage_entries}
    lines.extend(
      [
        f"Evidence metrics: `{evidence_metrics_rel_path}`",
        f"Point samples: `{evidence_points_rel_path}`",
        "",
        "Condition labels isolate strategy lanes (for example, tied/multistate apples-to-apples comparisons).",
        "",
        "| Path | Condition | Cases | Mean r | Min r | Pearson pass rate | Mean MAE | Mean RMSE | Mean token agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
      ],
    )
    for summary in evidence_summaries:
      missing_reason = _summary_na_reason(summary, coverage_by_path)
      lines.append(
        "| "
        f"`{summary.path_id}` | `{_fmt_condition(summary.condition)}` | {summary.case_count} | "
        f"{_fmt_optional(summary.pearson_mean, reason=missing_reason)} | "
        f"{_fmt_optional(summary.pearson_min, reason=missing_reason)} | "
        f"{_fmt_optional(summary.pearson_pass_rate, reason=missing_reason)} | "
        f"{_fmt_optional(summary.mae_mean, reason=missing_reason)} | "
        f"{_fmt_optional(summary.rmse_mean, reason=missing_reason)} | "
        f"{_fmt_optional(summary.token_agreement_mean, reason=missing_reason)} |"
      )
    lines.append("")
    lines.extend(
      _markdown_figure(
        alt="Pearson swarm by path and condition lanes",
        src="assets/parity_correlation_by_path.png",
        caption=(
          f"Pearson swarm across lane-level case rows (n={pearson_sample_count}). Each point is one"
          " path/condition/case/seed value, black ticks mark medians, and the dashed line marks"
          " the r=0.95 policy floor."
        ),
      ),
    )
    lines.extend(
      _markdown_figure(
        alt="Mean MAE and RMSE by parity lane",
        src="assets/parity_error_by_path.png",
        caption=(
          f"Mean MAE and RMSE across {scalar_lane_count} scalar-metric lanes. Rows shown as"
          " `n/a(...)` are pending, excluded, or categorical by design."
        ),
      ),
    )
    lines.extend(
      _markdown_figure(
        alt="Reference versus JAX scatter diagnostics for core parity paths",
        src="assets/parity_scatter_core_paths.png",
        caption=(
          "Reference vs JAX scatter samples for core paths; tighter diagonal alignment indicates"
          " stronger parity. Dense clouds may be downsampled for readability."
        ),
      ),
    )
    lines.extend(
      _markdown_figure(
        alt="Bland Altman drift diagnostics for decoder conditional scoring",
        src="assets/parity_bland_altman_decoder_conditional.png",
        caption=(
          "Bland-Altman view for decoder-conditional-scoring. The center line is mean drift and"
          " dotted lines show ±1.96 SD."
        ),
      ),
    )
    lines.extend(
      _markdown_figure(
        alt="Binary evidence coverage heatmap with present versus absent semantics",
        src="assets/parity_case_coverage_heatmap.png",
        caption=(
          f"Binary coverage matrix ({coverage_path_count} paths × {coverage_case_count} cases):"
          " green=present (>=1 evidence row), gray=absent. Color intensity does not encode"
          " magnitude."
        ),
      ),
    )

    lines.extend(
      [
        "### Baseline-aware intrinsic parity",
        "",
        "| Path | Condition | Cases | Mean cross/intrinsic ratio | Max ratio | Pass@95 envelope | Pass@99 envelope |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
      ],
    )
    if intrinsic_summaries:
      for summary in intrinsic_summaries:
        lines.append(
          "| "
          f"`{summary.path_id}` | `{_fmt_condition(summary.condition)}` | {summary.case_count} | "
          f"{_fmt_optional(summary.ratio_mean)} | {_fmt_optional(summary.ratio_max)} | "
          f"{_fmt_optional(summary.pass_95_rate)} | {_fmt_optional(summary.pass_99_rate)} |"
        )
    else:
      lines.append("| *(none)* | `default` | 0 | n/a | n/a | n/a | n/a |")
    lines.append("")
    lines.extend(
      _markdown_figure(
        alt="Log scale intrinsic ratio by path and condition lanes",
        src="assets/parity_intrinsic_ratio_by_path.png",
        caption=(
          f"Log-scale cross/intrinsic ratio chart across {intrinsic_lane_count} lane(s). The"
          " denominator is pooled intrinsic MAE, so near-zero baselines can inflate ratios."
        ),
      ),
    )

    lines.extend(
      [
        "### Macro distribution parity",
        "",
        "| Path | Condition | Cases | Mean identity Wasserstein | Mean entropy Wasserstein | Mean composition JS | Median identity KS p-value |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
      ],
    )
    if macro_summaries:
      for summary in macro_summaries:
        lines.append(
          "| "
          f"`{summary.path_id}` | `{summary.condition}` | {summary.case_count} | "
          f"{_fmt_optional(summary.identity_wasserstein_mean, reason=summary.note)} | "
          f"{_fmt_optional(summary.entropy_wasserstein_mean, reason=summary.note)} | "
          f"{_fmt_optional(summary.composition_js_mean, reason=summary.note)} | "
          f"{_fmt_optional(summary.identity_ks_pvalue_median, reason=summary.note)} |"
        )
    else:
      lines.append("| *(none)* | `n/a` | 0 | n/a | n/a | n/a | n/a |")
    lines.extend(
      [
        "",
        "### Side-chain-conditioned integration gate",
        "",
        "| Path | Condition | Cases | Pass | Warn | Fail | Outcome | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
      ],
    )
    if sidechain_gate_summaries:
      for summary in sidechain_gate_summaries:
        lines.append(
          "| "
          f"`{summary.path_id}` | `{summary.condition}` | {summary.case_count} | "
          f"{summary.pass_count} | {summary.warn_count} | {summary.fail_count} | "
          f"`{summary.status}` | {summary.reason or '-'} |"
        )
    else:
      lines.append("| *(none)* | `side_chain_conditioned` | 0 | 0 | 0 | 0 | `warn` | n/a |")
    lines.append("")
    lines.extend(
      _markdown_figure(
        alt="Macro distribution swarm comparing reference and JAX",
        src="assets/parity_macro_swarm.png",
        caption=(
          "Macro swarm compares reference vs JAX identity/entropy points by condition. Overlap"
          " implies better distributional parity; excluded lanes remain `n/a(...)`."
        ),
      ),
    )
    lines.extend(
      [
        "### Coverage and exclusions",
        "",
        "| Path | Tier | Status | Reason |",
        "| --- | --- | --- | --- |",
      ],
    )
    for entry in coverage_entries:
      lines.append(f"| `{entry.path_id}` | `{entry.tier}` | `{entry.status}` | {entry.reason} |")
    lines.append("")
  else:
    lines.extend(
      [
        "No evidence metrics CSV found. Run `scripts/collect_parity_evidence.py` first.",
        "",
      ],
    )

  lines.extend(
    [
      "## Appendix: path inventory and definitions",
      "",
      "### Parity path definitions",
      "",
      "| Path | Tier | Method | Metrics | Code paths |",
      "| --- | --- | --- | --- | --- |",
    ],
  )
  for path in matrix_paths:
    path_id = str(path.get("id", "unknown"))
    tier = str(path.get("tier", "unknown"))
    method = str(path.get("method", ""))
    metrics = ", ".join(str(metric) for metric in path.get("metrics", []))
    code_paths = ", ".join(str(code_path) for code_path in path.get("code_paths", []))
    lines.append(f"| `{path_id}` | `{tier}` | {method} | {metrics} | `{code_paths}` |")
  lines.extend(
    [
      "",
      "### Tier path inventory",
      "",
      "| Tier | Path count |",
      "| --- | ---: |",
    ],
  )
  for tier in ["parity_fast", "parity_heavy", "parity_audit"]:
    lines.append(f"| `{tier}` | {path_counts.get(tier, 0)} |")
  lines.append("")
  lines.extend(
    _markdown_figure(
      alt="Configured parity path counts by tier",
      src="assets/parity_paths_by_tier.png",
      caption=(
        "Inventory-only path counts by tier. Larger bars indicate broader instrumentation,"
        " not stronger parity."
      ),
    ),
  )
  lines.extend([f"Checkpoint checksum snapshot: `{checksum_rel_path}`", ""])
  return "\n".join(lines)


def _render_html(
  *,
  generated_at: str,
  path_counts: dict[str, int],
  tier_results: list[TierResult],
  family_expected: dict[str, int],
  family_available: dict[str, int],
  checksum_rel_path: str,
  matrix_paths: list[dict[str, Any]],
  evidence_summaries: list[EvidenceSummary],
  case_inventory_entries: list[CaseInventoryEntry],
  intrinsic_summaries: list[IntrinsicSummary],
  macro_summaries: list[MacroSummary],
  sidechain_gate_summaries: list[SidechainGateSummary],
  coverage_entries: list[CoverageEntry],
  evidence_metrics_rel_path: str,
  evidence_points_rel_path: str,
  evidence_metrics_present: bool,
  tied_lane_descriptors: list[TiedLaneDescriptor] | None = None,
) -> str:
  tied_lanes = tied_lane_descriptors or []
  snapshot = _build_executive_snapshot(
    tier_results=tier_results,
    evidence_summaries=evidence_summaries,
    intrinsic_summaries=intrinsic_summaries,
    sidechain_gate_summaries=sidechain_gate_summaries,
    coverage_entries=coverage_entries,
    evidence_metrics_present=evidence_metrics_present,
  )
  legend_rows = _terminology_legend_rows()
  pearson_sample_count = sum(
    summary.case_count
    for summary in evidence_summaries
    if summary.case_count > 0 and summary.pearson_mean is not None
  )
  scalar_lane_count = sum(
    1 for summary in evidence_summaries if summary.case_count > 0 and _has_scalar_metrics(summary)
  )
  intrinsic_lane_count = sum(1 for summary in intrinsic_summaries if summary.ratio_mean is not None)
  coverage_case_count = len(case_inventory_entries)
  coverage_path_count = len(coverage_entries)

  tier_rows = "\n".join(
    f"<tr><td><code>{tier}</code></td><td>{path_counts.get(tier, 0)}</td></tr>"
    for tier in ("parity_fast", "parity_heavy", "parity_audit")
  )
  result_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{result.name}</code></td>"
      f"<td>{result.tests}</td>"
      f"<td>{result.passed}</td>"
      f"<td>{result.skipped}</td>"
      f"<td>{result.failed}</td>"
      f"<td>{'missing junit' if result.missing else ''}</td>"
      "</tr>"
    )
    for result in tier_results
  )
  family_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{family}</code></td>"
      f"<td>{family_expected[family]}</td>"
      f"<td>{family_available.get(family, 0)}</td>"
      "</tr>"
    )
    for family in sorted(family_expected)
  )
  matrix_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{path.get('id', '')}</code></td>"
      f"<td><code>{path.get('tier', '')}</code></td>"
      f"<td>{path.get('method', '')}</td>"
      f"<td>{', '.join(str(metric) for metric in path.get('metrics', []))}</td>"
      f"<td><code>{', '.join(str(code_path) for code_path in path.get('code_paths', []))}</code></td>"
      "</tr>"
    )
    for path in matrix_paths
  )
  case_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{entry.case_id}</code></td>"
      f"<td><code>{entry.case_kind}</code></td>"
      f"<td>{_fmt_joined_ints(entry.sequence_lengths)}</td>"
      f"<td>{_fmt_joined_ints(entry.seeds)}</td>"
      f"<td>{_fmt_joined_strings(entry.backbone_ids)}</td>"
      f"<td>{_fmt_joined_strings(entry.checkpoint_ids)}</td>"
      f"<td>{_fmt_joined_strings(entry.tiers)}</td>"
      f"<td>{entry.path_count}</td>"
      "</tr>"
    )
    for entry in case_inventory_entries
  )
  tied_lane_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{lane.condition}</code></td>"
      f"<td><code>{lane.comparison_api}</code></td>"
      f"<td><code>{lane.reference_combiner}</code></td>"
      f"<td><code>{lane.jax_multi_state_strategy}</code></td>"
      f"<td><code>{'enabled' if lane.token_comparison_enabled else 'disabled'}</code></td>"
      f"<td><code>{'yes' if lane.is_primary else 'no'}</code></td>"
      "</tr>"
    )
    for lane in tied_lanes
  )

  coverage_by_path = {entry.path_id: entry for entry in coverage_entries}
  evidence_row_blocks: list[str] = []
  for summary in evidence_summaries:
    missing_reason = _summary_na_reason(summary, coverage_by_path)
    evidence_row_blocks.append(
      "<tr>"
      f"<td><code>{summary.path_id}</code></td>"
      f"<td><code>{_fmt_condition(summary.condition)}</code></td>"
      f"<td>{summary.case_count}</td>"
      f"<td>{_fmt_optional(summary.pearson_mean, reason=missing_reason)}</td>"
      f"<td>{_fmt_optional(summary.pearson_min, reason=missing_reason)}</td>"
      f"<td>{_fmt_optional(summary.pearson_pass_rate, reason=missing_reason)}</td>"
      f"<td>{_fmt_optional(summary.mae_mean, reason=missing_reason)}</td>"
      f"<td>{_fmt_optional(summary.rmse_mean, reason=missing_reason)}</td>"
      f"<td>{_fmt_optional(summary.token_agreement_mean, reason=missing_reason)}</td>"
      "</tr>"
    )
  evidence_rows = "\n".join(evidence_row_blocks)
  intrinsic_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{summary.path_id}</code></td>"
      f"<td><code>{_fmt_condition(summary.condition)}</code></td>"
      f"<td>{summary.case_count}</td>"
      f"<td>{_fmt_optional(summary.ratio_mean)}</td>"
      f"<td>{_fmt_optional(summary.ratio_max)}</td>"
      f"<td>{_fmt_optional(summary.pass_95_rate)}</td>"
      f"<td>{_fmt_optional(summary.pass_99_rate)}</td>"
      "</tr>"
    )
    for summary in intrinsic_summaries
  )
  macro_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{summary.path_id}</code></td>"
      f"<td><code>{summary.condition}</code></td>"
      f"<td>{summary.case_count}</td>"
      f"<td>{_fmt_optional(summary.identity_wasserstein_mean, reason=summary.note)}</td>"
      f"<td>{_fmt_optional(summary.entropy_wasserstein_mean, reason=summary.note)}</td>"
      f"<td>{_fmt_optional(summary.composition_js_mean, reason=summary.note)}</td>"
      f"<td>{_fmt_optional(summary.identity_ks_pvalue_median, reason=summary.note)}</td>"
      "</tr>"
    )
    for summary in macro_summaries
  )
  sidechain_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{summary.path_id}</code></td>"
      f"<td><code>{summary.condition}</code></td>"
      f"<td>{summary.case_count}</td>"
      f"<td>{summary.pass_count}</td>"
      f"<td>{summary.warn_count}</td>"
      f"<td>{summary.fail_count}</td>"
      f"<td><code>{summary.status}</code></td>"
      f"<td>{summary.reason or '-'}</td>"
      "</tr>"
    )
    for summary in sidechain_gate_summaries
  )
  coverage_rows = "\n".join(
    (
      "<tr>"
      f"<td><code>{entry.path_id}</code></td>"
      f"<td><code>{entry.tier}</code></td>"
      f"<td><code>{entry.status}</code></td>"
      f"<td>{entry.reason}</td>"
      "</tr>"
    )
    for entry in coverage_entries
  )

  case_rows_html = case_rows if case_rows else "<tr><td colspan='8'>No evaluated case metadata rows</td></tr>"
  tied_lane_rows_html = tied_lane_rows if tied_lane_rows else "<tr><td colspan='6'>No tied lanes declared</td></tr>"
  evidence_rows_html = evidence_rows if evidence_rows else "<tr><td colspan='9'>No scalar evidence rows</td></tr>"
  intrinsic_rows_html = intrinsic_rows if intrinsic_rows else "<tr><td colspan='7'>No intrinsic rows</td></tr>"
  macro_rows_html = macro_rows if macro_rows else "<tr><td colspan='7'>No macro rows</td></tr>"
  sidechain_rows_html = (
    sidechain_rows if sidechain_rows else "<tr><td colspan='8'>No side-chain-conditioned gate rows</td></tr>"
  )

  lines = [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="utf-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1">',
    "  <title>Parity Assessment Report</title>",
    "  <style>body{font-family:system-ui,Arial,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;"
    "line-height:1.45}code{background:#f4f4f5;padding:.1rem .3rem;border-radius:4px}"
    "table{border-collapse:collapse;margin:1rem 0;width:100%}th,td{border:1px solid #ddd;padding:.45rem;text-align:left}"
    "th{background:#f8f8f8}figure{margin:0.8rem 0}img{max-width:100%;height:auto;margin:0.25rem 0}"
    "figcaption{font-size:.92rem;color:#3f3f46}</style>",
    "</head>",
    "<body>",
    "  <h1>Parity Assessment Report</h1>",
    f"  <p>Generated: <strong>{generated_at}</strong> (UTC)</p>",
    "  <h2>Executive parity snapshot</h2>",
    f"  <p><strong>Status:</strong> {snapshot.status}</p>",
    "  <h3>Key outcomes</h3>",
    "  <ul>",
  ]
  lines.extend(f"    <li>{outcome}</li>" for outcome in snapshot.outcomes)
  lines.extend(["  </ul>", "  <h3>Caveats</h3>", "  <ul>"])
  lines.extend(f"    <li>{caveat}</li>" for caveat in snapshot.caveats)
  lines.extend(
    [
      "  </ul>",
      "  <h2>How parity is assessed</h2>",
      "  <ol>",
      "    <li><code>parity_fast</code>: deterministic fixture and helper-branch parity checks (no upstream checkout required).</li>",
      "    <li><code>parity_heavy</code>: reference-backed numerical parity checks against pinned LigandMPNN.</li>",
      "    <li><code>parity_audit</code>: converted checkpoint-family load/audit checks across model families.</li>",
      "  </ol>",
      "  <h2>Terminology / interpretation legend</h2>",
      "  <table><thead><tr><th>Term</th><th>Interpretation</th></tr></thead><tbody>",
    ],
  )
  lines.extend(f"    <tr><td>{term}</td><td>{interpretation}</td></tr>" for term, interpretation in legend_rows)
  lines.extend(
    [
      "  </tbody></table>",
      "  <h2>Tier test outcomes (from JUnit)</h2>",
      "  <table><thead><tr><th>Tier</th><th>Tests</th><th>Passed</th><th>Skipped</th><th>Failed</th><th>Notes</th></tr></thead><tbody>",
      f"{result_rows}",
      "  </tbody></table>",
      _html_figure(
        alt="Tier-level JUnit pass skip fail outcomes",
        src="assets/parity_test_outcomes_by_tier.png",
        caption=(
          "Stacked JUnit pass/skip/fail totals per tier; this is the first release gate for triage."
        ),
      ).rstrip(),
      "  <h2>Converted family checkpoint coverage</h2>",
      "  <table><thead><tr><th>Family</th><th>Expected converted</th><th>Available locally</th></tr></thead><tbody>",
      f"{family_rows}",
      "  </tbody></table>",
      _html_figure(
        alt="Converted checkpoint availability by family",
        src="assets/parity_family_checkpoint_coverage.png",
        caption=(
          "Expected vs locally present converted checkpoints for parity_audit; missing assets block"
          " audit evidence collection."
        ),
      ).rstrip(),
      "  <h2>Expanded evidence summary</h2>",
      "  <h3>Evaluated protein systems and cases</h3>",
      "  <p>Case metadata below is extracted directly from collected evidence rows.</p>",
      "  <table><thead><tr><th>Case ID</th><th>Kind</th><th>Length(s)</th><th>Seed(s)</th><th>Backbone ID(s)</th><th>Checkpoint ID(s)</th><th>Tier(s)</th><th>Paths</th></tr></thead><tbody>",
      f"{case_rows_html}",
      "  </tbody></table>",
    ],
  )
  if tied_lanes:
    lines.extend(
      [
        "  <h3>Tied/multistate apples-to-apples lanes</h3>",
        "  <p>Configured tied/multistate lanes are listed beside evidence tables so unlike strategies never merge.</p>",
        "  <table><thead><tr><th>Condition</th><th>Comparison API</th><th>Reference combiner</th><th>JAX strategy</th><th>Token comparison</th><th>Primary</th></tr></thead><tbody>",
        f"{tied_lane_rows_html}",
        "  </tbody></table>",
      ],
    )

  if evidence_metrics_present:
    lines.extend(
      [
        f"  <p>Evidence metrics: <code>{evidence_metrics_rel_path}</code><br>Point samples: <code>{evidence_points_rel_path}</code></p>",
        "  <p>Condition labels isolate strategy lanes (for example, tied/multistate apples-to-apples comparisons).</p>",
        "  <table><thead><tr><th>Path</th><th>Condition</th><th>Cases</th><th>Mean r</th><th>Min r</th><th>Pearson pass rate</th><th>Mean MAE</th><th>Mean RMSE</th><th>Mean token agreement</th></tr></thead><tbody>",
        f"{evidence_rows_html}",
        "  </tbody></table>",
        _html_figure(
          alt="Pearson swarm by path and condition lanes",
          src="assets/parity_correlation_by_path.png",
          caption=(
            f"Pearson swarm across lane-level case rows (n={pearson_sample_count}). Each point is one"
            " path/condition/case/seed value, black ticks mark medians, and the dashed line marks"
            " the r=0.95 policy floor."
          ),
        ).rstrip(),
        _html_figure(
          alt="Mean MAE and RMSE by parity lane",
          src="assets/parity_error_by_path.png",
          caption=(
            f"Mean MAE and RMSE across {scalar_lane_count} scalar-metric lanes. Rows shown as"
            " n/a(...) are pending, excluded, or categorical by design."
          ),
        ).rstrip(),
        _html_figure(
          alt="Reference versus JAX scatter diagnostics for core parity paths",
          src="assets/parity_scatter_core_paths.png",
          caption=(
            "Reference vs JAX scatter samples for core paths; tighter diagonal alignment indicates"
            " stronger parity. Dense clouds may be downsampled for readability."
          ),
        ).rstrip(),
        _html_figure(
          alt="Bland Altman drift diagnostics for decoder conditional scoring",
          src="assets/parity_bland_altman_decoder_conditional.png",
          caption=(
            "Bland-Altman view for decoder-conditional-scoring. The center line is mean drift and"
            " dotted lines show ±1.96 SD."
          ),
        ).rstrip(),
        _html_figure(
          alt="Binary evidence coverage heatmap with present versus absent semantics",
          src="assets/parity_case_coverage_heatmap.png",
          caption=(
            f"Binary coverage matrix ({coverage_path_count} paths × {coverage_case_count} cases):"
            " green=present (>=1 evidence row), gray=absent. Color intensity does not encode"
            " magnitude."
          ),
        ).rstrip(),
        "  <h3>Baseline-aware intrinsic parity</h3>",
        "  <table><thead><tr><th>Path</th><th>Condition</th><th>Cases</th><th>Mean cross/intrinsic ratio</th><th>Max ratio</th><th>Pass@95 envelope</th><th>Pass@99 envelope</th></tr></thead><tbody>",
        f"{intrinsic_rows_html}",
        "  </tbody></table>",
        _html_figure(
          alt="Log scale intrinsic ratio by path and condition lanes",
          src="assets/parity_intrinsic_ratio_by_path.png",
          caption=(
            f"Log-scale cross/intrinsic ratio chart across {intrinsic_lane_count} lane(s). The"
            " denominator is pooled intrinsic MAE, so near-zero baselines can inflate ratios."
          ),
        ).rstrip(),
        "  <h3>Macro distribution parity</h3>",
        "  <table><thead><tr><th>Path</th><th>Condition</th><th>Cases</th><th>Mean identity Wasserstein</th><th>Mean entropy Wasserstein</th><th>Mean composition JS</th><th>Median identity KS p-value</th></tr></thead><tbody>",
        f"{macro_rows_html}",
        "  </tbody></table>",
        "  <h3>Side-chain-conditioned integration gate</h3>",
        "  <table><thead><tr><th>Path</th><th>Condition</th><th>Cases</th><th>Pass</th><th>Warn</th><th>Fail</th><th>Outcome</th><th>Reason</th></tr></thead><tbody>",
        f"{sidechain_rows_html}",
        "  </tbody></table>",
        _html_figure(
          alt="Macro distribution swarm comparing reference and JAX",
          src="assets/parity_macro_swarm.png",
          caption=(
            "Macro swarm compares reference vs JAX identity/entropy points by condition. Overlap"
            " implies better distributional parity; excluded lanes remain n/a(...)."
          ),
        ).rstrip(),
        "  <h3>Coverage and exclusions</h3>",
        "  <table><thead><tr><th>Path</th><th>Tier</th><th>Status</th><th>Reason</th></tr></thead><tbody>",
        f"{coverage_rows}",
        "  </tbody></table>",
      ],
    )
  else:
    lines.append(
      "  <p>No evidence metrics CSV found. Run <code>scripts/collect_parity_evidence.py</code> first.</p>",
    )

  lines.extend(
    [
      "  <h2>Appendix: path inventory and definitions</h2>",
      "  <h3>Parity path definitions</h3>",
      "  <table><thead><tr><th>Path</th><th>Tier</th><th>Method</th><th>Metrics</th><th>Code paths</th></tr></thead><tbody>",
      f"{matrix_rows}",
      "  </tbody></table>",
      "  <h3>Tier path inventory</h3>",
      "  <table><thead><tr><th>Tier</th><th>Path count</th></tr></thead><tbody>",
      f"{tier_rows}",
      "  </tbody></table>",
      _html_figure(
        alt="Configured parity path counts by tier",
        src="assets/parity_paths_by_tier.png",
        caption=(
          "Inventory-only path counts by tier. Larger bars indicate broader instrumentation,"
          " not stronger parity."
        ),
      ).rstrip(),
      f"  <p>Checkpoint checksum snapshot: <code>{checksum_rel_path}</code></p>",
      "</body>",
      "</html>",
    ],
  )
  return "\n".join(lines) + "\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--project-root", default=".", help="Repository root path.")
  parser.add_argument(
    "--output-dir",
    default="docs/parity",
    help="Output directory for markdown/html report and assets.",
  )
  parser.add_argument("--fast-junit", default="docs/parity/reports/parity_fast.xml")
  parser.add_argument("--heavy-junit", default="docs/parity/reports/parity_heavy.xml")
  parser.add_argument("--audit-junit", default="docs/parity/reports/parity_audit.xml")
  parser.add_argument(
    "--evidence-metrics",
    default="docs/parity/reports/evidence/evidence_metrics.csv",
    help="Parity evidence metrics CSV.",
  )
  parser.add_argument(
    "--evidence-points",
    default="docs/parity/reports/evidence/evidence_points.csv",
    help="Parity evidence point CSV.",
  )
  parser.add_argument("--pdf", action="store_true", help="Also export parity_report.pdf from HTML.")
  args = parser.parse_args()

  project_root = Path(args.project_root).resolve()
  output_dir = (project_root / args.output_dir).resolve()
  assets_dir = output_dir / "assets"
  reports_dir = output_dir / "reports"
  output_dir.mkdir(parents=True, exist_ok=True)
  assets_dir.mkdir(parents=True, exist_ok=True)
  reports_dir.mkdir(parents=True, exist_ok=True)

  matrix_payload = json.loads((project_root / "tests/parity/parity_matrix.json").read_text(encoding="utf-8"))
  matrix_paths = matrix_payload.get("paths", [])
  if not isinstance(matrix_paths, list):
    msg = "parity_matrix.json has invalid `paths` payload."
    raise TypeError(msg)

  path_counts: dict[str, int] = {"parity_fast": 0, "parity_heavy": 0, "parity_audit": 0}
  for item in matrix_paths:
    if not isinstance(item, dict):
      continue
    tier = item.get("tier")
    if isinstance(tier, str) and tier in path_counts:
      path_counts[tier] += 1

  assets_payload = json.loads((project_root / "tests/parity/parity_assets.json").read_text(encoding="utf-8"))
  assets = assets_payload.get("assets", [])
  if not isinstance(assets, list):
    msg = "parity_assets.json has invalid `assets` payload."
    raise TypeError(msg)

  family_expected: dict[str, int] = {}
  family_available: dict[str, int] = {}
  for asset in assets:
    if not isinstance(asset, dict):
      continue
    if asset.get("asset_kind") != "converted_checkpoint":
      continue
    required_for = asset.get("required_for")
    if not isinstance(required_for, list) or "parity_audit" not in required_for:
      continue
    family = asset.get("family")
    rel_path = asset.get("path")
    if not isinstance(family, str) or not isinstance(rel_path, str):
      continue
    family_expected[family] = family_expected.get(family, 0) + 1
    target = project_root / rel_path
    if target.exists():
      family_available[family] = family_available.get(family, 0) + 1

  checksum_path = reports_dir / "converted_checkpoint_checksums.txt"
  _write_checkpoint_checksums(
    project_root=project_root,
    assets=assets,
    output_path=checksum_path,
  )

  tier_results = [
    _parse_junit(project_root / args.fast_junit, "parity_fast"),
    _parse_junit(project_root / args.heavy_junit, "parity_heavy"),
    _parse_junit(project_root / args.audit_junit, "parity_audit"),
  ]

  evidence_metrics_path = (project_root / args.evidence_metrics).resolve()
  evidence_points_path = (project_root / args.evidence_points).resolve()
  evidence_metric_rows = _load_evidence_metrics(evidence_metrics_path)
  evidence_point_rows = _load_evidence_points(evidence_points_path)
  evidence_summaries = _aggregate_evidence(
    [path for path in matrix_paths if isinstance(path, dict)],
    evidence_metric_rows,
  )
  tied_lane_descriptors = _extract_tied_lane_descriptors(
    [path for path in matrix_paths if isinstance(path, dict)],
  )
  correlation_points = _collect_case_pearson_points(evidence_metric_rows)
  case_inventory_entries = _aggregate_case_inventory(evidence_metric_rows, evidence_point_rows)
  intrinsic_summaries = _aggregate_intrinsic(evidence_metric_rows)
  macro_summaries = _aggregate_macro(evidence_metric_rows)
  sidechain_gate_summaries = _aggregate_sidechain_gate(evidence_metric_rows)
  coverage_entries = _build_coverage_entries(
    [path for path in matrix_paths if isinstance(path, dict)],
    evidence_summaries,
  )
  evidence_metrics_present = bool(evidence_metric_rows)

  _plot_path_counts(path_counts, assets_dir / "parity_paths_by_tier.png")
  _plot_test_status(tier_results, assets_dir / "parity_test_outcomes_by_tier.png")
  _plot_family_coverage(family_expected, family_available, assets_dir / "parity_family_checkpoint_coverage.png")
  _plot_correlation_by_path(correlation_points, assets_dir / "parity_correlation_by_path.png")
  _plot_error_by_path(evidence_summaries, assets_dir / "parity_error_by_path.png")
  _plot_intrinsic_ratio_by_path(intrinsic_summaries, assets_dir / "parity_intrinsic_ratio_by_path.png")
  _plot_macro_swarm(evidence_point_rows, assets_dir / "parity_macro_swarm.png")
  _plot_scatter_grid(evidence_point_rows, assets_dir / "parity_scatter_core_paths.png")
  _plot_bland_altman(evidence_point_rows, assets_dir / "parity_bland_altman_decoder_conditional.png")
  _plot_case_coverage_heatmap(
    [path for path in matrix_paths if isinstance(path, dict)],
    evidence_metric_rows,
    assets_dir / "parity_case_coverage_heatmap.png",
  )

  generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
  checksum_rel_path = "reports/converted_checkpoint_checksums.txt"
  markdown = _render_markdown(
    generated_at=generated_at,
    path_counts=path_counts,
    tier_results=tier_results,
    family_expected=family_expected,
    family_available=family_available,
    checksum_rel_path=checksum_rel_path,
    matrix_paths=[path for path in matrix_paths if isinstance(path, dict)],
    evidence_summaries=evidence_summaries,
    case_inventory_entries=case_inventory_entries,
    intrinsic_summaries=intrinsic_summaries,
    macro_summaries=macro_summaries,
    sidechain_gate_summaries=sidechain_gate_summaries,
    coverage_entries=coverage_entries,
    evidence_metrics_rel_path=str(evidence_metrics_path.relative_to(project_root)),
    evidence_points_rel_path=str(evidence_points_path.relative_to(project_root)),
    evidence_metrics_present=evidence_metrics_present,
    tied_lane_descriptors=tied_lane_descriptors,
  )
  html = _render_html(
    generated_at=generated_at,
    path_counts=path_counts,
    tier_results=tier_results,
    family_expected=family_expected,
    family_available=family_available,
    checksum_rel_path=checksum_rel_path,
    matrix_paths=[path for path in matrix_paths if isinstance(path, dict)],
    evidence_summaries=evidence_summaries,
    case_inventory_entries=case_inventory_entries,
    intrinsic_summaries=intrinsic_summaries,
    macro_summaries=macro_summaries,
    sidechain_gate_summaries=sidechain_gate_summaries,
    coverage_entries=coverage_entries,
    evidence_metrics_rel_path=str(evidence_metrics_path.relative_to(project_root)),
    evidence_points_rel_path=str(evidence_points_path.relative_to(project_root)),
    evidence_metrics_present=evidence_metrics_present,
    tied_lane_descriptors=tied_lane_descriptors,
  )

  markdown_path = output_dir / "parity_report.md"
  html_path = output_dir / "parity_report.html"
  markdown_path.write_text(markdown + "\n", encoding="utf-8")
  html_path.write_text(html, encoding="utf-8")
  print(f"Wrote {markdown_path}")
  print(f"Wrote {html_path}")
  print(f"Wrote {assets_dir}")
  print(f"Wrote {checksum_path}")

  if args.pdf:
    pdf_path = output_dir / "parity_report.pdf"
    pdf_script = project_root / "scripts" / "export_parity_report_pdf.py"
    command = [
      sys.executable,
      str(pdf_script),
      "--html",
      str(html_path),
      "--pdf",
      str(pdf_path),
    ]
    subprocess.run(command, check=True)  # noqa: S603
    print(f"Wrote {pdf_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
