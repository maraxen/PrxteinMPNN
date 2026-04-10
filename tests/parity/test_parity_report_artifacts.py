"""Tests for parity report rendering helpers."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATE_MODULE = runpy.run_path(str(ROOT / "scripts" / "generate_parity_report.py"))
EXPORT_MODULE = runpy.run_path(str(ROOT / "scripts" / "export_parity_report_pdf.py"))

EvidenceSummary = GENERATE_MODULE["EvidenceSummary"]
CaseInventoryEntry = GENERATE_MODULE["CaseInventoryEntry"]
SidechainGateSummary = GENERATE_MODULE["SidechainGateSummary"]
TiedLaneDescriptor = GENERATE_MODULE["TiedLaneDescriptor"]
TierResult = GENERATE_MODULE["TierResult"]
render_markdown = GENERATE_MODULE["_render_markdown"]
render_html = GENERATE_MODULE["_render_html"]
aggregate_sidechain_gate = GENERATE_MODULE["_aggregate_sidechain_gate"]
aggregate_evidence = GENERATE_MODULE["_aggregate_evidence"]
collect_case_pearson_points = GENERATE_MODULE["_collect_case_pearson_points"]
export_html_to_pdf = EXPORT_MODULE["export_html_to_pdf"]


def test_render_markdown_includes_expanded_evidence() -> None:
  """Markdown renderer includes path definitions and expanded evidence sections."""
  markdown = render_markdown(
    generated_at="2026-01-01T00:00:00+00:00",
    path_counts={"parity_fast": 3, "parity_heavy": 9, "parity_audit": 1},
    tier_results=[
      TierResult(name="parity_fast", tests=5, passed=5, failed=0, skipped=0),
      TierResult(name="parity_heavy", tests=10, passed=10, failed=0, skipped=0),
      TierResult(name="parity_audit", tests=6, passed=6, failed=0, skipped=0),
    ],
    family_expected={"protein": 4, "ligand": 4},
    family_available={"protein": 4, "ligand": 4},
    checksum_rel_path="reports/converted_checkpoint_checksums.txt",
    matrix_paths=[
      {
        "id": "decoder-conditional-scoring",
        "tier": "parity_heavy",
        "method": "logits and score parity",
        "metrics": ["pearson correlation", "nll drift"],
        "code_paths": ["src/prxteinmpnn/model/mpnn.py"],
      },
    ],
    evidence_summaries=[
      EvidenceSummary(
        path_id="decoder-conditional-scoring",
        case_count=5,
        pearson_mean=0.99,
        pearson_min=0.97,
        pearson_pass_rate=1.0,
        mae_mean=0.01,
        rmse_mean=0.02,
        token_agreement_mean=0.98,
        condition="reference_weighted_sum__jax_product",
      ),
    ],
    case_inventory_entries=[
      CaseInventoryEntry(
        case_id="1ubq",
        case_kind="real_backbone",
        sequence_lengths=(76,),
        seeds=(388,),
        backbone_ids=("1ubq",),
        checkpoint_ids=("proteinmpnn_v_48_020",),
        tiers=("parity_heavy",),
        path_count=4,
      ),
    ],
    intrinsic_summaries=[],
    macro_summaries=[],
    sidechain_gate_summaries=[
      SidechainGateSummary(
        path_id="ligand-autoregressive",
        condition="side_chain_conditioned",
        case_count=2,
        pass_count=0,
        warn_count=2,
        fail_count=0,
        status="warn",
        reason="Excluded by parity case corpus configuration.",
      ),
    ],
    coverage_entries=[],
    evidence_metrics_rel_path="docs/parity/reports/evidence/evidence_metrics.csv",
    evidence_points_rel_path="docs/parity/reports/evidence/evidence_points.csv",
    evidence_metrics_present=True,
  )
  assert "## Executive parity snapshot" in markdown
  assert "## Terminology / interpretation legend" in markdown
  assert "## Expanded evidence summary" in markdown
  assert "### Evaluated protein systems and cases" in markdown
  assert "## Appendix: path inventory and definitions" in markdown
  assert "### Parity path definitions" in markdown
  assert "`1ubq` | `real_backbone` | 76 | 388" in markdown
  assert "Pearson swarm across lane-level case rows" in markdown
  assert "Binary coverage matrix (" in markdown
  assert "Log-scale cross/intrinsic ratio chart" in markdown
  assert "decoder-conditional-scoring" in markdown
  assert "reference_weighted_sum__jax_product" in markdown
  assert "parity_scatter_core_paths.png" in markdown
  assert "### Side-chain-conditioned integration gate" in markdown
  assert "`warn`" in markdown
  assert markdown.index("## Expanded evidence summary") < markdown.index(
    "## Appendix: path inventory and definitions",
  )


def test_render_markdown_includes_tied_lane_alignment_section() -> None:
  """Markdown renderer surfaces tied/multistate lane metadata explicitly."""
  markdown = render_markdown(
    generated_at="2026-01-01T00:00:00+00:00",
    path_counts={"parity_fast": 1, "parity_heavy": 1, "parity_audit": 0},
    tier_results=[],
    family_expected={},
    family_available={},
    checksum_rel_path="reports/converted_checkpoint_checksums.txt",
    matrix_paths=[],
    evidence_summaries=[],
    case_inventory_entries=[],
    intrinsic_summaries=[],
    macro_summaries=[],
    sidechain_gate_summaries=[],
    coverage_entries=[],
    evidence_metrics_rel_path="docs/parity/reports/evidence/evidence_metrics.csv",
    evidence_points_rel_path="docs/parity/reports/evidence/evidence_points.csv",
    evidence_metrics_present=False,
    tied_lane_descriptors=[
      TiedLaneDescriptor(
        path_id="tied-positions-and-multi-state",
        input_context="ligand_context",
        condition="reference_weighted_sum__jax_product",
        comparison_api="sampling",
        reference_combiner="weighted_sum",
        jax_multi_state_strategy="product",
        token_comparison_enabled=True,
        is_primary=True,
        rollout_policy="parity_heavy=warn; parity_audit=fail",
      ),
    ],
  )
  assert "### Tied/multistate apples-to-apples lanes" in markdown
  assert "`tied-positions-and-multi-state`" in markdown
  assert "`ligand_context`" in markdown
  assert "`reference_weighted_sum__jax_product`" in markdown
  assert "`sampling`" in markdown
  assert "`product`" in markdown
  assert "`parity_heavy=warn; parity_audit=fail`" in markdown
  tied_index = markdown.index("### Tied/multistate apples-to-apples lanes")
  assert tied_index > markdown.index("## Expanded evidence summary")
  assert tied_index < markdown.index("## Appendix: path inventory and definitions")


def test_aggregate_sidechain_gate_rolls_up_status_counts() -> None:
  """Aggregator returns pass/warn/fail counts with fail precedence."""
  summaries = aggregate_sidechain_gate([
    {
      "path_id": "ligand-autoregressive",
      "condition": "side_chain_conditioned",
      "case_id": "c1",
      "case_kind": "synthetic",
      "metric_group": "macro",
      "metric_name": "macro_sidechain_conditioned_gate_status_code",
      "metric_value": 0.0,
      "note": "pass",
    },
    {
      "path_id": "ligand-autoregressive",
      "condition": "side_chain_conditioned",
      "case_id": "c2",
      "case_kind": "synthetic",
      "metric_group": "macro",
      "metric_name": "macro_sidechain_conditioned_gate_status_code",
      "metric_value": 1.0,
      "note": "warn",
    },
    {
      "path_id": "ligand-autoregressive",
      "condition": "side_chain_conditioned",
      "case_id": "c3",
      "case_kind": "synthetic",
      "metric_group": "macro",
      "metric_name": "macro_sidechain_conditioned_gate_status_code",
      "metric_value": 2.0,
      "note": "fail",
    },
  ])
  assert len(summaries) == 1
  summary = summaries[0]
  assert summary.pass_count == 1
  assert summary.warn_count == 1
  assert summary.fail_count == 1
  assert summary.status == "fail"


def test_aggregate_evidence_separates_condition_lanes() -> None:
  """Scalar evidence summaries keep same-path condition lanes separate."""
  summaries = aggregate_evidence(
    [{"id": "tied-positions-and-multi-state"}],
    [
      {
        "path_id": "tied-positions-and-multi-state",
        "condition": "lane-a",
        "case_id": "case-1",
        "case_kind": "synthetic",
        "metric_name": "pearson_correlation",
        "metric_value": 0.99,
        "passed": True,
      },
      {
        "path_id": "tied-positions-and-multi-state",
        "condition": "lane-a",
        "case_id": "case-1",
        "case_kind": "synthetic",
        "metric_name": "mae",
        "metric_value": 0.01,
      },
      {
        "path_id": "tied-positions-and-multi-state",
        "condition": "lane-b",
        "case_id": "case-1",
        "case_kind": "synthetic",
        "metric_name": "pearson_correlation",
        "metric_value": 0.55,
        "passed": False,
      },
      {
        "path_id": "tied-positions-and-multi-state",
        "condition": "lane-b",
        "case_id": "case-1",
        "case_kind": "synthetic",
        "metric_name": "mae",
        "metric_value": 0.25,
      },
    ],
  )
  assert len(summaries) == 2
  by_condition = {summary.condition: summary for summary in summaries}
  assert by_condition["lane-a"].pearson_mean == pytest.approx(0.99)
  assert by_condition["lane-a"].mae_mean == pytest.approx(0.01)
  assert by_condition["lane-b"].pearson_mean == pytest.approx(0.55)
  assert by_condition["lane-b"].mae_mean == pytest.approx(0.25)


def test_collect_case_pearson_points_separates_condition_lanes() -> None:
  """Pearson swarm points keep condition lanes separate for the same case/seed."""
  points = collect_case_pearson_points([
    {
      "path_id": "tied-positions-and-multi-state",
      "condition": "lane-a",
      "case_id": "case-1",
      "case_kind": "synthetic",
      "seed": 7,
      "metric_name": "pearson_correlation",
      "metric_value": 0.99,
    },
    {
      "path_id": "tied-positions-and-multi-state",
      "condition": "lane-b",
      "case_id": "case-1",
      "case_kind": "synthetic",
      "seed": 7,
      "metric_name": "pearson_correlation",
      "metric_value": 0.44,
    },
  ])
  assert len(points) == 2
  by_condition = {point.condition: point for point in points}
  assert by_condition["lane-a"].value == pytest.approx(0.99)
  assert by_condition["lane-b"].value == pytest.approx(0.44)


def test_export_html_to_pdf_requires_existing_html(tmp_path: Path) -> None:
  """PDF exporter raises a clear error when input HTML is missing."""
  missing_html = tmp_path / "missing.html"
  target_pdf = tmp_path / "report.pdf"
  with pytest.raises(FileNotFoundError):
    export_html_to_pdf(html_path=missing_html, pdf_path=target_pdf)


def test_render_html_includes_case_inventory_and_figure_captions() -> None:
  """HTML renderer includes evaluated cases section and descriptive figure captions."""
  html = render_html(
    generated_at="2026-01-01T00:00:00+00:00",
    path_counts={"parity_fast": 1, "parity_heavy": 1, "parity_audit": 0},
    tier_results=[TierResult(name="parity_heavy", tests=1, passed=1, failed=0, skipped=0)],
    family_expected={"protein": 1},
    family_available={"protein": 1},
    checksum_rel_path="reports/converted_checkpoint_checksums.txt",
    matrix_paths=[
      {
        "id": "decoder-conditional-scoring",
        "tier": "parity_heavy",
        "method": "logits and score parity",
        "metrics": ["pearson correlation"],
        "code_paths": ["src/prxteinmpnn/model/mpnn.py"],
      },
    ],
    evidence_summaries=[
      EvidenceSummary(
        path_id="decoder-conditional-scoring",
        case_count=1,
        pearson_mean=0.99,
        pearson_min=0.99,
        pearson_pass_rate=1.0,
        mae_mean=0.01,
        rmse_mean=0.02,
        token_agreement_mean=0.98,
        condition="reference_weighted_sum__jax_product",
      ),
    ],
    case_inventory_entries=[
      CaseInventoryEntry(
        case_id="1ubq",
        case_kind="real_backbone",
        sequence_lengths=(76,),
        seeds=(388,),
        backbone_ids=("1ubq",),
        checkpoint_ids=("proteinmpnn_v_48_020",),
        tiers=("parity_heavy",),
        path_count=1,
      ),
    ],
    intrinsic_summaries=[],
    macro_summaries=[],
    sidechain_gate_summaries=[],
    coverage_entries=[],
    evidence_metrics_rel_path="docs/parity/reports/evidence/evidence_metrics.csv",
    evidence_points_rel_path="docs/parity/reports/evidence/evidence_points.csv",
    evidence_metrics_present=True,
  )
  assert "<h2>Executive parity snapshot</h2>" in html
  assert "<h2>Terminology / interpretation legend</h2>" in html
  assert "<h3>Evaluated protein systems and cases</h3>" in html
  assert "<figcaption>Pearson swarm across lane-level case rows" in html
  assert "<figcaption>Binary coverage matrix (" in html
  assert "<h2>Appendix: path inventory and definitions</h2>" in html
  assert "<code>1ubq</code>" in html
  assert "<code>reference_weighted_sum__jax_product</code>" in html
