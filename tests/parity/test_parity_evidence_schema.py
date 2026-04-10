"""Unit tests for parity evidence schema and metric helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from prxteinmpnn.parity.evidence import (
  amino_acid_distribution,
  EvidenceMetricRecord,
  EvidencePointRecord,
  downsample_pair_points,
  fisher_pearson_ci,
  max_abs_error,
  mean_abs_error,
  mean_kl_divergence,
  per_position_entropy,
  root_mean_square_error,
  safe_pearson,
  sequence_identity,
  token_agreement,
  write_metric_records_csv,
  write_metric_records_json,
  write_point_records_csv,
)


def test_safe_pearson_and_confidence_interval() -> None:
  """Correlation helper returns stable values and bounded confidence intervals."""
  x = np.linspace(0.0, 1.0, 128, dtype=np.float64)
  y = x + 0.01
  corr = safe_pearson(x, y)
  assert 0.99 <= corr <= 1.0
  ci = fisher_pearson_ci(corr, sample_size=x.size)
  assert ci is not None
  assert -1.0 <= ci[0] <= ci[1] <= 1.0


def test_error_metrics_and_kl() -> None:
  """Error helpers compute non-negative values with expected ordering."""
  reference = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
  observed = reference + 0.05
  mae = mean_abs_error(reference, observed)
  rmse = root_mean_square_error(reference, observed)
  max_abs = max_abs_error(reference, observed)
  assert mae > 0.0
  assert rmse + 1e-12 >= mae
  assert max_abs >= rmse

  ref_log_probs = np.log(np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64))
  obs_log_probs = np.log(np.array([[0.6, 0.3, 0.1], [0.25, 0.25, 0.5]], dtype=np.float64))
  kl = mean_kl_divergence(ref_log_probs, obs_log_probs)
  assert kl >= 0.0


def test_token_agreement_and_pair_downsampling() -> None:
  """Token agreement respects masks and point downsampling is deterministic."""
  lhs = np.array([1, 2, 3, 4, 5], dtype=np.int32)
  rhs = np.array([1, 0, 3, 0, 5], dtype=np.int32)
  mask = np.array([1, 0, 1, 1, 1], dtype=np.int32)
  agreement = token_agreement(lhs, rhs, mask)
  assert agreement == 0.75

  x = np.arange(200, dtype=np.float64)
  y = x + 1.0
  sx1, sy1 = downsample_pair_points(x, y, max_points=50, seed=7)
  sx2, sy2 = downsample_pair_points(x, y, max_points=50, seed=7)
  assert sx1.shape == (50,)
  assert sy1.shape == (50,)
  np.testing.assert_array_equal(sx1, sx2)
  np.testing.assert_array_equal(sy1, sy2)


def test_sequence_macro_helpers() -> None:
  """Sequence-level helper metrics return finite values with expected ranges."""
  sampled = np.array([[1, 2, 3], [1, 2, 4]], dtype=np.int32)
  reference = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
  mask = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.int32)
  identity = sequence_identity(sampled, reference, mask)
  assert 0.0 <= identity <= 1.0

  log_probs = np.log(np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], dtype=np.float64))
  entropy = per_position_entropy(log_probs, np.array([1, 1], dtype=np.int32))
  assert entropy > 0.0

  dist = amino_acid_distribution(sampled, num_amino_acids=5, mask=mask)
  assert dist.shape == (5,)
  np.testing.assert_allclose(np.sum(dist), 1.0, atol=1e-8)


def test_evidence_record_writers(tmp_path: Path) -> None:
  """Evidence writers emit parseable CSV and JSON payloads."""
  metrics_path = tmp_path / "metrics.csv"
  points_path = tmp_path / "points.csv"
  metrics_json_path = tmp_path / "metrics.json"

  metric_records = [
    EvidenceMetricRecord(
      path_id="decoder-conditional-scoring",
      tier="parity_heavy",
      case_id="seed-7-len-12",
      case_kind="synthetic",
      backbone_id="synthetic",
      seed=7,
      sequence_length=12,
      checkpoint_id="proteinmpnn_v_48_020",
      metric_name="pearson_correlation",
      metric_value=0.99,
      threshold=0.95,
      passed=True,
    ),
  ]
  point_records = [
    EvidencePointRecord(
      path_id="decoder-conditional-scoring",
      tier="parity_heavy",
      case_id="seed-7-len-12",
      case_kind="synthetic",
      backbone_id="synthetic",
      seed=7,
      sequence_length=12,
      reference_value=0.2,
      observed_value=0.19,
    ),
  ]

  write_metric_records_csv(metric_records, metrics_path)
  write_metric_records_json(metric_records, metrics_json_path)
  write_point_records_csv(point_records, points_path)

  with metrics_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  assert len(rows) == 1
  assert rows[0]["path_id"] == "decoder-conditional-scoring"

  payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
  assert isinstance(payload, list)
  assert payload[0]["metric_name"] == "pearson_correlation"

  with points_path.open(newline="", encoding="utf-8") as handle:
    point_rows = list(csv.DictReader(handle))
  assert len(point_rows) == 1
  assert point_rows[0]["case_id"] == "seed-7-len-12"
