"""Utilities and schemas for parity evidence collection and reporting."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class EvidenceMetricRecord:
  """One scalar metric observation for a parity path and test case."""

  path_id: str
  tier: str
  case_id: str
  case_kind: str
  backbone_id: str
  seed: int
  sequence_length: int
  checkpoint_id: str
  metric_name: str
  metric_value: float
  threshold: float | None = None
  passed: bool | None = None
  metric_group: str | None = None
  condition: str | None = None
  note: str | None = None


@dataclass(frozen=True, slots=True)
class EvidencePointRecord:
  """One pointwise parity sample used by scatter/residual diagnostics."""

  path_id: str
  tier: str
  case_id: str
  case_kind: str
  backbone_id: str
  seed: int
  sequence_length: int
  reference_value: float
  observed_value: float
  point_kind: str | None = None
  condition: str | None = None


def safe_pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute Pearson correlation for flattened arrays with constant-array safeguards."""
  lhs_flat = np.asarray(lhs, dtype=np.float64).ravel()
  rhs_flat = np.asarray(rhs, dtype=np.float64).ravel()
  if lhs_flat.shape != rhs_flat.shape:
    msg = "Pearson inputs must have matching flattened shapes."
    raise ValueError(msg)
  if lhs_flat.size == 0:
    msg = "Pearson inputs must be non-empty."
    raise ValueError(msg)

  lhs_std = float(lhs_flat.std())
  rhs_std = float(rhs_flat.std())
  if lhs_std == 0.0 or rhs_std == 0.0:
    return 1.0 if np.allclose(lhs_flat, rhs_flat) else 0.0
  corr = float(np.corrcoef(lhs_flat, rhs_flat)[0, 1])
  return float(np.clip(corr, -1.0, 1.0))


def fisher_pearson_ci(
  correlation: float,
  sample_size: int,
  *,
  confidence: float = 0.95,
) -> tuple[float, float] | None:
  """Approximate Pearson confidence interval with Fisher z transform."""
  if sample_size <= 3:
    return None
  if not 0.0 < confidence < 1.0:
    msg = "Confidence must be in (0, 1)."
    raise ValueError(msg)

  # Avoid infinities at ±1.
  clipped = float(np.clip(correlation, -0.999999, 0.999999))
  z_value = math.atanh(clipped)
  stderr = 1.0 / math.sqrt(sample_size - 3)
  alpha = 1.0 - confidence
  z_crit = (
    1.959963984540054 if math.isclose(alpha, 0.05, rel_tol=0.0) else _normal_ppf(1.0 - alpha / 2.0)
  )
  lower = math.tanh(z_value - z_crit * stderr)
  upper = math.tanh(z_value + z_crit * stderr)
  return (float(lower), float(upper))


def _normal_ppf(probability: float) -> float:
  """Approximate inverse standard normal CDF with Acklam's rational approximation."""
  if probability <= 0.0 or probability >= 1.0:
    msg = "Probability must be in (0, 1)."
    raise ValueError(msg)

  # Coefficients from Peter J. Acklam's approximation.
  a = [
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
  ]
  b = [
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
  ]
  c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
  ]
  d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
  ]
  p_low = 0.02425
  p_high = 1.0 - p_low

  if probability < p_low:
    q = math.sqrt(-2.0 * math.log(probability))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
      (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )
  if probability > p_high:
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )
  q = probability - 0.5
  r = q * q
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
    ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
  )


def mean_abs_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute mean absolute error."""
  return float(
    np.mean(np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64))),
  )


def root_mean_square_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute root mean square error."""
  delta = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
  return float(np.sqrt(np.mean(delta * delta)))


def max_abs_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute maximum absolute error."""
  return float(
    np.max(np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64))),
  )


def mean_kl_divergence(reference_log_probs: np.ndarray, observed_log_probs: np.ndarray) -> float:
  """Compute mean per-row KL divergence KL(reference || observed)."""
  reference = np.asarray(reference_log_probs, dtype=np.float64)
  observed = np.asarray(observed_log_probs, dtype=np.float64)
  if reference.shape != observed.shape:
    msg = "KL inputs must have matching shapes."
    raise ValueError(msg)
  reference_probs = np.exp(reference - np.max(reference, axis=-1, keepdims=True))
  reference_probs = reference_probs / np.sum(reference_probs, axis=-1, keepdims=True)
  observed_probs = np.exp(observed - np.max(observed, axis=-1, keepdims=True))
  observed_probs = observed_probs / np.sum(observed_probs, axis=-1, keepdims=True)

  eps = 1e-12
  log_ratio = np.log(reference_probs + eps) - np.log(observed_probs + eps)
  kl = np.sum(reference_probs * log_ratio, axis=-1)
  return float(np.mean(kl))


def token_agreement(
  lhs_tokens: np.ndarray,
  rhs_tokens: np.ndarray,
  mask: np.ndarray | None = None,
) -> float:
  """Compute token-level agreement under an optional binary mask."""
  lhs = np.asarray(lhs_tokens)
  rhs = np.asarray(rhs_tokens)
  if lhs.shape != rhs.shape:
    msg = "Token agreement inputs must have matching shapes."
    raise ValueError(msg)
  if mask is None:
    return float(np.mean(lhs == rhs))
  valid = np.asarray(mask, dtype=bool)
  if valid.shape != lhs.shape:
    msg = "Mask shape must match token shapes."
    raise ValueError(msg)
  if not np.any(valid):
    return 0.0
  return float(np.mean(lhs[valid] == rhs[valid]))


def sequence_identity(
  sampled_tokens: np.ndarray,
  reference_tokens: np.ndarray,
  mask: np.ndarray | None = None,
) -> float:
  """Compute sequence identity against a reference sequence."""
  sampled = np.asarray(sampled_tokens)
  reference = np.asarray(reference_tokens)
  if sampled.shape != reference.shape:
    msg = "Sequence identity inputs must have matching shapes."
    raise ValueError(msg)
  if mask is None:
    return float(np.mean(sampled == reference))
  valid = np.asarray(mask, dtype=bool)
  if valid.shape != sampled.shape:
    msg = "Sequence identity mask shape must match token shapes."
    raise ValueError(msg)
  if not np.any(valid):
    return 0.0
  return float(np.mean(sampled[valid] == reference[valid]))


def per_position_entropy(log_probs: np.ndarray, mask: np.ndarray | None = None) -> float:
  """Compute mean categorical entropy from log-probabilities."""
  values = np.asarray(log_probs, dtype=np.float64)
  if values.ndim != 2:
    msg = "Entropy expects rank-2 log-probabilities with shape (N, C)."
    raise ValueError(msg)
  probs = np.exp(values - np.max(values, axis=-1, keepdims=True))
  probs = probs / np.sum(probs, axis=-1, keepdims=True)
  entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
  if mask is None:
    return float(np.mean(entropies))
  valid = np.asarray(mask, dtype=bool)
  if valid.shape != entropies.shape:
    msg = "Entropy mask must have shape (N,)."
    raise ValueError(msg)
  if not np.any(valid):
    return 0.0
  return float(np.mean(entropies[valid]))


def amino_acid_distribution(
  sampled_tokens: np.ndarray,
  *,
  num_amino_acids: int = 21,
  mask: np.ndarray | None = None,
) -> np.ndarray:
  """Compute normalized amino-acid frequency distribution."""
  sampled = np.asarray(sampled_tokens, dtype=np.int64).ravel()
  if mask is not None:
    valid = np.asarray(mask, dtype=bool).ravel()
    if valid.shape != sampled.shape:
      msg = "Distribution mask shape must match flattened token shape."
      raise ValueError(msg)
    sampled = sampled[valid]
  sampled = sampled[(sampled >= 0) & (sampled < num_amino_acids)]
  counts = np.bincount(sampled, minlength=num_amino_acids).astype(np.float64)
  total = float(np.sum(counts))
  if total <= 0.0:
    return np.full((num_amino_acids,), 1.0 / num_amino_acids, dtype=np.float64)
  return counts / total


def downsample_pair_points(
  reference: np.ndarray,
  observed: np.ndarray,
  *,
  max_points: int,
  seed: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Downsample flattened pair points deterministically."""
  lhs = np.asarray(reference, dtype=np.float64).ravel()
  rhs = np.asarray(observed, dtype=np.float64).ravel()
  if lhs.shape != rhs.shape:
    msg = "Point sampling inputs must have matching shapes."
    raise ValueError(msg)
  if lhs.size <= max_points:
    return lhs, rhs

  rng = np.random.default_rng(seed)
  indices = rng.choice(lhs.size, size=max_points, replace=False)
  return lhs[indices], rhs[indices]


def write_metric_records_csv(records: list[EvidenceMetricRecord], output_path: Path) -> None:
  """Write metric records to CSV."""
  field_names = (
    list(asdict(records[0]).keys()) if records else list(asdict(_empty_metric_record()).keys())
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=field_names)
    writer.writeheader()
    for record in records:
      writer.writerow(asdict(record))


def write_point_records_csv(records: list[EvidencePointRecord], output_path: Path) -> None:
  """Write pointwise records to CSV."""
  field_names = (
    list(asdict(records[0]).keys()) if records else list(asdict(_empty_point_record()).keys())
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=field_names)
    writer.writeheader()
    for record in records:
      writer.writerow(asdict(record))


def write_metric_records_json(records: list[EvidenceMetricRecord], output_path: Path) -> None:
  """Write metric records to JSON."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  payload = [asdict(record) for record in records]
  output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empty_metric_record() -> EvidenceMetricRecord:
  return EvidenceMetricRecord(
    path_id="",
    tier="",
    case_id="",
    case_kind="",
    backbone_id="",
    seed=0,
    sequence_length=0,
    checkpoint_id="",
    metric_name="",
    metric_value=0.0,
  )


def _empty_point_record() -> EvidencePointRecord:
  return EvidencePointRecord(
    path_id="",
    tier="",
    case_id="",
    case_kind="",
    backbone_id="",
    seed=0,
    sequence_length=0,
    reference_value=0.0,
    observed_value=0.0,
  )
