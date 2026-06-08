"""Shared helpers for comparing empirical AA histograms between samplers."""

from __future__ import annotations

import numpy as np


def kl_discrete(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
  p = np.clip(p.astype(np.float64), eps, 1.0)
  q = np.clip(q.astype(np.float64), eps, 1.0)
  p = p / p.sum()
  q = q / q.sum()
  return float(np.sum(p * (np.log(p) - np.log(q))))


def mean_positionwise_js(samples_a: np.ndarray, samples_b: np.ndarray, *, n_aa: int = 20) -> float:
  """Mean Jensen–Shannon divergence per residue over the first ``n_aa`` token classes."""
  ns, ell = samples_a.shape
  if samples_b.shape != (ns, ell):
    msg = "Sample batches must align in shape."
    raise ValueError(msg)
  divergences = []
  for pos in range(ell):
    ca = np.bincount(samples_a[:, pos], minlength=n_aa)[:n_aa].astype(np.float64)
    cb = np.bincount(samples_b[:, pos], minlength=n_aa)[:n_aa].astype(np.float64)
    ca /= max(float(ca.sum()), 1e-12)
    cb /= max(float(cb.sum()), 1e-12)
    m = 0.5 * (ca + cb)
    divergences.append(0.5 * (kl_discrete(ca, m) + kl_discrete(cb, m)))
  return float(np.mean(divergences))
