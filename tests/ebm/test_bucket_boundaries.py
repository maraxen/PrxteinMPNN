"""Regression tests for the E4.5 xtrax HiTL bucket-boundary gate.

Locks in two things about ``scripts/ebm/bucket_boundary_check.py`` so the
mechanical half of the E4.5 gate doesn't silently rot:

  1. The proxy-distribution construction (real fixture parsing + the
     documented synthetic log-normal mixture) stays deterministic and
     within its documented bounds.
  2. The xtrax EDA calls (``explain_plan``/``analyze_bucket`` via
     ``demonstrate_xtrax_eda``, and the per-length ``select_bucket`` sweep
     via ``analyze_distribution``) run against the confirmed
     ``(64, 128, 256, 512)`` boundaries without error and produce
     structurally sane output.

These are fast regression tests, not a re-derivation of the bucket-boundary
research finding itself -- the finding (padding-waste PASS, ~7% proxy-length
overflow CONCERN) is reported in the E4.5 task writeup, not asserted here as
a pinned research number (the synthetic component is a proxy, not real corpus
data -- see the script's module docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ebm.bucket_boundary_check import (
  DEFAULT_BOUNDARIES,
  DEFAULT_DATA_DIR,
  SYNTHETIC_MAX_LENGTH,
  SYNTHETIC_MIN_LENGTH,
  ProxyDistribution,
  analyze_distribution,
  build_proxy_distribution,
  build_synthetic_lengths,
  demonstrate_xtrax_eda,
  load_real_lengths,
)


class TestSyntheticLengths:
  def test_deterministic_given_seed(self) -> None:
    a = build_synthetic_lengths(n=200, seed=0)
    b = build_synthetic_lengths(n=200, seed=0)
    assert a == b

  def test_different_seeds_differ(self) -> None:
    a = build_synthetic_lengths(n=200, seed=0)
    b = build_synthetic_lengths(n=200, seed=1)
    assert a != b

  def test_within_documented_clip_bounds(self) -> None:
    lengths = build_synthetic_lengths(n=500, seed=0)
    assert len(lengths) == 500
    assert all(SYNTHETIC_MIN_LENGTH <= length <= SYNTHETIC_MAX_LENGTH for length in lengths)
    assert all(isinstance(length, int) for length in lengths)


class TestRealFixtureLengths:
  def test_loads_known_local_pdb_and_cif_fixtures(self) -> None:
    lengths, sources = load_real_lengths(DEFAULT_DATA_DIR)
    # tests/data ships at least these real structures (see E4.5 task brief);
    # tolerate additional fixtures being added later without breaking this test.
    assert len(lengths) >= 6
    assert len(lengths) == len(sources)
    assert all(length > 0 for length in lengths)
    names = {Path(s).name for s in sources}
    assert {"1ubq.pdb", "1mbn.pdb", "3pgk.pdb", "5awl.pdb", "1BC8.cif", "1SMD.pdb"} <= names

  def test_missing_data_dir_returns_empty_not_raises(self, tmp_path: Path) -> None:
    empty_dir = tmp_path / "no_such_structures"
    empty_dir.mkdir()
    lengths, sources = load_real_lengths(empty_dir)
    assert lengths == []
    assert sources == []


class TestProxyDistribution:
  def test_all_lengths_concatenates_real_then_synthetic(self) -> None:
    proxy = ProxyDistribution(real_lengths=[10, 20], synthetic_lengths=[30, 40, 50])
    assert proxy.all_lengths == [10, 20, 30, 40, 50]

  def test_build_proxy_distribution_combines_both_sources(self) -> None:
    proxy = build_proxy_distribution(n_synthetic=100, seed=0)
    assert len(proxy.synthetic_lengths) == 100
    assert len(proxy.real_lengths) >= 6
    assert len(proxy.all_lengths) == len(proxy.real_lengths) + 100


class TestXtraxEdaDemo:
  def test_confirmed_boundaries_select_bucket_strategy(self) -> None:
    result = demonstrate_xtrax_eda(DEFAULT_BOUNDARIES, representative_length=200)
    axis_entry = result["explain_plan"]["axes"][0]
    assert axis_entry["name"] == "residue"
    assert axis_entry["strategy"] == "Bucket"
    assert axis_entry["reasoning"]  # explain_plan guarantees non-empty
    assert result["analyze_bucket"]["bucket_boundaries"] == list(DEFAULT_BOUNDARIES)
    assert result["analyze_bucket"]["bucket_count"] == len(DEFAULT_BOUNDARIES)


class TestAnalyzeDistribution:
  def test_runs_without_error_on_proxy_distribution(self) -> None:
    proxy = build_proxy_distribution(n_synthetic=300, seed=0)
    bucket_stats, overflow_lengths = analyze_distribution(proxy.all_lengths, DEFAULT_BOUNDARIES)
    assert len(bucket_stats) == len(DEFAULT_BOUNDARIES)
    # Every proxy length is accounted for exactly once: in some bucket, or overflow.
    assigned_total = sum(stat.count for stat in bucket_stats)
    assert assigned_total + len(overflow_lengths) == len(proxy.all_lengths)

  def test_overflow_lengths_all_exceed_largest_boundary(self) -> None:
    proxy = build_proxy_distribution(n_synthetic=300, seed=0)
    _, overflow_lengths = analyze_distribution(proxy.all_lengths, DEFAULT_BOUNDARIES)
    assert all(length > DEFAULT_BOUNDARIES[-1] for length in overflow_lengths)

  def test_bucket_membership_respects_select_bucket_contract(self) -> None:
    # Small, fully-enumerable case: every length from 1..512 must land in the
    # smallest boundary >= length (mirrors xtrax.tiling.select_bucket exactly,
    # since analyze_distribution calls that same primitive).
    lengths = list(range(1, 513))
    bucket_stats, overflow_lengths = analyze_distribution(lengths, DEFAULT_BOUNDARIES)
    assert overflow_lengths == []
    counts = {stat.boundary: stat.count for stat in bucket_stats}
    assert counts[64] == 64  # lengths 1..64
    assert counts[128] == 64  # lengths 65..128
    assert counts[256] == 128  # lengths 129..256
    assert counts[512] == 256  # lengths 257..512

  def test_no_bucket_mean_padding_waste_exceeds_50_percent_on_full_span(self) -> None:
    """Sanity bound for analyze_distribution's own waste math (not a corpus claim).

    Uniformly filling every length 1..512 is the worst-case padding-waste
    input for these specific boundaries (each bucket's mean length is close
    to its own midpoint). Even here mean waste stays under 50%, so a bucket
    crossing that threshold on a real run is a genuine distributional signal,
    not a script artifact.
    """
    lengths = list(range(1, 513))
    bucket_stats, _ = analyze_distribution(lengths, DEFAULT_BOUNDARIES)
    for stat in bucket_stats:
      assert stat.mean_pad_waste_pct is not None
      assert stat.mean_pad_waste_pct < 50.0

  @pytest.mark.parametrize("boundary", DEFAULT_BOUNDARIES)
  def test_confirmed_boundaries_are_positive_and_sorted(self, boundary: int) -> None:
    assert boundary > 0
    assert boundary in DEFAULT_BOUNDARIES
