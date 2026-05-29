"""Synthetic-invariant tests for new parity metrics and mask support."""

from __future__ import annotations

import pytest
import numpy as np

from prxteinmpnn.parity.evidence import (
  safe_cosine_similarity,
  safe_spearman,
  safe_pearson,
  mean_abs_error,
  root_mean_square_error,
  max_abs_error,
  mean_kl_divergence,
)


class TestSafeCosineSimilarity:
  """safe_cosine_similarity invariant tests."""

  def test_cosine_self_similarity_is_one(self) -> None:
    """cos(x, x) == 1.0 for non-constant x."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = safe_cosine_similarity(x, x)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

  def test_cosine_opposite_vectors_is_negative_one(self) -> None:
    """cos(x, -x) == -1.0."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    neg_x = -x
    result = safe_cosine_similarity(x, neg_x)
    assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

  def test_cosine_orthogonal_vectors_is_zero(self) -> None:
    """cos(orthogonal vectors) ~ 0.0."""
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    result = safe_cosine_similarity(x, y)
    assert np.isclose(result, 0.0, atol=1e-12), f"Expected ~0.0, got {result}"

  def test_cosine_zero_vector_raises_error(self) -> None:
    """Zero-vector input raises ValueError."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="zero norm"):
      safe_cosine_similarity(x, zero)

  def test_cosine_shape_mismatch_raises_error(self) -> None:
    """Shape mismatch raises ValueError."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
      safe_cosine_similarity(x, y)

  def test_cosine_returns_float(self) -> None:
    """Return type is float (not np.ndarray)."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = safe_cosine_similarity(x, x)
    assert isinstance(result, float)


class TestSafeSpearman:
  """safe_spearman invariant tests."""

  def test_spearman_self_similarity_is_one(self) -> None:
    """spearman(x, x) == 1.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = safe_spearman(x, x)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

  def test_spearman_monotonic_nonlinear_is_one(self) -> None:
    """Monotonic nonlinear y=x**3 (x>0) -> spearman=1.0 (rank-correlation)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = x**3
    result = safe_spearman(x, y)
    assert np.isclose(result, 1.0), f"Expected 1.0 for monotonic transform, got {result}"

  def test_spearman_opposite_vectors_is_negative_one(self) -> None:
    """spearman(x, -x) == -1.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    neg_x = -x
    result = safe_spearman(x, neg_x)
    assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

  def test_spearman_constant_input_returns_fallback(self) -> None:
    """Constant input returns 1.0 if both constant and equal, else 0.0."""
    x_const = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    y_const = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    result = safe_spearman(x_const, y_const)
    assert np.isclose(result, 1.0), f"Expected 1.0 for equal constants, got {result}"

    y_const_diff = np.array([6.0, 6.0, 6.0, 6.0], dtype=np.float64)
    result = safe_spearman(x_const, y_const_diff)
    assert np.isclose(result, 0.0), f"Expected 0.0 for unequal constants, got {result}"

  def test_spearman_shape_mismatch_raises_error(self) -> None:
    """Shape mismatch raises ValueError."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
      safe_spearman(x, y)

  def test_spearman_returns_float(self) -> None:
    """Return type is float (not np.ndarray)."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = safe_spearman(x, x)
    assert isinstance(result, float)


class TestContinuousMetricsMaskPlumbing:
  """Mask parameter support for continuous metrics."""

  def test_safe_pearson_mask_filtering(self) -> None:
    """masked safe_pearson differs from unmasked when masked rows are different."""
    # Valid rows: [1.0, 1.0, 1.0] identical
    # Masked rows: [0.0, 100.0, 0.0] wildly different
    ref = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    obs = np.array([1.0, 100.0, 1.0, 1.0], dtype=np.float64)
    mask = np.array([True, False, True, True], dtype=bool)

    unmasked = safe_pearson(ref, obs)
    masked = safe_pearson(ref, obs, mask=mask)

    # Masked should ignore the outlier (100.0 vs 0.0 at index 1)
    assert not np.isclose(masked, unmasked), "Masked should differ from unmasked with outliers"
    # Masked on valid rows [indices 0,2,3] should be nearly perfect (all 1.0)
    assert np.isclose(masked, 1.0, atol=1e-6), f"Expected ~1.0 on valid rows, got {masked}"

  def test_safe_pearson_mask_on_valid_rows_matches_subset(self) -> None:
    """masked safe_pearson(valid subset) == safe_pearson(valid rows only)."""
    ref = np.array([1.0, 99.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.1, 50.0, 2.1, 3.1], dtype=np.float64)
    mask = np.array([True, False, True, True], dtype=bool)

    masked_full = safe_pearson(ref, obs, mask=mask)
    subset_only = safe_pearson(ref[mask], obs[mask])

    assert np.isclose(masked_full, subset_only, atol=1e-12), \
      f"masked({masked_full}) != subset_only({subset_only})"

  def test_mean_abs_error_mask_filtering(self) -> None:
    """masked mean_abs_error differs when masked rows are different."""
    ref = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.0, 100.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, False, True, True], dtype=bool)

    unmasked = mean_abs_error(ref, obs)
    masked = mean_abs_error(ref, obs, mask=mask)

    assert not np.isclose(masked, unmasked), "Masked should differ from unmasked"
    # Masked: only diff at indices 0,2,3 which are all 0.0
    assert np.isclose(masked, 0.0, atol=1e-12), f"Expected 0.0 on valid rows, got {masked}"

  def test_root_mean_square_error_mask_filtering(self) -> None:
    """masked root_mean_square_error differs when masked rows are different."""
    ref = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.0, 100.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, False, True, True], dtype=bool)

    unmasked = root_mean_square_error(ref, obs)
    masked = root_mean_square_error(ref, obs, mask=mask)

    assert not np.isclose(masked, unmasked), "Masked should differ from unmasked"
    # Masked: only valid rows, all 0.0 difference
    assert np.isclose(masked, 0.0, atol=1e-12), f"Expected 0.0 on valid rows, got {masked}"

  def test_max_abs_error_mask_filtering(self) -> None:
    """masked max_abs_error differs when masked rows are different."""
    ref = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.0, 100.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, False, True, True], dtype=bool)

    unmasked = max_abs_error(ref, obs)
    masked = max_abs_error(ref, obs, mask=mask)

    assert not np.isclose(masked, unmasked), "Masked should differ from unmasked"
    # Masked: max diff on valid rows is 0.0
    assert np.isclose(masked, 0.0, atol=1e-12), f"Expected 0.0 on valid rows, got {masked}"

  def test_mean_kl_divergence_mask_filtering_2d(self) -> None:
    """masked mean_kl_divergence for rank-2 log-probs (N, C)."""
    ref_log_probs = np.log(
      np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5]], dtype=np.float64)
    )
    obs_log_probs = np.log(
      np.array([[0.6, 0.4], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    )
    mask = np.array([True, False, True], dtype=bool)

    unmasked = mean_kl_divergence(ref_log_probs, obs_log_probs)
    masked = mean_kl_divergence(ref_log_probs, obs_log_probs, mask=mask)

    # Row 1 is masked out (has different KL)
    assert not np.isclose(masked, unmasked), "Masked should differ from unmasked"
    # Rows 0,2: row 0 has small KL, row 2 has KL=0 (both identical [0.5, 0.5])
    assert masked <= unmasked, "Masked mean should be <= unmasked when outlier masked out"

  def test_metric_with_no_valid_rows_returns_zero(self) -> None:
    """When mask has no True values, return 0.0."""
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    all_false_mask = np.array([False, False, False], dtype=bool)

    assert mean_abs_error(ref, obs, mask=all_false_mask) == 0.0
    assert root_mean_square_error(ref, obs, mask=all_false_mask) == 0.0
    assert max_abs_error(ref, obs, mask=all_false_mask) == 0.0

  def test_mask_shape_mismatch_raises_error(self) -> None:
    """Mask shape mismatch raises ValueError."""
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bad_mask = np.array([True, False], dtype=bool)  # Wrong size

    with pytest.raises(ValueError, match="mask"):
      safe_pearson(ref, obs, mask=bad_mask)

    with pytest.raises(ValueError, match="mask"):
      mean_abs_error(ref, obs, mask=bad_mask)

    with pytest.raises(ValueError, match="mask"):
      root_mean_square_error(ref, obs, mask=bad_mask)

    with pytest.raises(ValueError, match="mask"):
      max_abs_error(ref, obs, mask=bad_mask)


class TestExistingInvariants:
  """Regression: existing behavior unchanged."""

  def test_safe_pearson_existing(self) -> None:
    """safe_pearson(x, x) == 1 and safe_pearson(x, -x) == -1."""
    x = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    assert np.isclose(safe_pearson(x, x), 1.0)
    assert np.isclose(safe_pearson(x, -x), -1.0)

  def test_mean_abs_error_existing(self) -> None:
    """mean_abs_error(x, x) == 0."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert np.isclose(mean_abs_error(x, x), 0.0)

  def test_root_mean_square_error_existing(self) -> None:
    """root_mean_square_error(x, x) == 0."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert np.isclose(root_mean_square_error(x, x), 0.0)

  def test_mean_kl_divergence_asymmetry(self) -> None:
    """KL(p || q) != KL(q || p) [asymmetric]."""
    p_log = np.log(np.array([[0.7, 0.3]], dtype=np.float64))
    q_log = np.log(np.array([[0.4, 0.6]], dtype=np.float64))
    kl_pq = mean_kl_divergence(p_log, q_log)
    kl_qp = mean_kl_divergence(q_log, p_log)
    assert not np.isclose(kl_pq, kl_qp), "KL divergence should be asymmetric"
