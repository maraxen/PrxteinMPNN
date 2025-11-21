"""Integration tests for multi-state sampling through the full pipeline."""

from __future__ import annotations

import pytest

from prxteinmpnn.run.specs import SamplingSpecification


def test_sampling_spec_has_multi_state_parameters() -> None:
  """Test that SamplingSpecification accepts multi-state parameters.
  
  Args:
    None
  
  Returns:
    None
  
  Raises:
    AssertionError: If parameters are not properly set.
  
  Example:
    >>> test_sampling_spec_has_multi_state_parameters()
  
  """
  spec = SamplingSpecification(
    inputs=["test.pdb"],
    tied_positions="direct",
    pass_mode="inter",
    multi_state_strategy="min",
    multi_state_alpha=0.7,
  )

  assert spec.multi_state_strategy == "min"
  assert spec.multi_state_alpha == 0.7
  assert spec.tied_positions == "direct"
  assert spec.pass_mode == "inter"


def test_sampling_spec_default_multi_state_parameters() -> None:
  """Test default values for multi-state parameters.
  
  Args:
    None
  
  Returns:
    None
  
  Raises:
    AssertionError: If default values are incorrect.
  
  Example:
    >>> test_sampling_spec_default_multi_state_parameters()
  
  """
  spec = SamplingSpecification(inputs=["test.pdb"])

  assert spec.multi_state_strategy == "mean"
  assert spec.multi_state_alpha == 0.5


def test_all_multi_state_strategies_accepted() -> None:
  """Test that all multi-state strategies are accepted.
  
  Args:
    None
  
  Returns:
    None
  
  Raises:
    AssertionError: If any strategy is not accepted.
  
  Example:
    >>> test_all_multi_state_strategies_accepted()
  
  """
  strategies = ["mean", "min", "product", "max_min"]

  for strategy in strategies:
    spec = SamplingSpecification(
      inputs=["test.pdb"],
      multi_state_strategy=strategy,
    )
    assert spec.multi_state_strategy == strategy


def test_multi_state_alpha_range() -> None:
  """Test that multi_state_alpha accepts valid range.
  
  Args:
    None
  
  Returns:
    None
  
  Raises:
    AssertionError: If alpha values are not properly set.
  
  Example:
    >>> test_multi_state_alpha_range()
  
  """
  alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

  for alpha in alphas:
    spec = SamplingSpecification(
      inputs=["test.pdb"],
      multi_state_alpha=alpha,
    )
    assert spec.multi_state_alpha == alpha


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
