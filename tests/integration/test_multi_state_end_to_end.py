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
    multi_state_strategy="product",
  )

  assert spec.multi_state_strategy == "product"
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

  assert spec.multi_state_strategy == "arithmetic_mean"


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
  strategies = ["arithmetic_mean", "geometric_mean", "product"]

  for strategy in strategies:
    spec = SamplingSpecification(
      inputs=["test.pdb"],
      multi_state_strategy=strategy,
    )
    assert spec.multi_state_strategy == strategy


if __name__ == "__main__":
  pytest.main([__file__, "-v"])

