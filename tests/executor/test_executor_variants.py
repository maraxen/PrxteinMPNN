"""Basic smoke tests for executor variants."""

from __future__ import annotations

import pytest

from prxteinmpnn.executor import (
  AutoregressiveExecutor,
  ConditionalExecutor,
  Executor,
  UnconditionalExecutor,
)
from prxteinmpnn.pipeline_registry import StageSet


def test_executor_import():
  """Test that all executor variants can be imported."""
  assert Executor is not None
  assert UnconditionalExecutor is not None
  assert ConditionalExecutor is not None
  assert AutoregressiveExecutor is not None


def test_executor_base_initialization():
  """Test Executor base class initialization with default StageSet."""
  stage_set = StageSet.default()
  executor = Executor(stage_set)
  assert executor._stage_set is stage_set


def test_unconditional_executor_initialization():
  """Test UnconditionalExecutor initialization."""
  stage_set = StageSet.default()
  executor = UnconditionalExecutor(stage_set=stage_set)
  assert executor._stage_set is stage_set
  assert executor.multi_state_strategy_idx == 0
  assert executor.inference is True


def test_unconditional_executor_default_stage_set():
  """Test UnconditionalExecutor with default StageSet."""
  executor = UnconditionalExecutor()
  assert executor._stage_set is not None
  assert isinstance(executor._stage_set, StageSet)


def test_conditional_executor_initialization():
  """Test ConditionalExecutor initialization."""
  stage_set = StageSet.default()
  executor = ConditionalExecutor(stage_set=stage_set)
  assert executor._stage_set is stage_set
  assert executor.multi_state_strategy_idx == 0
  assert executor.inference is True


def test_conditional_executor_default_stage_set():
  """Test ConditionalExecutor with default StageSet."""
  executor = ConditionalExecutor()
  assert executor._stage_set is not None
  assert isinstance(executor._stage_set, StageSet)


def test_autoregressive_executor_initialization():
  """Test AutoregressiveExecutor initialization."""
  stage_set = StageSet.default()
  executor = AutoregressiveExecutor(stage_set=stage_set)
  assert executor._stage_set is stage_set
  assert executor.temperature == 1.0
  assert executor.multi_state_strategy_idx == 0


def test_autoregressive_executor_default_stage_set():
  """Test AutoregressiveExecutor with default StageSet."""
  executor = AutoregressiveExecutor()
  assert executor._stage_set is not None
  assert isinstance(executor._stage_set, StageSet)


def test_autoregressive_executor_custom_temperature():
  """Test AutoregressiveExecutor with custom temperature."""
  executor = AutoregressiveExecutor(temperature=0.5)
  assert executor.temperature == 0.5


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
