"""Sampling and scoring hyperparameter configuration.

These frozen dataclasses define algorithm-level configuration separate from
data bundles, allowing hyperparameter sweeps over fixed inference bundles.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class SamplingHyperparameters:
	"""Hyperparameters for sampling and scoring operations.

	Defines algorithm configuration (how to sample/score) separately from
	data bundles (what structure/sequence data to use).
	"""

	# Temperature sampling (autoregressive) hyperparameters
	temperature: float = 1.0
	"""Temperature for softmax sampling (1.0 = no temperature scaling)."""

	# Straight-through estimator (STE) hyperparameters
	iterations: int = 100
	"""Number of optimization iterations for STE-based sampling."""

	learning_rate: float = 0.01
	"""Learning rate for STE-based optimization."""

	use_concrete: bool = False
	"""Use Gumbel-Softmax (Concrete) distribution for straight_through mode."""

	tau_start: float = 1.0
	"""Starting temperature for Gumbel-Softmax annealing."""

	tau_end: float = 0.1
	"""Ending temperature for Gumbel-Softmax annealing."""

	# Multi-state / ensemble hyperparameters
	multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean"
	"""How to combine logits across multiple states: arithmetic_mean, geometric_mean, or product."""

	multi_state_temperature: float = 1.0
	"""Temperature scaling applied to multi-state logit combination."""

	# Batch/execution hyperparameters
	batch_size: int = 1
	"""Batch size for processing (future: support batched sampling)."""

	# State handling
	use_rolling_state: bool = False
	"""Use rolling state (scan) vs vmap across iteration steps."""
