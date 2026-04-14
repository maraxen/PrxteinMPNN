"""Tests for weighted multi-state logit combination."""

import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.model.multi_state_sampling import (
  arithmetic_mean_logits,
  geometric_mean_logits,
  product_of_probabilities_logits,
)

def test_weighted_arithmetic_mean():
  logits = jnp.array([
    [10.0, 0.0],  # State 0: strong preference for AA0
    [0.0, 10.0],  # State 1: strong preference for AA1
  ])
  group_mask = jnp.array([True, True])
  state_mapping = jnp.array([0, 1])
  
  # Equal weights: should be intermediate
  weights_equal = jnp.array([0.5, 0.5])
  avg_equal = arithmetic_mean_logits(logits, group_mask, weights_equal, state_mapping)
  
  # State 0 dominance: should prefer AA0
  weights_0 = jnp.array([0.99, 0.01])
  avg_0 = arithmetic_mean_logits(logits, group_mask, weights_0, state_mapping)
  assert avg_0[0, 0] > avg_0[0, 1]
  
  # State 1 dominance: should prefer AA1
  weights_1 = jnp.array([0.01, 0.99])
  avg_1 = arithmetic_mean_logits(logits, group_mask, weights_1, state_mapping)
  assert avg_1[0, 1] > avg_1[0, 0]

def test_weighted_geometric_mean():
  logits = jnp.array([
    [10.0, 0.0],
    [0.0, 10.0],
  ])
  group_mask = jnp.array([True, True])
  state_mapping = jnp.array([0, 1])
  temp = 1.0
  
  weights_0 = jnp.array([0.8, 0.2])
  avg_0 = geometric_mean_logits(logits, group_mask, temp, weights_0, state_mapping)
  # (0.8*10 + 0.2*0) / 1.0 = 8.0
  # (0.8*0 + 0.2*10) / 1.0 = 2.0
  assert jnp.allclose(avg_0, jnp.array([[8.0, 2.0]]))

def test_weighted_product():
  logits = jnp.array([
    [10.0, 0.0],
    [0.0, 10.0],
  ])
  group_mask = jnp.array([True, True])
  state_mapping = jnp.array([0, 1])
  
  weights = jnp.array([2.0, 0.5])
  res = product_of_probabilities_logits(logits, group_mask, weights, state_mapping)
  # 2.0*10 + 0.5*0 = 20
  # 2.0*0 + 0.5*10 = 5
  assert jnp.allclose(res, jnp.array([[20.0, 5.0]]))

def test_numerical_stability():
  # Large logits that might overflow exp()
  logits = jnp.array([
    [1000.0, 0.0],
    [0.0, 1000.0],
  ])
  group_mask = jnp.array([True, True])
  state_mapping = jnp.array([0, 1])
  weights = jnp.array([0.5, 0.5])
  
  # Stable implementation should handle this
  res = arithmetic_mean_logits(logits, group_mask, weights, state_mapping)
  assert not jnp.any(jnp.isnan(res))
  assert not jnp.any(jnp.isinf(res))
