# Multi-State Sampling Implementation - Complete End-to-End

This document describes the complete implementation of multi-state sampling for protein design, where the model generates sequences that satisfy multiple conformational states simultaneously.

## Overview

Multi-state sampling allows ProteinMPNN to design protein sequences that work well across **multiple structural states** of the same protein (e.g., different binding states, functional conformations). Instead of simple averaging (which creates compromise predictions), we provide three strategies for combining logits across states.

## Important Implementation Detail: JAX Tracing

**Critical Fix**: The `multi_state_strategy` parameter is a Python string, but JAX's JIT compilation cannot trace string values through transformations. The solution is to convert the strategy string to an integer index at the model's `__call__` entry point, pass the integer through JAX transformations, then convert back to a string literal or use `jax.lax.switch` with the index.

```python
# In model.__call__()
strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
multi_state_strategy_idx = jnp.array(strategy_map[multi_state_strategy], dtype=jnp.int32)

# In _run_autoregressive_scan() and other methods
# The strategy_idx is passed through and used with jax.lax.switch
```

This ensures the strategy can be traced through JAX's computational graph.

## Implementation Flow

### 1. User Specification (`specs.py`)

**File**: `src/prxteinmpnn/run/specs.py`

The `SamplingSpecification` and `ScoringSpecification` dataclasses include new parameters:

```python
@dataclass
class SamplingSpecification(RunSpecification):
  # ...
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean"

@dataclass
class RunSpecification:
  # ...
  multi_state_temperature: float = 1.0
```

**Parameters**:

- `multi_state_strategy`: Strategy for combining logits across tied positions
  - `"arithmetic_mean"`: Average logits using log-sum-exp (standard consensus)
  - `"geometric_mean"`: Geometric mean of probabilities with temperature scaling
  - `"product"`: Sum of logits (multiplies probabilities, favors consistency)
- `multi_state_temperature`: Temperature scaling specifically for the `geometric_mean` strategy.

### 2. Sampling Entry Point (`sampling.py`)

**File**: `src/prxteinmpnn/run/sampling.py`

The `sample()` function passes parameters through to the model via the sampler function:

```python
# In _sample_batch
sample_fn_with_params = partial(
  sampler_fn,
  # ...
  multi_state_strategy=spec.multi_state_strategy,
  multi_state_temperature=spec.multi_state_temperature,
  num_groups=num_groups,
)
```

### 3. Sample Function Factory (`sample.py`)

**File**: `src/prxteinmpnn/sampling/sample.py`

The `make_sample_sequences()` factory creates sampling functions that accept multi-state parameters:

```python
def sample_sequences(
  # ...
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: Float = 1.0,
) -> tuple[ProteinSequence, Logits, DecodingOrder]:
  # ... 
  sampled_sequence, logits = model(
    # ...
    tie_group_map=tie_group_map,
    multi_state_strategy=multi_state_strategy,
    multi_state_temperature=multi_state_temperature,
  )
```

### 4. Model Call Chain (`mpnn.py`)

**File**: `src/prxteinmpnn/model/mpnn.py`

The model's `__call__` method receives the parameters, converts the strategy to an index, and passes it through the autoregressive or conditional call chain:

```python
def __call__(
  # ...
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
) -> tuple[OneHotProteinSequence, Logits]:
  # ...
  strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
  multi_state_strategy_idx = jnp.array(strategy_map[multi_state_strategy], dtype=jnp.int32)
  # ...
  return jax.lax.switch(branch_index, branches, *operands)
```

### 5. Multi-State Logit Combination (`multi_state_sampling.py`)

**File**: `src/prxteinmpnn/model/multi_state_sampling.py`

The implementation provides three mathematical functions for combining logits:

1. **Arithmetic Mean** (`arithmetic_mean_logits`):
   Uses log-sum-exp for stable averaging of logits.
   
2. **Geometric Mean** (`geometric_mean_logits`):
   Computes `sum(logits) / (temperature * num_in_group)`.

3. **Product** (`product_of_probabilities_logits`):
   Simply sums the logits across states (equivalent to multiplying probabilities).

## Usage Examples

### Basic Usage

```python
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

# Create specification with multi-state sampling
spec = SamplingSpecification(
  inputs=["state1.pdb", "state2.pdb"],
  tied_positions="direct",             # Tie corresponding positions
  pass_mode="inter",                   # Required for direct tying
  multi_state_strategy="product",      # Favor consistency across states
  num_samples=100,
  temperature=0.1,
)

# Sample sequences
results = sample(spec)
sequences = results["sequences"]
```

### Comparing Strategies

```python
strategies = ["arithmetic_mean", "geometric_mean", "product"]

for strategy in strategies:
  spec = SamplingSpecification(
    inputs=["state1.pdb", "state2.pdb"],
    tied_positions="direct",
    pass_mode="inter",
    multi_state_strategy=strategy,
    num_samples=100,
  )
  
  results = sample(spec)
```

## Expected Behavior

### Strategy Comparison

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **Arithmetic Mean** | log-sum-exp average | Consensus prediction, balanced compromise |
| **Geometric Mean** | scaled average | Balanced probabilities in log-space |
| **Product** | Logit summation | Multiplies probabilities; strongly favors residues that are high-probability in ALL states |

## Testing

### Unit Tests

**File**: `tests/model/test_multi_state_sampling.py`

Tests the mathematical correctness of each strategy:
- `test_arithmetic_mean_logits()`
- `test_geometric_mean_logits()`
- `test_product_of_probabilities_logits()`
- `test_jit_compatibility()`

**Run**: `uv run pytest tests/model/test_multi_state_sampling.py -v`

### Integration Tests

**File**: `tests/integration/test_multi_state_end_to_end.py`

Tests the end-to-end parameter flow through the pipeline.

**Run**: `uv run pytest tests/integration/test_multi_state_end_to_end.py -v`

## Implementation Status

✅ **Completed**:
- Multi-state sampling strategies implemented in `multi_state_sampling.py`
- Parameters added to `SamplingSpecification` and `ScoringSpecification`
- Parameters wired through `sampling.py` → `sample.py` → `mpnn.py`
- Unified implementation for both `PrxteinMPNN` and `PrxteinLigandMPNN`
- Numerical parity validated against LigandMPNN reference for `arithmetic_mean` and `product` strategies
- Unit tests and Integration tests passing

✅ **Ready for Use**:
The implementation is complete and verified. Users can specify `multi_state_strategy` in their sampling or scoring specs.
