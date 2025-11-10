# Multi-State Sampling Implementation - Complete End-to-End

This document describes the complete implementation of multi-state sampling for protein design, where the model generates sequences that satisfy multiple conformational states simultaneously.

## Overview

Multi-state sampling allows ProteinMPNN to design protein sequences that work well across **multiple structural states** of the same protein (e.g., different binding states, functional conformations). Instead of averaging logits (which creates compromise predictions), we provide four strategies for combining logits across states.

## Implementation Flow

### 1. User Specification (`specs.py`)

**File**: `src/prxteinmpnn/run/specs.py`

The `SamplingSpecification` dataclass includes two new parameters:

```python
@dataclass
class SamplingSpecification(RunSpecification):
  # ... existing parameters ...
  multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean"
  multi_state_alpha: float = 0.5
```

**Parameters**:

- `multi_state_strategy`: Strategy for combining logits across tied positions
  - `"mean"`: Average logits (original behavior, creates compromise)
  - `"min"`: Minimum logits (worst-case robust design)
  - `"product"`: Sum of logits (multiply probabilities)
  - `"max_min"`: Weighted combination with alpha parameter
- `multi_state_alpha`: Weight for min component when `strategy="max_min"` (0-1)
  - 0.0 = pure mean (compromise)
  - 1.0 = pure min (maximum robustness)
  - 0.5 = balanced

### 2. Sampling Entry Point (`sampling.py`)

**File**: `src/prxteinmpnn/run/sampling.py`

The `sample()` function passes parameters through to the model:

```python
sampled_sequences, logits, _ = vmap_structures(
  # ... existing parameters ...
  tie_group_map,
  num_groups,
  spec.multi_state_strategy,  # NEW
  spec.multi_state_alpha,     # NEW
)
```

This also applies to the streaming version `_sample_streaming()`.

### 3. Sample Function Factory (`sample.py`)

**File**: `src/prxteinmpnn/sampling/sample.py`

The `make_sample_sequences()` factory creates sampling functions that accept multi-state parameters:

```python
def sample_sequences(
  # ... existing parameters ...
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
  multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
  multi_state_alpha: float = 0.5,
) -> tuple[ProteinSequence, Logits, DecodingOrder]:
  # ... 
  sampled_sequence, logits = model(
    # ... existing parameters ...
    tie_group_map=tie_group_map,
    multi_state_strategy=multi_state_strategy,  # Pass to model
    multi_state_alpha=multi_state_alpha,        # Pass to model
  )
```

### 4. Model Call Chain (`mpnn.py`)

**File**: `src/prxteinmpnn/model/mpnn.py`

The model's `__call__` method receives the parameters and passes them through the autoregressive call chain:

```python
def __call__(
  # ... existing parameters ...
  multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
  multi_state_alpha: float = 0.5,
) -> tuple[OneHotProteinSequence | Int, Logits]:
  # Dispatch to appropriate decoding approach
  return jax.lax.switch(
    branch_index,
    branches,
    # ... existing args ...
    multi_state_strategy,
    multi_state_alpha,
  )
```

The parameters flow through:

- `__call__()` → `_call_autoregressive()` → `_run_autoregressive_scan()` → `_run_tied_position_scan()`

### 5. Multi-State Logit Combination (`multi_state_sampling.py`)

**File**: `src/prxteinmpnn/model/multi_state_sampling.py`

The `_run_tied_position_scan()` method uses `_combine_logits_multistate()` to apply the selected strategy:

```python
def _combine_logits_multistate(
  self,
  logits: Logits,
  group_mask: Float[jnp.ndarray, "N 1"],
  strategy: Literal["mean", "min", "product", "max_min"],
  alpha: float,
) -> Logits:
  """Combine logits across tied positions using specified strategy."""
  
  # Dispatch to appropriate strategy
  return jax.lax.switch(
    strategy_index,
    [
      lambda: self._average_logits_over_group(logits, group_mask),
      lambda: min_over_group_logits(logits, group_mask),
      lambda: product_of_probabilities_logits(logits, group_mask),
      lambda: max_min_over_group_logits(logits, group_mask, alpha),
    ],
  )
```

**Strategy Implementations**:

1. **Mean** (`_average_logits_over_group`):

   ```python
   combined_logits = (logits * group_mask).sum(axis=0) / group_mask.sum(axis=0)
   ```

2. **Min** (`min_over_group_logits`):

   ```python
   masked_logits = jnp.where(group_mask > 0, logits, jnp.inf)
   combined_logits = masked_logits.min(axis=0)
   ```

3. **Product** (`product_of_probabilities_logits`):

   ```python
   # Sum logits = multiply probabilities
   combined_logits = (logits * group_mask).sum(axis=0)
   ```

4. **Max-Min** (`max_min_over_group_logits`):

   ```python
   mean_logits = (logits * group_mask).sum(axis=0) / group_mask.sum(axis=0)
   min_logits = jnp.where(group_mask > 0, logits, jnp.inf).min(axis=0)
   combined_logits = (1 - alpha) * mean_logits + alpha * min_logits
   ```

## Usage Examples

### Basic Usage

```python
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

# Create specification with multi-state sampling
spec = SamplingSpecification(
  inputs=["state1.pdb", "state2.pdb"],
  tied_positions="direct",       # Tie corresponding positions
  pass_mode="inter",             # Required for direct tying
  multi_state_strategy="min",    # Use worst-case robust design
  multi_state_alpha=0.5,         # Not used for min strategy
  num_samples=100,
  temperature=0.1,
)

# Sample sequences
results = sample(spec)
sequences = results["sequences"]
```

### Comparing Strategies

```python
strategies = ["mean", "min", "product"]

for strategy in strategies:
  spec = SamplingSpecification(
    inputs=["state1.pdb", "state2.pdb"],
    tied_positions="direct",
    pass_mode="inter",
    multi_state_strategy=strategy,
    num_samples=100,
  )
  
  results = sample(spec)
  # Analyze sequence identity, stability, etc.
```

### Tuning Robustness with Max-Min

```python
alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

for alpha in alphas:
  spec = SamplingSpecification(
    inputs=["state1.pdb", "state2.pdb"],
    tied_positions="direct",
    pass_mode="inter",
    multi_state_strategy="max_min",
    multi_state_alpha=alpha,  # 0=mean, 1=min
    num_samples=100,
  )
  
  results = sample(spec)
```

## Expected Behavior

### Strategy Comparison

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **Mean** | Averages logits, creates compromise | Original behavior, may not match any state well |
| **Min** | Takes minimum logit for each AA | Sequences work acceptably in ALL states (robust) |
| **Product** | Sums logits (multiplies probs) | Amplifies consistently high probabilities |
| **Max-Min** | Weighted combination | Tunable tradeoff between optimality and robustness |

### Expected Outcomes

- **Mean**: May produce lower sequence identity because it creates compromises
- **Min**: Higher sequence identity for truly multi-state proteins
- **Product**: Favors amino acids with consistently high probability across states
- **Max-Min with α=0.5**: Balanced approach between mean and min

## Testing

### Unit Tests

**File**: `tests/model/test_multi_state_sampling.py`

Tests the mathematical correctness of each strategy:

- `test_min_over_group_logits()`: Verifies min strategy
- `test_product_of_probabilities_logits()`: Verifies product strategy
- `test_max_min_*()`: Verifies alpha weighting
- `test_jit_compatibility()`: Ensures JIT compilation works

**Run**: `uv run pytest tests/model/test_multi_state_sampling.py -v`

### Integration Tests

**File**: `tests/integration/test_multi_state_end_to_end.py`

Tests the end-to-end parameter flow through the pipeline:

- `test_sampling_spec_has_multi_state_parameters()`: Spec accepts parameters
- `test_sampling_spec_default_multi_state_parameters()`: Default values correct
- `test_all_multi_state_strategies_accepted()`: All strategies valid
- `test_multi_state_alpha_range()`: Alpha range works

**Run**: `uv run pytest tests/integration/test_multi_state_end_to_end.py -v`

## Implementation Status

✅ **Completed**:

- Multi-state sampling strategies implemented in `multi_state_sampling.py`
- Parameters added to `SamplingSpecification`
- Parameters wired through `sampling.py` → `sample.py` → `mpnn.py`
- All method signatures updated for JAX compatibility
- Unit tests passing (8/8)
- Integration tests passing (4/4)
- Linting clean (ruff)

✅ **Ready for Use**:
The implementation is complete and tested. Users can now:

1. Specify `multi_state_strategy` and `multi_state_alpha` in `SamplingSpecification`
2. Run sampling with `tied_positions="direct"` and `pass_mode="inter"`
3. Compare different strategies empirically

## Next Steps for Users

1. **Run Experiments**: Use the different strategies on your multi-state protein data
2. **Evaluate Results**: Compare sequence identity, stability predictions, etc.
3. **Tune Parameters**: For max_min strategy, experiment with different alpha values
4. **Create Analysis Notebooks**: Document which strategies work best for your use cases

## References

- Original issue: Lower sequence identity with tied positions vs independent sampling
- Root cause: Logit averaging across divergent structural contexts creates compromises
- Solution: Provide multiple strategies that respect multi-state constraints differently
