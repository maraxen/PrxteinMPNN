# SamplingInputs Public API + Batch Dispatcher Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `MultistateStackPayload.slice()` and `SamplingInputs.slice_states()` for n_states axis slicing, a host-level `PayloadDispatcher` for multi-structure scoring, and wrap 3 deprecated test calls with `pytest.warns()`.

**Architecture:** Bottom-up slicing design — each payload defines its own `.slice(start, count)` method, composed in `SamplingInputs.slice_states()`. `PayloadDispatcher` (new module `run/_dispatcher.py`) iterates over `list[SamplingInputs]` at the host level, pre-splitting PRNG keys for determinism, and calls model methods per structure.

**Tech Stack:** JAX, Equinox, pytest

---

## Phase 1: Payload Slicing Core Implementation

### Task 1.1: Add `MultistateStackPayload.slice(start: int, count: int) -> MultistateStackPayload`

**Files:**
- Modify: `src/prxteinmpnn/payloads.py`

**Steps:**

- [ ] **1.1.1: Read current MultistateStackPayload definition** (understand all fields)
  
  Run: `PYTHONPATH=src uv run python -c "from prxteinmpnn.payloads import MultistateStackPayload; import inspect; print(inspect.getsource(MultistateStackPayload))"`

- [ ] **1.1.2: Add `slice()` method to `MultistateStackPayload`**

  After the `replace()` method in the class, add:
  
  ```python
  def slice(self, start: int, count: int) -> "MultistateStackPayload":
      """Return a new payload with states [start, start+count).
      
      Updates n_states = count.
      n_flat = count * n_canonical (assumes uniform state lengths — valid for tied multistate).
      All (S, ...) arrays are sliced on axis 0.
      flat_row_offsets is rebased: flat_row_offsets[start:start+count] - flat_row_offsets[start].
      
      Raises ValueError if start < 0 or start + count > n_states.
      """
      if start < 0 or count <= 0 or start + count > self.n_states:
          raise ValueError(
              f"slice out of range: start={start}, count={count}, n_states={self.n_states}"
          )
      
      # Slice arrays on axis 0 (the S dimension)
      sliced_coords = self.coords_stack[start:start+count]
      sliced_mask = self.mask_stack[start:start+count]
      sliced_ri = self.residue_index_stack[start:start+count]
      sliced_ci = self.chain_index_stack[start:start+count]
      sliced_tie = self.tie_group_map_stack[start:start+count]
      sliced_fixed_mask = self.fixed_mask_stack[start:start+count]
      sliced_fixed_tokens = self.fixed_tokens_stack[start:start+count]
      sliced_state_flat = self.state_flat_rows[start:start+count]
      sliced_state_idx = self.state_index[start:start+count]
      sliced_state_emb = self.state_embedding[start:start+count]
      
      # Rebase flat_row_offsets: subtract the offset at index start
      old_offset = self.flat_row_offsets[start]
      sliced_offsets = self.flat_row_offsets[start:start+count] - old_offset
      
      return MultistateStackPayload(
          coords_stack=sliced_coords,
          mask_stack=sliced_mask,
          residue_index_stack=sliced_ri,
          chain_index_stack=sliced_ci,
          tie_group_map_stack=sliced_tie,
          fixed_mask_stack=sliced_fixed_mask,
          fixed_tokens_stack=sliced_fixed_tokens,
          state_flat_rows=sliced_state_flat,
          flat_row_offsets=sliced_offsets,
          state_index=sliced_state_idx,
          state_embedding=sliced_state_emb,
          n_states=count,
          n_canonical=self.n_canonical,
          n_flat=count * self.n_canonical,
      )
  ```

- [ ] **1.1.3: Verify slice method with inline test**

  Run: 
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run python -c "
  import jax.numpy as jnp
  from prxteinmpnn.payloads import MultistateStackPayload
  
  s = MultistateStackPayload(
      coords_stack=jnp.zeros((4, 6, 4, 3)),
      mask_stack=jnp.ones((4, 6)),
      residue_index_stack=jnp.zeros((4, 6), dtype=jnp.int32),
      chain_index_stack=jnp.zeros((4, 6), dtype=jnp.int32),
      tie_group_map_stack=jnp.zeros((4, 6), dtype=jnp.int32),
      fixed_mask_stack=jnp.zeros((4, 6)),
      fixed_tokens_stack=jnp.zeros((4, 6), dtype=jnp.int32),
      state_flat_rows=jnp.zeros((4, 6), dtype=jnp.int32),
      flat_row_offsets=jnp.arange(4, dtype=jnp.int32) * 6,
      state_index=jnp.arange(4, dtype=jnp.int32),
      state_embedding=jnp.zeros((4, 1)),
      n_states=4, n_canonical=6, n_flat=24,
  )
  
  sliced = s.slice(1, 2)
  assert sliced.n_states == 2, f'got {sliced.n_states}'
  assert sliced.coords_stack.shape == (2, 6, 4, 3), f'got {sliced.coords_stack.shape}'
  assert sliced.flat_row_offsets[0] == 0, f'offsets not rebased: {sliced.flat_row_offsets}'
  assert sliced.n_flat == 12, f'n_flat wrong: {sliced.n_flat}'
  print('PASS: MultistateStackPayload.slice() works correctly')
  "
  ```

- [ ] **1.1.4: Commit**

  ```bash
  git add src/prxteinmpnn/payloads.py
  git commit -m "feat(payloads): add MultistateStackPayload.slice(start, count) for n_states axis slicing"
  ```

**Gate:** Inline test passes without assertion errors.

---

### Task 1.2: Add `SamplingInputs.slice_states(start: int, count: int) -> SamplingInputs`

**Files:**
- Modify: `src/prxteinmpnn/model_inputs.py`

**Steps:**

- [ ] **1.2.1: Add `slice_states()` method to `SamplingInputs`**

  After the `__init__` method in `SamplingInputs`, add:
  
  ```python
  def slice_states(self, start: int, count: int) -> "SamplingInputs":
      """Return a SamplingInputs with state_stack sliced to [start, start+count).
      
      backbone, wave_parallel, and conditioning are passed through unchanged
      (they carry no n_states axis at the SamplingInputs level).
      """
      return SamplingInputs(
          backbone=self.backbone,
          state_stack=self.state_stack.slice(start, count),
          wave_parallel=self.wave_parallel,
          conditioning=self.conditioning,
      )
  ```

- [ ] **1.2.2: Verify slice_states with inline test**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run python -c "
  import jax.numpy as jnp
  from prxteinmpnn.model_inputs import SamplingInputs, BackboneGeometry, ConditioningFeatures
  from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
  
  S, L = 4, 6
  stack = MultistateStackPayload(
      coords_stack=jnp.zeros((S, L, 4, 3)), mask_stack=jnp.ones((S, L)),
      residue_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
      chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
      tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
      fixed_mask_stack=jnp.zeros((S, L)), fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
      state_flat_rows=jnp.zeros((S, L), dtype=jnp.int32),
      flat_row_offsets=jnp.arange(S, dtype=jnp.int32) * L,
      state_index=jnp.arange(S, dtype=jnp.int32), state_embedding=jnp.zeros((S, 1)),
      n_states=S, n_canonical=L, n_flat=S*L,
  )
  bb = BackboneGeometry(
      coords=jnp.zeros((L, 4, 3)), mask=jnp.ones(L),
      residue_index=jnp.zeros(L, dtype=jnp.int32), chain_index=jnp.zeros(L, dtype=jnp.int32),
  )
  wp = WaveParallelPayload(
      wave_group_ids=jnp.zeros((L,), dtype=jnp.int32),
      wave_group_positions=jnp.zeros((L,), dtype=jnp.int32),
      wave_group_valid=jnp.ones((1,), dtype=bool),
      wave_position_valid=jnp.ones((L,), dtype=bool),
  )
  cf = ConditioningFeatures(
      fixed_tokens=jnp.zeros(L, dtype=jnp.int32),
      bias=jnp.zeros((L, 21)), ar_mask=jnp.eye(L),
  )
  si = SamplingInputs(backbone=bb, state_stack=stack, wave_parallel=wp, conditioning=cf)
  sliced = si.slice_states(1, 2)
  assert sliced.state_stack.n_states == 2, f'got {sliced.state_stack.n_states}'
  assert sliced.backbone is si.backbone  # unchanged
  print('PASS: SamplingInputs.slice_states() works correctly')
  "
  ```

- [ ] **1.2.3: Commit**

  ```bash
  git add src/prxteinmpnn/model_inputs.py
  git commit -m "feat(model_inputs): add SamplingInputs.slice_states(start, count) delegation method"
  ```

**Gate:** Inline test passes without assertion errors.

---

### Task 1.3: Add Tests for `MultistateStackPayload.slice()`

**Files:**
- Create: `tests/payloads/__init__.py`
- Create: `tests/payloads/test_multistate_stack_payload_slice.py`

**Steps:**

- [ ] **1.3.1: Create `tests/payloads/` directory and `__init__.py`**

  Run:
  ```bash
  mkdir -p /home/marielle/projects/tev_design/prxteinmpnn/tests/payloads
  touch /home/marielle/projects/tev_design/prxteinmpnn/tests/payloads/__init__.py
  ```

- [ ] **1.3.2: Write test file with 5 test functions**

  Create `/home/marielle/projects/tev_design/prxteinmpnn/tests/payloads/test_multistate_stack_payload_slice.py`:
  
  ```python
  import jax.numpy as jnp
  import pytest
  from prxteinmpnn.payloads import MultistateStackPayload
  
  
  def _make_payload(n_states, n_canonical):
      """Helper to create a MultistateStackPayload for testing."""
      return MultistateStackPayload(
          coords_stack=jnp.zeros((n_states, n_canonical, 4, 3)),
          mask_stack=jnp.ones((n_states, n_canonical)),
          residue_index_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
          chain_index_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
          tie_group_map_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
          fixed_mask_stack=jnp.zeros((n_states, n_canonical)),
          fixed_tokens_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
          state_flat_rows=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
          flat_row_offsets=jnp.arange(n_states, dtype=jnp.int32) * n_canonical,
          state_index=jnp.arange(n_states, dtype=jnp.int32),
          state_embedding=jnp.zeros((n_states, 1)),
          n_states=n_states,
          n_canonical=n_canonical,
          n_flat=n_states * n_canonical,
      )
  
  
  def test_slice_basic():
      """Test basic slicing: S=4, L=6, slice [1:3]."""
      s = _make_payload(4, 6)
      sliced = s.slice(1, 2)
      
      assert sliced.n_states == 2
      assert sliced.coords_stack.shape == (2, 6, 4, 3)
      assert sliced.state_embedding.shape == (2, 1)
      assert sliced.n_canonical == 6
      assert sliced.n_flat == 12
  
  
  def test_slice_flat_row_offsets_rebased():
      """Test that flat_row_offsets are rebased to start at 0."""
      s = _make_payload(4, 6)
      sliced = s.slice(1, 2)
      
      # Original offsets: [0, 6, 12, 18]
      # Slice [1:3] should give [6, 12] rebased to [0, 6]
      assert sliced.flat_row_offsets[0] == 0
      assert sliced.flat_row_offsets[1] == 6
  
  
  def test_slice_n_flat_recomputed():
      """Test that n_flat is correctly recomputed."""
      s = _make_payload(4, 6)
      sliced = s.slice(2, 1)  # slice [2:3]
      
      assert sliced.n_flat == 1 * 6
      assert sliced.n_canonical == 6
  
  
  def test_slice_out_of_range_raises():
      """Test that slicing out of range raises ValueError."""
      s = _make_payload(4, 6)
      
      with pytest.raises(ValueError, match="slice out of range"):
          s.slice(-1, 1)
      
      with pytest.raises(ValueError, match="slice out of range"):
          s.slice(4, 1)  # start=4 but n_states=4, so out of range
      
      with pytest.raises(ValueError, match="slice out of range"):
          s.slice(2, 3)  # start=2, count=3, total=5 > 4
  
  
  def test_slice_full_is_identity():
      """Test that slice(0, n_states) produces an equivalent payload."""
      s = _make_payload(4, 6)
      sliced = s.slice(0, 4)
      
      # Check that all arrays are element-wise equal
      assert jnp.array_equal(sliced.coords_stack, s.coords_stack)
      assert jnp.array_equal(sliced.mask_stack, s.mask_stack)
      assert jnp.array_equal(sliced.state_index, s.state_index)
      assert sliced.n_states == s.n_states
      assert sliced.n_flat == s.n_flat
  ```

- [ ] **1.3.3: Run test suite**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/payloads/test_multistate_stack_payload_slice.py -v
  ```

  Expected: All 5 tests pass.

- [ ] **1.3.4: Commit**

  ```bash
  git add tests/payloads/
  git commit -m "test(payloads): add MultistateStackPayload.slice() unit tests (5 test functions)"
  ```

**Gate:** `pytest tests/payloads/test_multistate_stack_payload_slice.py -v` passes with 5/5 tests.

---

## Phase 2: Deprecated Test Cleanup

### Task 2.1: Wrap deprecated call in test_unconditional.py line 44

**Files:**
- Modify: `tests/pipeline/test_unconditional.py`

**Steps:**

- [ ] **2.1.1: Find the test and deprecated call**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  grep -n "score_unconditional_state_vmap_exact_from_payload" tests/pipeline/test_unconditional.py
  ```

  Expected: Find at line ~44 in `test_score_unconditional_from_payload_accepts_logit_transform_fn`.

- [ ] **2.1.2: Add `import pytest` if not present**

  Check if present: `grep "^import pytest" tests/pipeline/test_unconditional.py`
  
  If missing, add at top of file.

- [ ] **2.1.3: Wrap the deprecated call with `pytest.warns()`**

  Change line ~44 from:
  ```python
  logits = m.score_unconditional_state_vmap_exact_from_payload(...)
  ```
  
  To:
  ```python
  with pytest.warns(DeprecationWarning, match="use score_unconditional_from_payload"):
      logits = m.score_unconditional_state_vmap_exact_from_payload(...)
  ```

- [ ] **2.1.4: Verify test runs**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py::test_score_unconditional_from_payload_accepts_logit_transform_fn -v
  ```

- [ ] **2.1.5: Commit**

  ```bash
  git add tests/pipeline/test_unconditional.py
  git commit -m "test(pipeline): wrap deprecated score_unconditional_state_vmap_exact_from_payload call with pytest.warns()"
  ```

**Gate:** Test passes and explicitly catches `DeprecationWarning`.

---

### Task 2.2: Wrap deprecated call in test_unconditional.py line 125

**Files:**
- Modify: `tests/pipeline/test_unconditional.py`

**Steps:**

- [ ] **2.2.1: Find the test and deprecated call**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  grep -n "score_unconditional_state_vmap_exact_from_payload" tests/pipeline/test_unconditional.py | grep -v "def test"
  ```

  Expected: Find second occurrence at line ~125 in `test_unconditional_pipeline_matches_direct_call`.

- [ ] **2.2.2: Wrap the deprecated call**

  Change the call from:
  ```python
  direct_logits = m.score_unconditional_state_vmap_exact_from_payload(...)
  ```
  
  To:
  ```python
  with pytest.warns(DeprecationWarning, match="use score_unconditional_from_payload"):
      direct_logits = m.score_unconditional_state_vmap_exact_from_payload(...)
  ```

- [ ] **2.2.3: Verify test runs**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py::test_unconditional_pipeline_matches_direct_call -v
  ```

- [ ] **2.2.4: Commit**

  ```bash
  git add tests/pipeline/test_unconditional.py
  git commit -m "test(pipeline): wrap second deprecated score_unconditional_state_vmap_exact_from_payload call with pytest.warns()"
  ```

**Gate:** Test passes and explicitly catches `DeprecationWarning`.

---

### Task 2.3: Wrap deprecated call in test_conditional.py line 46

**Files:**
- Modify: `tests/pipeline/test_conditional.py`

**Steps:**

- [ ] **2.3.1: Find the test and deprecated call**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  grep -n "score_conditional_state_vmap_exact_from_payload" tests/pipeline/test_conditional.py
  ```

  Expected: Find at line ~46 in `test_score_conditional_from_payload_accepts_logit_transform_fn`.

- [ ] **2.3.2: Add `import pytest` if not present**

  Check if present: `grep "^import pytest" tests/pipeline/test_conditional.py`
  
  If missing, add at top of file.

- [ ] **2.3.3: Wrap the deprecated call**

  Change line ~46 from:
  ```python
  logits = m.score_conditional_state_vmap_exact_from_payload(...)
  ```
  
  To:
  ```python
  with pytest.warns(DeprecationWarning, match="use score_conditional_from_payload"):
      logits = m.score_conditional_state_vmap_exact_from_payload(...)
  ```

- [ ] **2.3.4: Verify test runs**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/pipeline/test_conditional.py::test_score_conditional_from_payload_accepts_logit_transform_fn -v
  ```

- [ ] **2.3.5: Commit**

  ```bash
  git add tests/pipeline/test_conditional.py
  git commit -m "test(pipeline): wrap deprecated score_conditional_state_vmap_exact_from_payload call with pytest.warns()"
  ```

**Gate:** Test passes and explicitly catches `DeprecationWarning`.

---

## Phase 3: PayloadDispatcher Implementation

### Task 3.0: Ensure tests/run/ package exists

**Files:**
- Create: `tests/run/__init__.py`

**Steps:**

- [ ] **3.0.1: Create tests/run/ directory and __init__.py**

  Run:
  ```bash
  mkdir -p /home/marielle/projects/tev_design/prxteinmpnn/tests/run
  touch /home/marielle/projects/tev_design/prxteinmpnn/tests/run/__init__.py
  ```

- [ ] **3.0.2: Commit**

  ```bash
  git add tests/run/__init__.py
  git commit -m "chore(tests): initialize tests/run/ package for dispatcher tests"
  ```

**Gate:** `tests/run/__init__.py` exists and pytest can collect tests from `tests/run/`.

---

### Task 3.1: Implement `PayloadDispatcher.score_unconditional()`

**Files:**
- Create: `src/prxteinmpnn/run/_dispatcher.py`

**Steps:**

- [ ] **3.1.1: Create the module with PayloadDispatcher class**

  Create `/home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/run/_dispatcher.py`:
  
  ```python
  """Host-level dispatcher for multi-structure SamplingInputs scoring."""
  
  import dataclasses
  from typing import TYPE_CHECKING, Any, Callable
  
  import jax
  import jax.numpy as jnp
  
  if TYPE_CHECKING:
      from prxteinmpnn.model_inputs import SamplingInputs
      from prxteinmpnn.model.mpnn import PrxteinMPNN
  
  
  @dataclasses.dataclass(frozen=True)
  class PayloadDispatcher:
      """Host-level dispatcher for per-structure SamplingInputs iteration.
      
      Iterates over a list[SamplingInputs] (n_structures axis).
      Within each structure, dispatches to the model method directly.
      prng_key splitting is pre-computed before the host loop to guarantee
      plan-independence (identical keys whether caller uses vmap or safe_map externally).
      
      Does NOT use safe_map internally — the structure loop is a plain Python for-loop
      since SamplingInputs is a heterogeneous type (variable n_states per structure).
      The model method (score_unconditional_from_payload etc.) handles internal vmap.
      """
  
      def score_unconditional(
          self,
          model: Any,  # PrxteinMPNN, but using Any to avoid circular imports
          prng_key,  # JAX PRNG key
          stack_list,  # list[MultistateStackPayload]
          *,
          tie_group_map,
          multi_state_strategy_idx: int,
          state_weights,
          state_mapping,
          inference: bool = True,
          logit_transform_fn: Callable | None = None,
          encoder_state_fn: Callable | None = None,
      ):
          """Score each MultistateStackPayload in stack_list unconditionally.
          
          Args:
              model: PrxteinMPNN instance.
              prng_key: JAX PRNG key for the entire structure batch.
              stack_list: list of MultistateStackPayload, one per structure.
              tie_group_map: forwarded to model.score_unconditional_from_payload.
              multi_state_strategy_idx: forwarded to model method.
              state_weights: forwarded to model method.
              state_mapping: forwarded to model method.
              inference: forwarded to model method.
              logit_transform_fn: forwarded to model method.
              encoder_state_fn: forwarded to model method.
          
          Returns:
              list of Logits, one per structure (list of arrays).
          """
          # Guard for empty list
          if not stack_list:
              return []
          
          # Pre-split PRNG keys for determinism
          n = len(stack_list)
          structure_keys = jax.random.split(prng_key, n)  # shape (n, 2)
          
          results = []
          for i, stack in enumerate(stack_list):
              logits = model.score_unconditional_from_payload(
                  structure_keys[i],
                  stack,
                  tie_group_map=tie_group_map,
                  multi_state_strategy_idx=multi_state_strategy_idx,
                  state_weights=state_weights,
                  state_mapping=state_mapping,
                  inference=inference,
                  logit_transform_fn=logit_transform_fn,
                  encoder_state_fn=encoder_state_fn,
              )
              results.append(logits)
          
          return results
  
  
  __all__ = ["PayloadDispatcher"]
  ```

- [ ] **3.1.2: Verify module imports and exercise dispatch path**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run python -c "
  import jax
  import jax.numpy as jnp
  from prxteinmpnn.run._dispatcher import PayloadDispatcher
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.model_inputs import BackboneGeometry, ConditioningFeatures, SamplingInputs
  from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
  
  # Create a small model
  model = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
  
  # Create a minimal stack
  L = 6
  stack = MultistateStackPayload(
      coords_stack=jnp.zeros((2, L, 4, 3)), mask_stack=jnp.ones((2, L)),
      residue_index_stack=jnp.zeros((2, L), dtype=jnp.int32),
      chain_index_stack=jnp.zeros((2, L), dtype=jnp.int32),
      tie_group_map_stack=jnp.zeros((2, L), dtype=jnp.int32),
      fixed_mask_stack=jnp.zeros((2, L)), fixed_tokens_stack=jnp.zeros((2, L), dtype=jnp.int32),
      state_flat_rows=jnp.zeros((2, L), dtype=jnp.int32),
      flat_row_offsets=jnp.arange(2, dtype=jnp.int32) * L,
      state_index=jnp.arange(2, dtype=jnp.int32), state_embedding=jnp.zeros((2, 1)),
      n_states=2, n_canonical=L, n_flat=2*L,
  )
  
  # Call dispatcher
  dispatcher = PayloadDispatcher()
  key = jax.random.PRNGKey(0)
  results = dispatcher.score_unconditional(
      model, key, [stack],
      tie_group_map=None, multi_state_strategy_idx=0,
      state_weights=None, state_mapping=None, inference=True,
  )
  
  assert isinstance(results, list) and len(results) == 1
  assert isinstance(results[0], jax.Array)
  print('PayloadDispatcher.score_unconditional() works correctly')
  "
  ```

- [ ] **3.1.3: Commit**

  ```bash
  git add src/prxteinmpnn/run/_dispatcher.py
  git commit -m "feat(run): add PayloadDispatcher class with score_unconditional() method"
  ```

**Gate:** Module imports without errors.

---

### Task 3.2: Implement `PayloadDispatcher.score_conditional()`

**Files:**
- Modify: `src/prxteinmpnn/run/_dispatcher.py`

**Steps:**

- [ ] **3.2.1: Add `score_conditional()` method to PayloadDispatcher**

  After the `score_unconditional` method, add:
  
  ```python
  def score_conditional(
      self,
      model: Any,  # PrxteinMPNN
      prng_key,  # JAX PRNG key
      stack_list,  # list[MultistateStackPayload]
      seq_oh_stack_list,  # list of one-hot sequence arrays
      ar_mask_stack_list,  # list of AR mask arrays
      *,
      tie_group_map,
      multi_state_strategy_idx: int,
      state_weights,
      state_mapping,
      bias_flat_stack_list=None,  # optional list of bias arrays, one per structure
      inference: bool = True,
      logit_transform_fn: Callable | None = None,
      encoder_state_fn: Callable | None = None,
  ):
      """Score each MultistateStackPayload in stack_list conditionally.
      
      Args:
          model: PrxteinMPNN instance.
          prng_key: JAX PRNG key for the entire structure batch.
          stack_list: list of MultistateStackPayload, one per structure.
          seq_oh_stack_list: list of one-hot sequence arrays, aligned with stack_list.
          ar_mask_stack_list: list of AR mask arrays, aligned with stack_list.
          tie_group_map: forwarded to model.score_conditional_from_payload.
          multi_state_strategy_idx: forwarded to model method.
          state_weights: forwarded to model method.
          state_mapping: forwarded to model method.
          bias_flat_stack_list: optional list of bias arrays, one per structure. If None, bias_flat=None is passed to model method.
          inference: forwarded to model method.
          logit_transform_fn: forwarded to model method.
          encoder_state_fn: forwarded to model method.
      
      Returns:
          list of Logits, one per structure (list of arrays).
      """
      # Guard for empty list
      if not stack_list:
          return []
      
      # Validate aligned list lengths
      assert len(stack_list) == len(seq_oh_stack_list) == len(ar_mask_stack_list), \
          f"List lengths must match: {len(stack_list)}, {len(seq_oh_stack_list)}, {len(ar_mask_stack_list)}"
      
      # Pre-split PRNG keys for determinism
      n = len(stack_list)
      structure_keys = jax.random.split(prng_key, n)  # shape (n, 2)
      
      results = []
      for i, stack in enumerate(stack_list):
          bias_flat = None if bias_flat_stack_list is None else bias_flat_stack_list[i]
          logits = model.score_conditional_from_payload(
              structure_keys[i],
              stack,
              seq_oh_stack_list[i],
              ar_mask_stack_list[i],
              tie_group_map=tie_group_map,
              multi_state_strategy_idx=multi_state_strategy_idx,
              state_weights=state_weights,
              state_mapping=state_mapping,
              bias_flat=bias_flat,
              inference=inference,
              logit_transform_fn=logit_transform_fn,
              encoder_state_fn=encoder_state_fn,
          )
          results.append(logits)
      
      return results
  ```

- [ ] **3.2.2: Verify method signature**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run python -c "
  from prxteinmpnn.run._dispatcher import PayloadDispatcher
  from inspect import signature
  d = PayloadDispatcher()
  print('score_conditional signature:', signature(d.score_conditional))
  "
  ```

- [ ] **3.2.3: Commit**

  ```bash
  git add src/prxteinmpnn/run/_dispatcher.py
  git commit -m "feat(run): add PayloadDispatcher.score_conditional() method with seq/mask stacks"
  ```

**Gate:** Method is callable with expected signature.

---

## Phase 4: PayloadDispatcher Test Suite

### Task 4.1: Basic tests and determinism gate

**Files:**
- Create: `tests/run/test_payload_dispatcher.py`

**Steps:**

- [ ] **4.1.1: Create test file with basic tests**

  Create `/home/marielle/projects/tev_design/prxteinmpnn/tests/run/test_payload_dispatcher.py`:
  
  ```python
  """Tests for PayloadDispatcher."""
  
  import jax
  import jax.numpy as jnp
  import pytest
  
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.model_inputs import BackboneGeometry, ConditioningFeatures, SamplingInputs
  from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
  from prxteinmpnn.run._dispatcher import PayloadDispatcher
  
  
  @pytest.fixture
  def model():
      """Create a PrxteinMPNN instance for testing."""
      import jax
      from prxteinmpnn.model.mpnn import PrxteinMPNN
      return PrxteinMPNN(
          16, 16, 16, 1, 1, 6,
          key=jax.random.PRNGKey(0),
      )
  
  
  def _make_sampling_inputs(n_states=2, L=6):
      """Helper to create a SamplingInputs for testing."""
      stack = MultistateStackPayload(
          coords_stack=jnp.zeros((n_states, L, 4, 3)),
          mask_stack=jnp.ones((n_states, L)),
          residue_index_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
          chain_index_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
          tie_group_map_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
          fixed_mask_stack=jnp.zeros((n_states, L)),
          fixed_tokens_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
          state_flat_rows=jnp.zeros((n_states, L), dtype=jnp.int32),
          flat_row_offsets=jnp.arange(n_states, dtype=jnp.int32) * L,
          state_index=jnp.arange(n_states, dtype=jnp.int32),
          state_embedding=jnp.zeros((n_states, 1)),
          n_states=n_states,
          n_canonical=L,
          n_flat=n_states * L,
      )
      backbone = BackboneGeometry(
          coords=jnp.zeros((L, 4, 3)),
          mask=jnp.ones(L),
          residue_index=jnp.zeros(L, dtype=jnp.int32),
          chain_index=jnp.zeros(L, dtype=jnp.int32),
      )
      wave = WaveParallelPayload(
          wave_group_ids=jnp.zeros((L,), dtype=jnp.int32),
          wave_group_positions=jnp.zeros((L,), dtype=jnp.int32),
          wave_group_valid=jnp.ones((1,), dtype=bool),
          wave_position_valid=jnp.ones((L,), dtype=bool),
      )
      cond = ConditioningFeatures(
          fixed_tokens=jnp.zeros(L, dtype=jnp.int32),
          bias=jnp.zeros((L, 21)),
          ar_mask=jnp.eye(L),
      )
      return SamplingInputs(backbone=backbone, state_stack=stack, wave_parallel=wave, conditioning=cond)
  
  
  def test_dispatcher_score_unconditional_basic(model):
      """Test basic score_unconditional dispatch with 2 structures."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(0)
      # Extract .state_stack from SamplingInputs to get MultistateStackPayload
      stack_list = [_make_sampling_inputs().state_stack, _make_sampling_inputs().state_stack]
      
      results = dispatcher.score_unconditional(
          model, key, stack_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      assert isinstance(results, list)
      assert len(results) == 2
      assert all(isinstance(r, jax.Array) for r in results)
  
  
  def test_dispatcher_key_split_determinism(model):
      """Test that identical prng_key produces identical results."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(42)
      stack_list = [_make_sampling_inputs().state_stack]
      
      results1 = dispatcher.score_unconditional(
          model, key, stack_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      results2 = dispatcher.score_unconditional(
          model, key, stack_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      assert jnp.allclose(results1[0], results2[0], rtol=1e-6, atol=1e-6)
  
  
  def test_dispatcher_different_keys_differ(model):
      """Test that different prng_keys produce different results when inference=False (stochastic)."""
      dispatcher = PayloadDispatcher()
      key1 = jax.random.PRNGKey(0)
      key2 = jax.random.PRNGKey(999)
      stack_list = [_make_sampling_inputs().state_stack]
      
      # Note: With inference=False, dropout and other stochastic ops use the key.
      # With inference=True, the key is typically unused and results are deterministic.
      results1 = dispatcher.score_unconditional(
          model, key1, stack_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=False,  # Enable stochastic operations (dropout)
      )
      
      results2 = dispatcher.score_unconditional(
          model, key2, stack_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=False,  # Enable stochastic operations (dropout)
      )
      
      # Results should differ because keys are different and model uses stochasticity
      assert not jnp.allclose(results1[0], results2[0], rtol=1e-6, atol=1e-6)
  
  
  def test_dispatcher_empty_list_returns_empty(model):
      """Test that empty inputs_list returns empty list."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(0)
      
      results = dispatcher.score_unconditional(
          model, key, [],
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      assert results == []
  ```

- [ ] **4.1.2: Run tests**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/run/test_payload_dispatcher.py -v -k "basic or determinism or empty"
  ```

- [ ] **4.1.3: Commit**

  ```bash
  git add tests/run/test_payload_dispatcher.py
  git commit -m "test(run): add PayloadDispatcher tests (basic, determinism, empty list gates)"
  ```

**Gate:** `pytest tests/run/test_payload_dispatcher.py::test_dispatcher_score_unconditional_basic -v` passes.

---

### Task 4.2: Parity test (single-element list vs direct call)

**Steps:**

- [ ] **4.2.1: Add parity test to test_payload_dispatcher.py**

  Add after the empty-list test:
  
  ```python
  def test_dispatcher_parity_single_vs_list(model):
      """Test that single-element dispatcher call matches direct model call."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(123)
      stack = _make_sampling_inputs().state_stack
      
      # Call via dispatcher with 1-element list
      dispatcher_results = dispatcher.score_unconditional(
          model, key, [stack],
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      # Call directly (must split key identically)
      direct_key = jax.random.split(key, 1)[0]  # Replicate dispatcher's split logic
      direct_logits = model.score_unconditional_from_payload(
          direct_key, stack,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      # Compare
      assert jnp.allclose(dispatcher_results[0], direct_logits, rtol=1e-6, atol=1e-6)
  ```

- [ ] **4.2.2: Run parity test**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/run/test_payload_dispatcher.py::test_dispatcher_parity_single_vs_list -v
  ```

- [ ] **4.2.3: Commit**

  ```bash
  git add tests/run/test_payload_dispatcher.py
  git commit -m "test(run): add PayloadDispatcher parity test (single-element list vs direct call)"
  ```

**Gate:** Parity test passes (atol=1e-6).

---

### Task 4.3: Conditional and edge-case tests

**Steps:**

- [ ] **4.3.1: Add conditional test to test_payload_dispatcher.py**

  Add:
  
  ```python
  def test_dispatcher_conditional_basic(model):
      """Test basic score_conditional dispatch with 2 structures."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(0)
      L = 6
      
      stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack for _ in range(2)]
      seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]
      ar_mask_list = [jnp.eye(L) for _ in range(2)]
      
      results = dispatcher.score_conditional(
          model, key, stack_list,
          seq_oh_stack_list=seq_oh_list,
          ar_mask_stack_list=ar_mask_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          inference=True,
      )
      
      assert isinstance(results, list)
      assert len(results) == 2
  
  
  def test_dispatcher_mismatched_list_lengths_raises(model):
      """Test that mismatched input list lengths raise AssertionError."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(0)
      L = 6
      
      stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack]  # 1 element
      seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]  # 2 elements - mismatch!
      ar_mask_list = [jnp.eye(L)]  # 1 element
      
      with pytest.raises(AssertionError, match="List lengths must match"):
          dispatcher.score_conditional(
              model, key, stack_list,
              seq_oh_stack_list=seq_oh_list,
              ar_mask_stack_list=ar_mask_list,
              tie_group_map=None,
              multi_state_strategy_idx=0,
              state_weights=None,
              state_mapping=None,
              inference=True,
          )


  def test_dispatcher_conditional_with_bias_flat(model):
      """Test score_conditional with bias_flat_stack_list provided."""
      dispatcher = PayloadDispatcher()
      key = jax.random.PRNGKey(0)
      L = 6
      
      stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack for _ in range(2)]
      seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]
      ar_mask_list = [jnp.eye(L) for _ in range(2)]
      bias_flat_list = [jnp.zeros((L,)) for _ in range(2)]
      
      results = dispatcher.score_conditional(
          model, key, stack_list,
          seq_oh_stack_list=seq_oh_list,
          ar_mask_stack_list=ar_mask_list,
          tie_group_map=None,
          multi_state_strategy_idx=0,
          state_weights=None,
          state_mapping=None,
          bias_flat_stack_list=bias_flat_list,
          inference=True,
      )
      
      assert isinstance(results, list)
      assert len(results) == 2
  ```

- [ ] **4.3.2: Run full dispatcher test suite**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest tests/run/test_payload_dispatcher.py -v
  ```

  Expected: All tests pass.

- [ ] **4.3.3: Commit**

  ```bash
  git add tests/run/test_payload_dispatcher.py
  git commit -m "test(run): add PayloadDispatcher conditional test and edge cases (mismatched list lengths)"
  ```

**Gate:** `pytest tests/run/test_payload_dispatcher.py -v` passes with 7+ test functions.

---

## Phase 5: Integration and Full Test Suite

### Task 5.1: Verify pipeline tests with deprecated alias wrapping

**Steps:**

- [ ] **5.1.1: Run pipeline tests**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/pipeline/test_unconditional.py \
    tests/pipeline/test_conditional.py \
    -v --tb=short
  ```

  Expected: All tests pass, deprecated aliases emit and are caught by `pytest.warns()`.

- [ ] **5.1.2: Run with strict deprecation flag**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/pipeline/test_unconditional.py \
    tests/pipeline/test_conditional.py \
    -q
  ```

  Expected: All tests pass silently (no uncaught deprecation warnings).

**Gate:** Both commands pass.

---

### Task 5.2: Verify payload and dispatcher test suites together

**Steps:**

- [ ] **5.2.1: Run combined test suite**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/payloads/test_multistate_stack_payload_slice.py \
    tests/run/test_payload_dispatcher.py \
    -v --tb=short
  ```

  Expected: All tests pass (11+ test functions total).

- [ ] **5.2.2: Run with quiet output**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/payloads/test_multistate_stack_payload_slice.py \
    tests/run/test_payload_dispatcher.py \
    -q
  ```

  Expected: All tests pass.

**Gate:** Both commands pass with no errors.

---

### Task 5.3: Verify imports and no regressions

**Steps:**

- [ ] **5.3.1: Check all test imports resolve**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/payloads/ \
    tests/pipeline/test_unconditional.py \
    tests/pipeline/test_conditional.py \
    tests/run/test_payload_dispatcher.py \
    --co -q
  ```

  Expected: No import errors, all tests are collected.

- [ ] **5.3.2: Run full test sweep (broad regression check)**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  PYTHONPATH=src uv run pytest \
    tests/payloads/ \
    tests/pipeline/ \
    tests/run/test_payload_dispatcher.py \
    -q --tb=line 2>&1 | head -100
  ```

  Expected: All tests pass or only pre-existing failures (no new failures introduced).

- [ ] **5.3.3: Final commit summary**

  Run:
  ```bash
  cd /home/marielle/projects/tev_design/prxteinmpnn && \
  git log --oneline -20
  ```

  Expected: See commits for all completed tasks (one per task, up to ~16 total: 1 from Task 3.0, 1 from each of Tasks 1.1–5.3, plus the initial plan-fix commit).

**Gate:** No new regressions detected; all integration tests pass.

---

## Success Criteria

- ✅ `MultistateStackPayload.slice()` implemented and tested (5 tests)
- ✅ `SamplingInputs.slice_states()` implemented and tested (inline)
- ✅ 3 deprecated test calls wrapped with `pytest.warns()`
- ✅ `PayloadDispatcher` with both `score_unconditional()` and `score_conditional()` methods
- ✅ `PayloadDispatcher` test suite with 6+ tests (basic, determinism, parity, conditional, empty list, mismatched lengths)
- ✅ All tests pass without regressions
- ✅ No import cycles detected
- ✅ ~16 atomic commits created (one per task: Task 3.0 + Tasks 1.1–5.3, consistent with Task 5.3.3 expected output)

---

## Out of Scope

- Changes to `mpnn.py`, `ligand_mpnn.py`, or any `pipeline/*.py` file
- `WaveParallelPayload.slice()` (not needed)
- `BatchPlan`-aware dispatch inside `PayloadDispatcher` (deferred to PR-C)
- Wiring dispatcher into `run/sampling.py` or `run/scoring.py` (deferred)
- `n_structures` leading axis on any payload type
- LigandMPNN variant tests
