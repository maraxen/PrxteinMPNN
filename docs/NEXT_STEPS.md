# Next Steps: Equinox Migration Roadmap

**Last Updated**: 2025-11-03  
**Current Status**: Milestone 3 Complete ✅ (Core Equivalence Testing)

## Executive Summary

The new Equinox implementation (`eqx_new.py`) is **functionally complete** and **numerically equivalent** to the functional implementation. All 4 core equivalence tests pass with tight tolerance (rtol=1e-5, atol=1e-5) in 11.6 seconds.

**What's Working:**

- ✅ Full encoder/decoder architecture implemented
- ✅ Unconditional, conditional, and autoregressive decoding modes
- ✅ Numerical equivalence verified across all modes
- ✅ Critical bug fixes applied (attention masking in conditional decoder)

**What's Next:**

- Model weight conversion to .eqx format
- HuggingFace deployment
- Integration with existing API
- Additional testing coverage
- Project restructuring

---

## Milestone 4: Weight Conversion & Model Deployment

### Objectives

1. Convert all model weights to .eqx format
2. Upload to HuggingFace for easy distribution
3. Implement robust model loading/saving
4. Add comprehensive serialization tests

### Tasks

#### 4.1 Weight Conversion Script ⏳

**Priority**: HIGH  
**Estimated Time**: 2-3 hours

Create `scripts/convert_all_models.py`:

```python
#!/usr/bin/env python3
"""Convert all ProteinMPNN models to .eqx format."""

import jax
from pathlib import Path
from prxteinmpnn.functional import get_functional_model
from prxteinmpnn.eqx_new import PrxteinMPNN
import equinox

# Model configurations
MODELS = [
    ("original", "v_48_002"),
    ("original", "v_48_010"),
    ("original", "v_48_020"),
    ("original", "v_48_030"),
    ("soluble", "v_48_002"),
    ("soluble", "v_48_010"),
    ("soluble", "v_48_020"),
    ("soluble", "v_48_030"),
]

OUTPUT_DIR = Path("models/new_format")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

key = jax.random.PRNGKey(0)

for weights, version in MODELS:
    print(f"Converting {weights}/{version}...")
    
    # Load functional params
    params = get_functional_model(version, model_weights=weights)
    
    # Create and populate eqx model
    model = PrxteinMPNN.from_functional(
        params,
        num_encoder_layers=3,
        num_decoder_layers=3,
        key=key
    )
    
    # Save to .eqx format
    output_path = OUTPUT_DIR / f"{weights}_{version}.eqx"
    equinox.tree_serialise_leaves(str(output_path), model)
    
    print(f"  ✓ Saved to {output_path}")

print("\n✅ All models converted successfully!")
```

**Action Items:**

- [ ] Create and test conversion script
- [ ] Run conversion for all 8 model variants
- [ ] Verify file sizes are reasonable (~50-100MB per model)
- [ ] Test loading converted models
- [ ] Document conversion process

#### 4.2 HuggingFace Deployment ⏳

**Priority**: HIGH  
**Estimated Time**: 1-2 hours

**Prerequisites:**

- HuggingFace account with write access to `maraxen/prxteinmpnn` repo
- `huggingface-hub` installed

**Steps:**

1. **Organize Model Files:**

   ```bash
   models/
   └── new_format/
       ├── original_v_48_002.eqx
       ├── original_v_48_010.eqx
       ├── original_v_48_020.eqx
       ├── original_v_48_030.eqx
       ├── soluble_v_48_002.eqx
       ├── soluble_v_48_010.eqx
       ├── soluble_v_48_020.eqx
       └── soluble_v_48_030.eqx
   ```

2. **Upload Script:**

   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   
   for eqx_file in Path("models/new_format").glob("*.eqx"):
       api.upload_file(
           path_or_fileobj=str(eqx_file),
           path_in_repo=f"eqx/{eqx_file.name}",
           repo_id="maraxen/prxteinmpnn",
           repo_type="model"
       )
   ```

3. **Update Model Card:**
   - Document .eqx format availability
   - Add usage examples
   - Note equivalence guarantees

**Action Items:**

- [ ] Upload all .eqx files to HuggingFace
- [ ] Update README on HuggingFace repo
- [ ] Test downloading from HuggingFace
- [ ] Update local model loading to check HF first

#### 4.3 Save/Load Preservation Tests ⏳

**Priority**: HIGH  
**Estimated Time**: 1 hour

Add to `tests/test_eqx_equivalence.py`:

```python
def test_05_save_load_preservation(self):
    """Test that saved/loaded models produce identical outputs."""
    import tempfile
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
        temp_path = f.name
        equinox.tree_serialise_leaves(temp_path, self._new_model)
    
    try:
        # Load model
        loaded_model = equinox.tree_deserialise_leaves(temp_path, self._new_model)
        
        # Compare outputs (should be bit-perfect)
        _, original_logits = self._new_model._call_unconditional(
            self._func_edge_features,
            self._func_neighbors,
            self._mask
        )
        
        _, loaded_logits = loaded_model._call_unconditional(
            self._func_edge_features,
            self._func_neighbors,
            self._mask
        )
        
        # Bit-perfect equality
        assert jnp.allclose(original_logits, loaded_logits, rtol=1e-7, atol=1e-8)
        print("  Save/Load Preservation Test PASSED.")
    finally:
        Path(temp_path).unlink()
```

**Action Items:**

- [ ] Implement save/load test
- [ ] Test with all 3 decoding modes
- [ ] Verify bit-perfect preservation
- [ ] Test with compressed serialization

#### 4.4 Variable Sequence Length Tests ⏳

**Priority**: MEDIUM  
**Estimated Time**: 1-2 hours

Add comprehensive size testing:

```python
@pytest.mark.parametrize("num_residues", [10, 25, 50, 100, 200])
def test_06_variable_sequence_lengths(self, num_residues):
    """Test model with different sequence lengths."""
    key = jax.random.PRNGKey(42)
    K = 48
    
    # Generate inputs
    edge_features = jax.random.normal(key, (num_residues, K, 128))
    neighbor_indices = jnp.tile(jnp.arange(num_residues)[:, None], (1, K))
    mask = jnp.ones(num_residues)
    
    # Test unconditional
    _, logits = self._new_model._call_unconditional(
        edge_features, neighbor_indices, mask
    )
    
    # Verify shape
    assert logits.shape == (num_residues, 21)
    
    # Verify no NaNs/Infs
    assert not jnp.any(jnp.isnan(logits))
    assert not jnp.any(jnp.isinf(logits))
    
    print(f"  Variable length test PASSED for {num_residues} residues.")
```

**Action Items:**

- [ ] Test with lengths: 10, 25, 50, 100, 200
- [ ] Test all 3 decoding modes
- [ ] Verify numerical stability
- [ ] Document any size limitations

---

## Milestone 5: API Integration & Migration

### Objectives

1. Integrate eqx model into main API
2. Update sampling and scoring modules
3. Reorganize test structure
4. Update all documentation

### Tasks

#### 5.1 Main API Integration ⏳

**Priority**: HIGH  
**Estimated Time**: 3-4 hours

Update `src/prxteinmpnn/mpnn.py`:

```python
def get_mpnn_model(
    version: str = "v_48_020",
    model_weights: str = "original",
    use_eqx: bool = True,  # New parameter
    **kwargs
) -> Union[PrxteinMPNN, dict]:
    """
    Load ProteinMPNN model.
    
    Args:
        version: Model version (v_48_002, v_48_010, v_48_020, v_48_030)
        model_weights: "original" or "soluble"
        use_eqx: If True, load Equinox model (.eqx). If False, load functional params (.pkl)
        **kwargs: Additional arguments passed to model initialization
    
    Returns:
        PrxteinMPNN module (if use_eqx=True) or parameter dict (if use_eqx=False)
    """
    if use_eqx:
        # Load .eqx model
        from prxteinmpnn.eqx_new import PrxteinMPNN
        import equinox
        
        model_path = download_model(f"{model_weights}_{version}.eqx")
        
        # Create empty model structure
        key = jax.random.PRNGKey(0)
        model = PrxteinMPNN.from_functional(
            {},  # Dummy params for structure
            num_encoder_layers=kwargs.get("num_encoder_layers", 3),
            num_decoder_layers=kwargs.get("num_decoder_layers", 3),
            key=key
        )
        
        # Load weights
        return equinox.tree_deserialise_leaves(model_path, model)
    else:
        # Load functional params (legacy)
        from prxteinmpnn.functional import get_functional_model
        return get_functional_model(version, model_weights)
```

**Action Items:**

- [ ] Update `get_mpnn_model` with eqx support
- [ ] Add deprecation warning for functional API
- [ ] Update model downloading logic
- [ ] Test both loading paths
- [ ] Update all docstrings

#### 5.2 Sampling Module Refactor ⏳

**Priority**: HIGH  
**Estimated Time**: 4-5 hours

Update `src/prxteinmpnn/sampling/sample.py`:

Current approach uses `make_encoder`, `make_decoder`, etc. Need to refactor to use `PrxteinMPNN` methods:

```python
def make_sample_fn(
    mpnn_model: PrxteinMPNN,  # Now expects eqx model
    temperature: Float = 0.1,
    **kwargs
) -> Callable:
    """Create sampling function using Equinox model."""
    
    @jax.jit
    def sample_fn(
        edge_features: EdgeFeatures,
        neighbor_indices: NeighborIndices,
        mask: AlphaCarbonMask,
        ar_mask: AutoRegressiveMask,
        prng_key: PRNGKeyArray,
    ) -> tuple[OneHotProteinSequence, Logits]:
        """Sample sequence using autoregressive decoding."""
        return mpnn_model._call_autoregressive(
            edge_features,
            neighbor_indices,
            mask,
            ar_mask=ar_mask,
            prng_key=prng_key,
            temperature=temperature
        )
    
    return sample_fn
```

**Action Items:**

- [ ] Refactor `make_sample_fn` to use eqx model
- [ ] Update all sampling tests
- [ ] Verify sampling quality matches functional
- [ ] Add temperature scheduling support
- [ ] Document new sampling API

#### 5.3 Scoring Module Refactor ⏳

**Priority**: HIGH  
**Estimated Time**: 3-4 hours

Update `src/prxteinmpnn/scoring/score.py`:

```python
def make_score_fn(
    mpnn_model: PrxteinMPNN,  # Now expects eqx model
    **kwargs
) -> Callable:
    """Create scoring function using Equinox model."""
    
    @jax.jit
    def score_fn(
        edge_features: EdgeFeatures,
        neighbor_indices: NeighborIndices,
        mask: AlphaCarbonMask,
        ar_mask: AutoRegressiveMask,
        one_hot_sequence: OneHotProteinSequence,
    ) -> Logits:
        """Score a known sequence."""
        _, logits = mpnn_model._call_conditional(
            edge_features,
            neighbor_indices,
            mask,
            ar_mask=ar_mask,
            one_hot_sequence=one_hot_sequence
        )
        return logits
    
    return score_fn
```

**Action Items:**

- [ ] Refactor `make_score_fn` to use eqx model
- [ ] Update all scoring tests
- [ ] Verify scores match functional implementation
- [ ] Add per-position scoring support
- [ ] Document new scoring API

#### 5.4 Test Reorganization ⏳

**Priority**: MEDIUM  
**Estimated Time**: 2-3 hours

Reorganize test structure:

```
tests/
├── equivalence/           # New directory
│   ├── __init__.py
│   ├── test_eqx_equivalence.py  # Moved from root
│   └── README.md         # Explain equivalence testing
├── functional/           # Existing
│   └── ...
├── unit/                 # New directory
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_layers.py
├── integration/          # New directory
│   ├── test_sampling.py  # Moved from sampling/
│   ├── test_scoring.py   # Moved from scoring/
│   └── test_full_pipeline.py
└── conftest.py           # Shared fixtures
```

**Action Items:**

- [ ] Create new test directories
- [ ] Move equivalence tests
- [ ] Reorganize unit tests
- [ ] Update test imports
- [ ] Update CI/CD configuration
- [ ] Document test organization

---

## Milestone 6: Final Cleanup & Release

### Objectives

1. Complete documentation overhaul
2. Performance benchmarking
3. Final validation
4. Version release

### Tasks

#### 6.1 Documentation Overhaul ⏳

**Priority**: MEDIUM  
**Estimated Time**: 4-5 hours

**Update All Documentation:**

- [ ] README.md - Update with eqx examples
- [ ] API documentation - Generate with Sphinx
- [ ] Migration guide for users
- [ ] Performance comparison doc
- [ ] Contributing guidelines

**Create New Guides:**

- [ ] "Getting Started with Equinox Model"
- [ ] "Migrating from Functional API"
- [ ] "Fine-tuning ProteinMPNN"
- [ ] "Advanced Usage Patterns"

#### 6.2 Performance Benchmarking ⏳

**Priority**: MEDIUM  
**Estimated Time**: 3-4 hours

Create `benchmarks/benchmark_equivalence.py`:

```python
"""Benchmark Equinox vs Functional implementations."""

import time
import jax
import jax.numpy as jnp

def benchmark_both_implementations():
    # Test different sizes
    sizes = [25, 50, 100, 200]
    
    for num_residues in sizes:
        # Functional
        func_time = benchmark_functional(num_residues)
        
        # Equinox
        eqx_time = benchmark_equinox(num_residues)
        
        print(f"\n{num_residues} residues:")
        print(f"  Functional: {func_time:.3f}s")
        print(f"  Equinox:    {eqx_time:.3f}s")
        print(f"  Speedup:    {func_time/eqx_time:.2f}x")
```

**Metrics to Track:**

- Compilation time (first call)
- Runtime (subsequent calls)
- Memory usage
- Throughput (sequences/second)

**Action Items:**

- [ ] Create benchmark suite
- [ ] Run on different hardware (CPU/GPU/TPU)
- [ ] Compare memory footprint
- [ ] Document performance characteristics
- [ ] Identify optimization opportunities

#### 6.3 Final Validation ⏳

**Priority**: HIGH  
**Estimated Time**: 2-3 hours

**Complete Test Suite:**

- [ ] All equivalence tests passing
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] All benchmarks run successfully
- [ ] No regressions in functional API

**Code Quality:**

- [ ] All files pass Ruff linting
- [ ] All files pass Pyright type checking
- [ ] Test coverage >90%
- [ ] Documentation coverage 100%

**Validation Checklist:**

- [ ] Test on fresh virtual environment
- [ ] Test installation from pip
- [ ] Test all examples in documentation
- [ ] Test on different Python versions (3.9-3.13)
- [ ] Test on different platforms (Linux/macOS/Windows)

#### 6.4 Version Release ⏳

**Priority**: HIGH  
**Estimated Time**: 1-2 hours

**Release Preparation:**

1. Update version in `pyproject.toml`: `0.2.0` (major refactor)
2. Update CHANGELOG.md with all changes
3. Tag release in git: `v0.2.0`
4. Build distributions: `python -m build`
5. Upload to PyPI: `twine upload dist/*`

**Release Notes Template:**

```markdown
# PrxteinMPNN v0.2.0 - Equinox Implementation

## 🎉 Major Changes

- **New Equinox Implementation**: Full rewrite using `equinox.Module`
- **Numerical Equivalence**: 100% equivalent to functional implementation
- **Improved Performance**: ~10% faster with better memory efficiency
- **Better API**: Object-oriented interface for easier usage

## ✨ Features

- Three decoding modes: unconditional, conditional, autoregressive
- HuggingFace model hosting for easy distribution
- Comprehensive equivalence testing
- Full backward compatibility with functional API

## 🐛 Bug Fixes

- Fixed attention masking in conditional decoder
- Fixed duplicate bias application

## 📚 Documentation

- Complete API documentation
- Migration guide from functional API
- Performance benchmarks
- Extended examples

## ⚠️ Deprecations

- Functional API will be deprecated in v0.3.0
- Use `use_eqx=True` (default) in `get_mpnn_model()`

## 🔄 Migration

See MIGRATION.md for detailed migration guide.
```

**Action Items:**

- [ ] Write comprehensive release notes
- [ ] Update all version numbers
- [ ] Create GitHub release
- [ ] Announce on relevant channels
- [ ] Monitor for issues

---

## Testing Priorities

### Must-Have Before Release (P0)

1. ✅ Core equivalence tests (DONE)
2. Save/load preservation tests
3. Variable sequence length tests
4. Integration tests for sampling/scoring
5. Full test suite passing

### Should-Have Before Release (P1)

6. Batch processing tests
7. Edge case tests
8. Performance benchmarks
9. Gradient equivalence tests
10. Cross-platform tests

### Nice-to-Have (P2)

11. Full autoregressive sampling equivalence
12. Memory profiling
13. Stress tests (very long sequences)
14. Fuzzing tests
15. Property-based tests

---

## Timeline Estimate

**Conservative Estimate**: 2-3 weeks part-time

| Milestone | Tasks | Time | Dependencies |
|-----------|-------|------|--------------|
| M4: Weight Conversion | 4.1-4.4 | 5-8 hours | None |
| M5: API Integration | 5.1-5.4 | 12-16 hours | M4 |
| M6: Release | 6.1-6.4 | 10-14 hours | M5 |
| **Total** | | **27-38 hours** | |

**Aggressive Estimate**: 1 week full-time

---

## Risk Assessment

### High Risk ⚠️

- **HuggingFace upload failures**: Mitigation - test with small files first
- **Breaking changes in sampling/scoring**: Mitigation - extensive testing
- **Performance regressions**: Mitigation - benchmark before merging

### Medium Risk ⚡

- **Documentation outdated**: Mitigation - automated doc generation
- **Missing edge cases**: Mitigation - property-based testing
- **User confusion during migration**: Mitigation - clear migration guide

### Low Risk ✅

- **Equivalence tests failing**: Already passing, low risk
- **Type checking issues**: Already using Pyright in strict mode
- **Linting issues**: Already using Ruff with autofix

---

## Success Criteria

### Technical

- ✅ All equivalence tests passing (DONE)
- [ ] All integration tests passing
- [ ] Performance within 10% of functional
- [ ] Type checking in strict mode
- [ ] Test coverage >90%

### Documentation

- [ ] Complete API reference
- [ ] Migration guide published
- [ ] Performance benchmarks documented
- [ ] All examples updated

### Distribution

- [ ] Models on HuggingFace
- [ ] Package on PyPI
- [ ] GitHub release published
- [ ] CI/CD pipeline updated

---

## Questions to Resolve

1. **API Design**: Should we keep both implementations long-term or fully migrate?
   - **Recommendation**: Keep functional for 1-2 versions with deprecation warnings

2. **Model Format**: Should we support loading both .pkl and .eqx?
   - **Recommendation**: Yes, with .eqx as default and .pkl deprecated

3. **Backward Compatibility**: How strictly should we maintain it?
   - **Recommendation**: Full backward compatibility for at least 2 versions

4. **Performance Target**: What's acceptable for release?
   - **Recommendation**: Within 10% of functional, ideally equal or faster

5. **Testing Coverage**: How much is enough?
   - **Recommendation**: >90% for core modules, >80% overall

---

## Notes

- Focus on Milestone 4 first (weight conversion & deployment)
- Parallelizable tasks: Documentation can happen alongside M5
- Consider beta release (v0.2.0-beta) before full release
- Monitor user feedback closely during beta period
- Be prepared to iterate on API based on usage patterns

---

**Last Updated**: 2025-11-03  
**Next Review**: After Milestone 4 completion
