# Quick Start: GMM Stability Features

## TL;DR

Your GMM fitting now automatically includes:
✅ Positive variance constraints (prevents numerical errors)
✅ Weight normalization (valid probabilities)
✅ Component pruning (removes degenerate components)
✅ BIC calculation (model selection metric)

## What Changed?

### Before

```python
# Old code - no changes needed!
from prxteinmpnn.ensemble.gmm import make_fit_gmm_in_memory

fit_gmm = make_fit_gmm_in_memory(n_components=10)
result = fit_gmm(data, key)
```

### After

```python
# Same code, but now with:
# - Automatic variance stabilization
# - Automatic component pruning
# - BIC logging
# - Enhanced convergence monitoring

from prxteinmpnn.ensemble.gmm import make_fit_gmm_in_memory

fit_gmm = make_fit_gmm_in_memory(n_components=10)
result = fit_gmm(data, key)
# ✨ Enhanced stability automatically applied!
```

## New Logging Output

You'll now see richer logging:

```log
INFO: Fitting GMM using in-memory EM...
INFO: EM iteration 1/100: log-likelihood = -1234.56, diff = inf, min_weight = 0.05, max_weight = 0.25
INFO: EM iteration 2/100: log-likelihood = -1156.78, diff = 77.78, min_weight = 0.04, max_weight = 0.28
...
INFO: GMM fitting finished in 45 iterations. Converged: True, BIC: 2589.12
WARNING: Pruned 2 components with extreme weights. Remaining: 8
```

## New Utility Functions

### Check Model Quality with BIC

```python
from prxteinmpnn.ensemble.gmm import compute_bic

bic = compute_bic(
    log_likelihood=result.log_likelihood,
    n_samples=data.shape[0],
    n_components=result.gmm.n_components,
    n_features=result.gmm.n_features,
    covariance_type="diag"
)
print(f"BIC: {float(bic):.2f}")  # Lower is better!
```

### Manual Component Pruning

```python
from prxteinmpnn.ensemble.gmm import prune_components

# Already done automatically, but you can do it manually too
pruned_gmm, n_removed = prune_components(
    gmm=result.gmm,
    min_weight=0.001,  # Remove components < 0.1%
    max_weight=0.99    # Remove dominant components > 99%
)
print(f"Removed {int(n_removed)} components")
```

## Why These Changes?

### Problem: Unstable GMM Components

Overlapping protein conformational states (energy wells) can cause:

- ❌ Near-zero variances → singular matrices → NaN errors
- ❌ Negative variances → invalid Gaussians
- ❌ Degenerate components → poor DBSCAN clustering

### Solution: Automatic Stabilization

- ✅ **Softplus variance transform**: Guarantees positive variances
- ✅ **Component pruning**: Removes unstable components
- ✅ **Weight normalization**: Ensures valid probabilities

## Impact on Your Protein Analysis

### GMM Stage

- More stable component estimates
- Fewer numerical errors
- Better capture of overlapping states

### DBSCAN Stage

- Cleaner input (pruned components)
- Better state identification
- Fewer false positives

### Final Results

- More reliable entropy measurements
- Accurate density matrices
- Robust transition state identification

## When to Tune Parameters

### Too Many Components Pruned?

```python
# Reduce the number of components
fit_gmm = make_fit_gmm_in_memory(
    n_components=5,  # Instead of 10
    kmeans_max_iters=200  # More K-means iterations
)
```

### Components Too Unstable?

```python
# Increase regularization
fit_gmm = make_fit_gmm_in_memory(
    n_components=10,
    covariance_regularization=1e-5  # Default is 1e-6
)
```

### Need More Control?

See `docs/GMM_STABILITY.md` for detailed documentation.

## Testing

All new features are tested:

```bash
uv run pytest tests/ensemble/test_gmm_stability.py -v
# 11 passed in 2.69s ✅
```

## Questions?

- 📚 Full docs: `docs/GMM_STABILITY.md`
- 🔍 Summary: `docs/STABILITY_SUMMARY.md`
- 🧪 Tests: `tests/ensemble/test_gmm_stability.py`

## Bottom Line

**You don't need to change your code.** The stability improvements work automatically and make your GMM→DBSCAN protein analysis more robust! 🎉
