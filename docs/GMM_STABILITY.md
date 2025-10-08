# GMM Stability Improvements for Protein Conformational Analysis

## Overview

This document describes the stability enhancements implemented in the Gaussian Mixture Model (GMM) fitting pipeline for protein conformational state analysis. These improvements ensure robust fitting when modeling overlapping conformational "energy wells" that are subsequently refined by DBSCAN clustering.

## The Analysis Pipeline

### 1. GMM: Capturing the Energy Landscape

The GMM models the continuous probability landscape of protein conformational states:

- **Overlapping Components**: Each GMM component represents a potential conformational microstate or energy well
- **Soft Assignments**: Provides probabilistic assignments, capturing conformational heterogeneity
- **Residue-Level Resolution**: Can model per-residue conformational variations

### 2. DBSCAN: Coarse-Graining and Noise Removal

DBSCAN refines the GMM results by:

- **Eliminating Isolated States**: Removes spurious/noise components that don't form coherent clusters
- **Preserving Overlaps**: Maintains information about overlapping energy wells (transition states)
- **Coarse-Graining**: Merges nearby GMM components into biologically meaningful conformational states

### 3. Result: Density Matrix Representation

The final output is a density matrix representing:

- Probability distributions of true conformational states
- Overlap between states (important for understanding transitions)
- Statistical measures (entropy, uncertainty)

## Stability Improvements

### 1. Positive Variance Constraints

**Problem**: Standard EM can produce negative or near-zero variances, leading to singular covariance matrices.

**Solution**: Apply a softplus transformation to enforce positive variance:

```python
# Enforce positive variance with softplus-like transformation
diag_values = jnp.maximum(diag_values, min_variance)
diag_values = jax.nn.softplus(diag_values - min_variance) + min_variance
```

**Benefits**:

- Guarantees `variance >= min_variance` (default: 1e-3)
- Smooth, differentiable transformation
- Prevents numerical instability in likelihood calculations

**Mathematical Justification**:
The softplus function `softplus(x) = log(1 + exp(x))` ensures:

- Always positive output
- Smooth gradient for optimization
- Asymptotically linear for large positive x

### 2. Weight Constraints (Simplex Projection)

**Problem**: Component weights can drift from summing to 1 due to numerical errors.

**Solution**: Explicit normalization after each M-step:

```python
# Normalize weights to sum to 1 (enforce weight constraint)
weights = nk / jnp.sum(nk)
weights = weights / jnp.sum(weights)  # Extra normalization for numerical stability
```

**Benefits**:

- Maintains valid probability distribution
- Prevents weight drift over iterations
- Double normalization provides extra numerical stability

### 3. Component Pruning

**Problem**: GMM can develop degenerate components with extreme weights (too small or too large).

**Solution**: Automatic pruning of components outside acceptable weight range:

```python
def prune_components(
    gmm: GMM,
    min_weight: float = 1e-3,
    max_weight: float = 0.99,
) -> tuple[GMM, Array]:
    """Remove components with very small or very large weights."""
    valid_mask = (gmm.weights >= min_weight) & (gmm.weights <= max_weight)
    # ... prune and renormalize ...
```

**When to Prune**:

- **Small weights** (< 1e-3): Likely noise or degenerate components
- **Large weights** (> 0.99): One component dominates, indicating potential overfitting

**Benefits**:

- Prevents degenerate solutions
- Improves DBSCAN clustering quality
- Reduces computational cost

### 4. BIC for Model Selection

**Problem**: Difficult to choose optimal number of GMM components.

**Solution**: Compute Bayesian Information Criterion (BIC) to balance fit quality and complexity:

```python
def compute_bic(
    log_likelihood: float | Array,
    n_samples: int,
    n_components: int,
    n_features: int,
    covariance_type: Literal["full", "diag"] = "full",
) -> Array:
    """Compute BIC = -2 * log_likelihood + n_params * log(n_samples)"""
```

**Interpretation**:

- **Lower BIC = Better model**
- Balances model fit (log-likelihood) against complexity (number of parameters)
- Helps prevent overfitting

**Parameter Counting**:

- Means: `n_components * n_features`
- Full covariance: `n_components * n_features * (n_features + 1) / 2`
- Diagonal covariance: `n_components * n_features`
- Weights: `n_components - 1` (one determined by sum-to-one constraint)

## Usage Examples

### Basic GMM Fitting with Stability Features

```python
from prxteinmpnn.ensemble.gmm import make_fit_gmm_in_memory
import jax.random as random

# Create GMM fitter
fit_gmm = make_fit_gmm_in_memory(
    n_components=10,
    covariance_type="diag",
    kmeans_max_iters=100,
    gmm_max_iters=100,
    covariance_regularization=1e-6,
)

# Fit to data
key = random.PRNGKey(42)
result = fit_gmm(data, key)

# Result includes:
# - result.gmm: Fitted GMM (after automatic pruning)
# - result.log_likelihood: Final log-likelihood
# - result.converged: Whether EM converged
# BIC is automatically logged during fitting
```

### Checking Stability

```python
from prxteinmpnn.ensemble.gmm import compute_bic, prune_components

# Compute BIC
bic = compute_bic(
    log_likelihood=result.log_likelihood,
    n_samples=data.shape[0],
    n_components=result.gmm.n_components,
    n_features=result.gmm.n_features,
    covariance_type="diag",
)

# Manual pruning (already done automatically in fit)
pruned_gmm, n_removed = prune_components(
    result.gmm,
    min_weight=1e-3,
    max_weight=0.99,
)

print(f"BIC: {float(bic):.2f}")
print(f"Removed {int(n_removed)} components")
print(f"Final components: {pruned_gmm.n_components}")
```

### Full Pipeline: GMM + DBSCAN

```python
from prxteinmpnn.ensemble.dbscan import dbscan_cluster
from prxteinmpnn.ensemble.gmm import make_fit_gmm_in_memory, compute_component_distances

# 1. Fit GMM
fit_gmm = make_fit_gmm_in_memory(n_components=20, covariance_type="diag")
gmm_result = fit_gmm(protein_logits, key)

# 2. Compute distances between GMM components
distance_matrix = compute_component_distances(
    gmm_result.gmm.means,
    distance_metric="euclidean",
)

# 3. Apply DBSCAN to coarse-grain states
dbscan_result = dbscan_cluster(
    distance_matrix=distance_matrix,
    component_weights=gmm_result.gmm.weights,
    responsibility_matrix=gmm_result.gmm.responsibilities,
    eps=0.5,
    min_cluster_weight=0.01,
    connectivity_method="expm",
)

print(f"GMM components: {gmm_result.gmm.n_components}")
print(f"DBSCAN states: {len(dbscan_result.state_probabilities)}")
print(f"Plug-in entropy: {float(dbscan_result.plug_in_entropy):.3f}")
print(f"Von Neumann entropy: {float(dbscan_result.von_neumann_entropy):.3f}")
```

## Monitoring and Debugging

### Logging

The GMM fitter provides detailed logging:

```log
INFO: Initializing centroids with K-Means++...
INFO: Number of clusters: 10
INFO: Fitting GMM using in-memory EM...
INFO: EM iteration 1/100: log-likelihood = -1234.56, diff = inf, min_weight = 0.05, max_weight = 0.25
...
INFO: GMM fitting finished in 45 iterations. Converged: True, BIC: 2589.12
WARNING: Pruned 2 components with extreme weights. Remaining: 8
```

### Key Metrics to Monitor

1. **Log-likelihood**: Should increase monotonically
2. **Weight distribution**: Check `min_weight` and `max_weight` during EM
3. **BIC**: Compare across different numbers of components
4. **Components pruned**: If many components are pruned, consider adjusting initialization
5. **Convergence**: Number of iterations and convergence status

### Common Issues and Solutions

#### Issue: Too many components pruned

**Symptom**: Most GMM components have very small weights

**Solutions**:

- Reduce `n_components`
- Increase K-means iterations for better initialization
- Check data quality and preprocessing

#### Issue: Single component dominates (weight > 0.99)

**Symptom**: Warning about pruning large-weight components

**Solutions**:

- Data may not have multiple conformational states
- Increase `n_components` or adjust initialization
- Check if data is properly centered/normalized

#### Issue: Poor DBSCAN clustering

**Symptom**: Most points classified as noise

**Solutions**:

- Adjust DBSCAN `eps` parameter
- Lower `min_cluster_weight` threshold
- Ensure GMM has enough components to capture structure

## Mathematical Background

### Softplus Variance Transformation

Given raw variance estimate σ² from EM:

```python
σ²_constrained = softplus(σ² - min_var) + min_var
                = log(1 + exp(σ² - min_var)) + min_var
```

Properties:

- Always ≥ min_var
- Smooth, differentiable
- Preserves large positive values asymptotically

### BIC Formula

```latex
BIC = -2 * log L(θ | X) + k * log(n)
```

Where:

- L(θ | X): Likelihood of data X given parameters θ
- k: Number of free parameters
- n: Number of samples

Lower BIC indicates better model, balancing fit quality and complexity.

### Coarse-Graining Matrix

DBSCAN produces a coarse-graining operator M that maps GMM components to states:

```latex
P(state_j | frame_i) = Σ_k P(component_k | frame_i) * M_kj
```

The density matrix is:

```latex
ρ_jl = (1/N) Σ_i √P(state_j | i) * √P(state_l | i)
```

This captures overlap between states, crucial for understanding transitions.

## References

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society*, 39(1), 1-38.

2. Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.

3. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Kdd*, 96(34), 226-231.

4. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms*, 1027-1035.
