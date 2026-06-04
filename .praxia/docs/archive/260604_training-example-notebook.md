---
archive: 260604_training-example-notebook.tar.zst
created: 260604
source: examples/training_example_notebook.ipynb
size_bytes: 23674
contents:
  - path: training_example_notebook.ipynb
    size_bytes: 23674
---

# Archived: training_example_notebook.ipynb

**Verdict:** Archived due to feature not ready for use.

## Summary

The `prxteinmpnn.training` module intentionally raises `NotImplementedError` for all attribute access, with a docstring warning that the module "has not been updated to use the Sprint 2 composable inference architecture."

Although the underlying submodules exist (`specs.py`, `trainer.py`), the public package gateway explicitly gates access with:

```python
def __getattr__(name):
    raise NotImplementedError("Training module has not been updated...")
```

This means any notebook attempting to import from `prxteinmpnn.training` cannot work in the current release, regardless of whether the internal code exists.

## Dependencies

The training feature is tracked as Sprint 3 tech debt and blocked by the TrainingPlan work. The module cannot be made functional until:
1. Training code is refactored to use composable-inference API
2. `TrainingSpecification` is updated to match new architecture
3. Public API is unblocked

## Recommendation

Archive now. Re-introduce when training feature lands in Sprint 3 (currently blocked by Sprint 3 TrainingPlan).
