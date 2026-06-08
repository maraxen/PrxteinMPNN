---
archive: 260604_example-notebook.tar.zst
created: 260604
source: examples/example_notebook.ipynb
size_bytes: 47700
contents:
  - path: example_notebook.ipynb
    size_bytes: 47700
---

# Archived: example_notebook.ipynb

**Verdict:** Archived due to API drift and deprecated symbols.

## Summary

This notebook references multiple symbols that no longer exist in the current package:
- `get_mpnn_model`, `ModelWeights`, `ProteinMPNNModelVersion` from `aminx.mpnn` (not found anywhere)
- `from_structure_file`, `protein_structure_to_model_inputs`, `string_to_protein_sequence` from `aminx.io` (exports not available)
- `SamplingConfig` from `aminx.sampling` (does not exist)

Additionally, function signatures have changed:
- `make_sample_sequences` no longer accepts `config` or `model_inputs` kwargs
- `make_score_sequence` now requires 6+ positional arguments instead of 2

This notebook predates the Sprint 2 composable-inference refactor and would require substantial rewriting to use current APIs.

## Recommendation

Archive now. Re-introduce as a fresh example demonstrating composable-inference API in a later sprint (Tech Debt: Sprint 3).
