"""Measure how closely aminx and the reference agree on the CONDITIONAL NLL, in nats.

WHY THIS EXISTS. ``tests/parity/test_full_model_parity.py`` gates every post-featurization
stage on ``pearson_correlation >= 0.95``. Correlation is affine-invariant, so it cannot see a
scale or offset error at all: perturbing logits by ``* 2.0`` correlates at r = 0.9740 -- a
comfortable pass -- while shifting the reported NLL by +0.2477 nats. The aggregate scalar
that downstream claims are actually denominated in is asserted NOWHERE.

Before pinning a tolerance in a test, measure what agreement the implementations actually
achieve. A tolerance guessed rather than derived either fails on arrival or silently admits
more error than the effects it is supposed to protect (the reference effect size in the
consuming project is ~0.016 nats).

Reports, for the conditional estimand p(s_i | s_{-i}, X) -- the one ``runner.score`` now
computes by default:

  - Pearson r, for comparison against the existing gate
  - max |delta| on log-probs, which is what correlation is blind to
  - the reference's own ``get_score`` (``data_utils.py``) applied to BOTH sides' log-probs

Run with ``REFERENCE_PATH`` pointing at the LigandMPNN checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from aminx.inference import score_conditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from tests.parity.test_full_model_parity import (
  _build_parity_batch,
  _build_torch_feature_dict,
  _jax_protein_for_source,
  _load_heavy_parity_models_impl,
  _pearson_correlation,
)

models = _load_heavy_parity_models_impl()
batch = _build_parity_batch()

feature_dict = _build_torch_feature_dict(models.torch, batch)
with models.torch.no_grad():
  pt_score = models.pt_model.score(feature_dict, use_sequence=True)
pt_log_probs = pt_score["log_probs"].numpy()[0]

jax_model = _jax_protein_for_source(models, "bundled")
bundle, config = build_inference_bundle(
  coords=jnp.array(batch.x_pytorch[0])[None, ...],
  mask=jnp.array(batch.mask[0])[None, ...],
  residue_index=jnp.array(batch.residue_index[0])[None, ...],
  chain_index=jnp.array(batch.chain_index[0])[None, ...],
  sequence=jax.nn.one_hot(jnp.array(batch.sequence[0]), 21),
  ar_mask=jnp.array(batch.ar_mask)[None, ...],
  mode="score_conditional",
)
jax_logits = score_conditional.kernel(
  jax_model,
  jax.random.PRNGKey(0),
  bundle,
  config,
  make_stage_set(),
)
jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

# The reference aggregator itself. `import data_utils` would pull in `prody`, which is not a
# dependency here, so lift the function out of the reference SOURCE by AST instead of
# transcribing it. Transcription drift is precisely the failure a parity test exists to
# catch, so the reference formula must never be retyped into this repo -- extracted this way
# it still tracks any upstream change to `get_score`.
import ast
import os

import torch


def _load_reference_get_score() -> Any:  # noqa: ANN401
  """Extract ``get_score`` from the reference ``data_utils.py`` without importing it."""
  source_path = Path(os.environ["REFERENCE_PATH"]) / "data_utils.py"
  tree = ast.parse(source_path.read_text())
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "get_score":
      namespace: dict[str, Any] = {"torch": torch}
      exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)  # noqa: S102
      return namespace["get_score"]
  msg = f"get_score not found in {source_path}"
  raise RuntimeError(msg)


get_score = _load_reference_get_score()

sequence = np.asarray(batch.sequence[0])
mask = np.asarray(batch.mask[0])


def reference_nll(log_probs: np.ndarray) -> float:
  """Apply the reference's own ``get_score`` to either side's log-probs."""
  average, _ = get_score(
    torch.from_numpy(sequence).long()[None, ...],
    torch.from_numpy(np.asarray(log_probs, dtype=np.float32))[None, ...],
    torch.from_numpy(mask.astype(np.float32))[None, ...],
  )
  return float(average.item())


pt_nll = reference_nll(pt_log_probs)
jax_nll = reference_nll(jax_log_probs)

print(f"L                        : {sequence.shape[0]}")
print(f"pearson r (existing gate): {_pearson_correlation(pt_log_probs, jax_log_probs):.6f}")
print(f"max  |delta| log-probs   : {np.abs(pt_log_probs - jax_log_probs).max():.3e}")
print(f"mean |delta| log-probs   : {np.abs(pt_log_probs - jax_log_probs).mean():.3e}")
print(f"reference NLL (torch)    : {pt_nll:.8f}")
print(f"reference NLL (aminx)    : {jax_nll:.8f}")
print(f"|delta| NLL, nats        : {abs(pt_nll - jax_nll):.3e}")
print("  project effect size    : 1.6e-02  (protein-vs-soluble d = -0.0160)")
