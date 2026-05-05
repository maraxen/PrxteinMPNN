"""One-shot: copy pytest-json-report raw output and add resolution fields per failure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

RAW = Path(__file__).resolve().parent / "pytest_errors" / "260505_1ead0b39_pytest_raw.json"
OUT = Path(__file__).resolve().parent / "pytest_errors" / "260505_1ead0b39_pytest.json"
COVERAGE = Path(__file__).resolve().parent / "pytest_errors" / "260505_1ead0b39_coverage.json"

# fmt: off
ENRICH: dict[str, dict[str, str]] = {
  "tests/io/test_weights.py::test_smart_factory_model_loading[ligandmpnn_v_32_010_25]": {
    "resolution_strategy": (
      "Regenerate the packaged `ligandmpnn_v_32_010_25.eqx.zst` from the reference `.pt` so the "
      "Equinox tree matches the current `PrxteinMPNN` module, or narrow the parametrized test to "
      "checkpoint IDs that are known-good for this code revision."
    ),
    "diff": "Run `scripts/regenerate_packaged_eqx_zst.py`; verify `src/prxteinmpnn/io/weights.py` "
    "registry maps the checkpoint id to the refreshed asset; adjust `tests/io/test_weights.py` if the "
    "matrix of ids is stale.",
    "reasoning": (
      "`TreePathError` under `y_context_encoder` / `dropout` indicates deserializing a checkpoint "
      "whose leaves do not align with the live model PyTree (common after architecture or layer renames)."
    ),
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_with_structure_mapping[True]": {
    "resolution_strategy": (
      "Thread the same multistate arguments the public `__call__` expects through the Jacobian / "
      "conditional-logits construction path (or call a small adapter that prepares `state_weights`, "
      "`state_mapping`, fixed masks/tables, etc.)."
    ),
    "diff": "`src/prxteinmpnn/run/jacobian.py`, `src/prxteinmpnn/sampling/conditional_logits.py`, "
    "and/or `tests/run/test_jacobian_multistate.py` fixtures that build `conditional_logits_fn`.",
    "reasoning": (
      "`PrxteinMPNN._call_conditional` now requires multistate/grouping arguments; tests still invoke "
      "the lower-level path without those parameters."
    ),
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_with_structure_mapping[False]": {
    "resolution_strategy": (
      "Same as JIT=True variant: ensure `make_conditional_logits_fn` binds all `_call_conditional` "
      "positional arguments regardless of `with_jit`."
    ),
    "diff": "`src/prxteinmpnn/run/jacobian.py`; `src/prxteinmpnn/sampling/conditional_logits.py`.",
    "reasoning": "Identical `TypeError` signature mismatch; only the JIT wrapper differs.",
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_without_structure_mapping": {
    "resolution_strategy": (
      "Define explicit single-state defaults (identity mapping, neutral weights, empty fixed positions) "
      "and pass them into `_call_conditional` so 'no structure mapping' is represented as data, not "
      "as omitted arguments."
    ),
    "diff": "`conditional_logits` factory and/or `run/jacobian.py` single-structure branch.",
    "reasoning": "Backward compatibility must be encoded as default tables/masks, not fewer call arguments.",
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_multiple_structures": {
    "resolution_strategy": (
      "After fixing the factory signature, revalidate that batched structure indices still produce "
      "bounded logits; add assertions if the API now requires precomputed group tables."
    ),
    "diff": "Multistate preparation utilities used by this test + conditional logits factory.",
    "reasoning": "Fails for the same missing-parameters error before multistate logic is evaluated.",
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_gradient_computation": {
    "resolution_strategy": (
      "Unblock the forward conditional call first; then confirm `jax.jacrev`/`jacfwd` sees a pure "
      "function with closed-over multistate arrays."
    ),
    "diff": "Same multistate plumbing as other jacobian tests.",
    "reasoning": "Gradients never run because the primal call does not type-check.",
  },
  "tests/run/test_jacobian_multistate.py::test_categorical_jacobian_jit_compatible": {
    "resolution_strategy": (
      "Ensure the kwargs/args passed into the jitted closure match the updated `_call_conditional` "
      "signature; avoid Python-side optional branches that skip required arrays under JIT."
    ),
    "diff": "`test_jacobian_multistate.py` + `run/jacobian.py` JIT entrypoints.",
    "reasoning": "Same root cause: stale partial application around the MPNN call.",
  },
  "tests/run/test_prep.py::test_prep_resolves_registry_checkpoint_for_ligand": {
    "resolution_strategy": (
      "Restore or replace `prep.load_ligand_model` (e.g. unify on `load_model` / `smart_factory`) "
      "and update tests to patch the actual symbol."
    ),
    "diff": "`src/prxteinmpnn/run/prep.py`; `tests/run/test_prep.py`.",
    "reasoning": "AttributeError shows the registry test still references a removed helper.",
  },
  "tests/run/test_prep.py::test_prep_registry_checksum_mismatch_raises": {
    "resolution_strategy": (
      "Re-enable strict checksum validation: when registry SHA256 disagrees with artifact bytes, "
      "raise `ValueError` before loading."
    ),
    "diff": "`prep.py` registry resolution path; align exception type with test expectations.",
    "reasoning": "Test expected `ValueError`; implementation now swallows mismatch or uses another path.",
  },
  "tests/run/test_prep.py::test_prep_registry_missing_entry_raises": {
    "resolution_strategy": (
      "If the checkpoint id is absent from the registry JSON, fail fast with `ValueError` carrying "
      "the missing id (restore prior contract)."
    ),
    "diff": "`prep.py` lookup + `tests/run/test_prep.py` if the registry schema moved.",
    "reasoning": "`pytest.fail DID NOT RAISE` means the negative branch no longer triggers.",
  },
  "tests/sampling/test_conditional_logits.py::test_conditional_logits_helper_matches_model_branch": {
    "resolution_strategy": (
      "Update the parity helper to invoke the same argument bundle as production (`state_weights`, "
      "`state_mapping`, fixed masks, group tables), or add a thin wrapper on `PrxteinMPNN` for tests."
    ),
    "diff": "`src/prxteinmpnn/sampling/conditional_logits.py`; possibly test-local helpers.",
    "reasoning": "Direct `_call_conditional` use in tests lags the production `__call__` contract.",
  },
  "tests/sampling/test_conditional_logits.py::test_split_conditional_logits_matches_full_helper": {
    "resolution_strategy": (
      "Keep split/full encode-decode helpers in lockstep: both must pass the multistate kwargs "
      "that `_call_conditional` now requires."
    ),
    "diff": "`conditional_logits.py` split vs full implementations.",
    "reasoning": "Same signature drift as the adjacent parity test.",
  },
  "tests/training/test_diffusion_step.py::test_diffusion_train_step": {
    "resolution_strategy": (
      "Load a diffusion checkpoint that matches `DiffusionMPNN` (regenerate packaged weights) or "
      "update the test to construct a fresh model without deserializing an obsolete tree."
    ),
    "diff": "`src/prxteinmpnn/model/diffusion_mpnn.py`; packaged `model_params`; torch→eqx regen script.",
    "reasoning": (
      "`TreePathError` / `EOFError` while loading `t_embed_sin.embedding_dim` means the file truncates "
      "or the hyperparameters changed vs the serialized module."
    ),
  },
  "tests/training/test_trainer_config.py::test_load_model_diffusion": {
    "resolution_strategy": (
      "Same as `test_diffusion_train_step`: align trainer-config fixture checkpoint with the current "
      "diffusion model class or skip until assets are refreshed."
    ),
    "diff": "`tests/training/test_trainer_config.py`; diffusion weight packaging.",
    "reasoning": "Identical deserialization failure mode against `t_embed_sin`.",
  },
}
# fmt: on


def main() -> None:
  data = json.loads(RAW.read_text(encoding="utf-8"))
  tests = data.get("tests", [])
  failed = [t["nodeid"] for t in tests if t.get("outcome") == "failed"]
  missing = [n for n in failed if n not in ENRICH]
  if missing:
    msg = "Missing ENRICH entries:\n" + "\n".join(missing)
    raise SystemExit(msg)
  extra = {n for n in ENRICH if n not in failed}
  if extra:
    raise SystemExit(f"Stale ENRICH keys (not failed): {sorted(extra)}")
  for t in tests:
    if t.get("outcome") != "failed":
      continue
    meta = ENRICH[t["nodeid"]]
    t["resolution_strategy"] = meta["resolution_strategy"]
    t["diff"] = meta["diff"]
    t["reasoning"] = meta["reasoning"]
  data["_praxia"] = {
    "annotated_at": "2026-05-05",
    "pytest_invocation": (
      'uv run --with pytest-json-report --with pytest-timeout pytest tests '
      '-m "not slow and not parity_heavy" --timeout=30 --cov=prxteinmpnn ...'
    ),
    "coverage_json_path": str(COVERAGE.name) if COVERAGE.exists() else None,
    "raw_source": RAW.name,
  }
  OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  print(f"Wrote {OUT} ({len(failed)} failures annotated)")


if __name__ == "__main__":
  main()
