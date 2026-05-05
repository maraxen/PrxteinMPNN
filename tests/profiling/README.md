# Profiling tests (Phase 0)

## Q8 allowlist (`hlo_allowlist.toml`)

Open question **Q8** (roadmap §13) caps exported HLO **byte size** per parity-pinned callable key. `tests/profiling/test_hlo_baseline.py::test_export_hlo_model_call_under_allowlist` lowers a tiny unconditional `PrxteinMPNN` and asserts the UTF-8 length of the exported IR is at most `max_hlo_bytes` for `model_call`. The same file is the right place to add parallel export checks for other keys when needed.

Rationale strings live in the TOML next to each ceiling; bump a ceiling only with a short justification in PR text.

## Review-only baselines (`baseline_hlo/*.txt`)

Files under `baseline_hlo/` are **human review artifacts**, not CI equality baselines.

- **CI** (`test_baseline_hlo_review_artifacts_exist`): each of `model_call`, `score`, `sample`, `logits` must have a non-empty file on disk. There is **no** byte-for-byte or substring gate on the text in CI.
- **Smoke** (`test_export_hlo_model_call_under_allowlist`, `test_assert_zero_copy_overhead_self_check`): export and memory wiring for the tiny model path.

If you intentionally change JIT-relevant code, re-capture the text you care about, refresh the files, and mention **baseline refresh** in the PR so reviewers expect a large diff.

### Allowlist keys (intent)

| File | Intent |
|:-----|:-------|
| `model_call.txt` | Parity-pinned **`model.__call__`** unconditional logits slice: same `tiny_model` / `n=4` setup as `test_export_hlo_model_call_under_allowlist`. |
| `logits.txt` | Parity-pinned **`make_unconditional_logits_fn(model)(...)`** on the same tiny structure (minimal unconditional logits factory). |
| `score.txt` | Parity-pinned **`make_score_fn(model)(...)`** conditional score on the same backbone with a zero integer sequence (scalar score subtree exported). |
| `sample.txt` | Parity-pinned **`make_sample_sequences(model, sampling_strategy="temperature")`** first return (sampled sequence) on the same tiny backbone. |

Full production factories can be larger; these baselines are **minimal** lowers that map 1:1 to the four roadmap callables for review and local diffing.

## Regeneration (no `REFERENCE_PATH`)

From the repo root, after changing JAX/Equinox or the tiny-model definition, you can refresh all four review files with a heredoc (no `REFERENCE_PATH`):

```bash
uv run python - <<'PY'
from datetime import date
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.profiling.hlo_tools import export_hlo
from prxteinmpnn.scoring.score import make_score_fn
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn

capture = date.today().isoformat()
header = (
    "# StableHLO / HLO text (review artifact only; CI does not diff this file).\n"
    f"# Capture date: {capture}\n"
    f"# JAX {jax.__version__}, jaxlib {jaxlib.__version__}, Equinox {eqx.__version__}\n"
    "# Same tiny unconditional path as tests/profiling/test_hlo_baseline.py (n=4).\n"
    "# Regeneration: see tests/profiling/README.md\n\n"
)
key = jax.random.PRNGKey(0)
tiny = PrxteinMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    key=key,
)
tiny = eqx.tree_inference(tiny, value=True)
n = 4
coords = jnp.zeros((n, 4, 3), jnp.float32)
mask = jnp.ones((n,), jnp.float32)
ri = jnp.arange(n, dtype=jnp.int32)
ci = jnp.zeros((n,), jnp.int32)
pk = jax.random.PRNGKey(1)


def model_call(pk):
    return tiny(coords, mask, ri, ci, "unconditional", prng_key=pk)[1]


logits_fn = make_unconditional_logits_fn(tiny)


def logits_path(pk):
    return logits_fn(pk, coords, mask, ri, ci, None, None)


score_fn = make_score_fn(tiny)
seq = jnp.zeros((n,), jnp.int32)


def score_path(pk):
    return score_fn(pk, seq, coords, mask, ri, ci)[0]


sample_fn = make_sample_sequences(tiny)


def sample_path(pk):
    return sample_fn(pk, coords, mask, ri, ci)[0]


out = Path("tests/profiling/baseline_hlo")
extra = {
    "model_call.txt": "# Callable key: model_call — unconditional logits slice.\n",
    "logits.txt": "# Callable key: logits — make_unconditional_logits_fn.\n",
    "score.txt": "# Callable key: score — make_score_fn conditional.\n",
    "sample.txt": "# Callable key: sample — make_sample_sequences temperature.\n",
}
for fname, fn in [
    ("model_call.txt", model_call),
    ("logits.txt", logits_path),
    ("score.txt", score_path),
    ("sample.txt", sample_path),
]:
    body = export_hlo(fn, pk)
    (out / fname).write_text(header + extra[fname] + "\n" + body, encoding="utf-8")
    print("wrote", fname, len(body))
PY
```

Run profiling tests locally:

```bash
uv run pytest tests/profiling -q
```

Quick check that review files exist (no `REFERENCE_PATH`, no JAX lower):

```bash
uv run python -c "from pathlib import Path; p=Path('tests/profiling/baseline_hlo'); print({f.name: f.stat().st_size for f in sorted(p.glob('*.txt'))})"
```

## `REFERENCE_PATH`

Heavy parity and reference-backed tests use `REFERENCE_PATH` elsewhere (`tests/parity`, etc.). The profiling README snippets above do **not** require it.
