# ast-grep Command Library for aminx

ast-grep 0.42.1 is installed at `~/.cargo/bin/ast-grep` (also aliased as `sg`).

## Refactor Reconnaissance

### Find all Optional[Array] parameters (primary migration targets)
```bash
ast-grep run --pattern '$A: $B | None = None' --lang python src/aminx/ --json \
  | python3 -c "
import json, sys
from collections import Counter
data = json.load(sys.stdin)
files = Counter(m['file'] for m in data)
for f, c in files.most_common(20):
    print(f'{c:3d}  {f}')
print(f'Total: {len(data)} Optional params')
"
```

### Find MultistateStackPayload call sites (migration tracking)
```bash
ast-grep run --pattern 'MultistateStackPayload($$$)' --lang python src/aminx/ --json \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data:
    print(f\"{m['file']}:{m['range']['start']['line']+1}\")
"
```

### Find WaveParallelPayload spread (after PR-1)
```bash
ast-grep run --pattern 'wave_group_ids_local' --lang python src/aminx/ --json \
  | python3 -c "
import json, sys
from collections import Counter
data = json.load(sys.stdin)
files = Counter(m['file'] for m in data)
for f, c in files.most_common():
    print(f'{c:3d}  {f}')
"
```

### Find all **kwargs escape hatches
```bash
ast-grep run --pattern 'def $NAME($$$, **$KWARGS):
  $$$' --lang python src/aminx/ --json \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data:
    print(f\"{m['file']}:{m['range']['start']['line']+1}  {m['text'].split(chr(10))[0][:80].strip()}\")
"
```

### Find all jax.lax.switch sites
```bash
ast-grep run --pattern 'jax.lax.switch($$$)' --lang python src/aminx/
```

### Find ModelInputs constructor calls (post-PR-1, migration tracking)
```bash
ast-grep run --pattern 'SamplingInputs($$$)' --lang python src/
ast-grep run --pattern 'ScoringInputs($$$)' --lang python src/
ast-grep run --pattern 'ModelStaticConfig($$$)' --lang python src/
```

---

## Signature Migration Rewrites

### Replace wave_group_ids_local positional args with WaveParallelPayload
Preview (dry-run on one file):
```bash
ast-grep run \
  --pattern 'run_sample_autoregressive_state_vmap_exact($$$, wave_group_ids_local, wave_group_positions_local, wave_group_valid_local, wave_position_valid_local)' \
  --rewrite 'run_sample_autoregressive_state_vmap_exact($$$, wave_payload)' \
  --lang python \
  src/aminx/sampling/sample.py
```
Note: Use `--interactive` for staged application; verify diff before committing.

### Find functions that still accept old loose wave args (post-PR-2 audit)
```bash
ast-grep run --pattern 'def $NAME($$$wave_group_ids_local$$$):
  $$$' --lang python src/aminx/
```

---

## Guardrails / Regressions

### Detect bare Optional[Array] on pytree-registered classes (anti-pattern)
After PR-1, ModelInputs fields must not have `| None`. Run to catch drift:
```bash
ast-grep run --pattern '$FIELD: jax.Array | None' --lang python src/aminx/model_inputs.py
ast-grep run --pattern '$FIELD: jnp.ndarray | None' --lang python src/aminx/model_inputs.py
```
Both should return zero results.

### Detect remaining **kwargs on JIT-boundary functions (post-PR-3)
```bash
ast-grep run --pattern '@partial(jax.jit, $$$)
def $NAME($$$, **$KWARGS):
  $$$' --lang python src/aminx/
```

### Find any new bool positional args added since FBT sweep
```bash
ast-grep run --pattern 'def $NAME($$$, $P: bool$$$):
  $$$' --lang python src/aminx/
```

---

## Structural Audit

### Count functions by approximate parameter density
Uses Python AST (faster than ast-grep for param counting):
```bash
python3 -c "
import ast, pathlib
results = []
for f in pathlib.Path('src/aminx').rglob('*.py'):
    try:
        tree = ast.parse(f.read_text())
    except:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            total = len(args.args) + len(args.kwonlyargs)
            if total >= 10:
                results.append((total, str(f), node.name, node.lineno))
results.sort(reverse=True)
for count, path, name, lineno in results[:20]:
    print(f'{count:3d}  {path}:{lineno}  {name}')
"
```

---

## sgconfig.yml for CI enforcement (post-migration)

Save as `sgconfig.yml` in repo root after PR-4 completes:

```yaml
rules:
  - id: no-optional-array-on-model-inputs
    message: "ModelInputs fields must not be Optional — resolve on host before JIT"
    severity: error
    language: python
    rule:
      pattern: "$FIELD: $T | None = None"
      inside:
        kind: class_definition
        has:
          pattern: "ModelInputs"

  - id: no-kwargs-at-jit-boundary
    message: "JIT-compiled functions must not use **kwargs — use ModelInputs instead"
    severity: warning
    language: python
    rule:
      pattern: "def $NAME($$$, **$KWARGS):"
      inside:
        kind: decorated_definition
        has:
          pattern: "jax.jit"
```
