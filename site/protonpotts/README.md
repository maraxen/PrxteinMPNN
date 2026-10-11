# ProtonPottsMPNN in the browser

A static, bundler-free page that runs three validated ProtonPottsMPNN methods client-side through `onnxruntime-web`,
with the arithmetic done by the browser modules in `browser/protonpotts-scorer/`:

- **Score**: energies of the native sequence and any variant rows (one structure, one Potts energy per row).
- **pH design**: protonation-state design over `PhDesigner`, returning designs sorted by final energy. Five methods, chosen
  in the page's Method selector (see below).
- **Sample**: one sequence from the decoder (`ProtonPottsSampler`), with its per-position log-probabilities.

Other ProtonPotts paths (for example the `autoregressive` pH method, which needs the decoder field, and non-selective
planning) are not exposed here.

## pH design methods and validation status

| Method | Browser inputs | Validation status |
| :--- | :--- | :--- |
| `block_descent` (default) | binder chain, temperature, samples per site, max mutations, seed | Validated in Chromium and as deployed (X7c, X10). Its uniforms are browser-drawn unless supplied, so they are not the validated stream. |
| `converged_mcmc` | + cvPatience, cvMax | Validated under Node against aminx on replayed draws (X11b, in progress). Not validated in the browser. |
| `converged_mcmc_combined` | + cvPatience, cvMax, combinedLambda | Validated under Node against aminx on replayed draws (X11b, in progress). Not validated in the browser. |
| `two_phase` | + cvPatience, cvMax, twoPhaseFrac | Validated under Node against aminx on replayed draws (X11b, in progress). Not validated in the browser. |
| `gibbs` | none beyond the common inputs | Validated under Node against aminx on replayed draws (X11b, in progress). Not validated in the browser. |

Be clear about what this means. The four newer methods have been checked against aminx in Node on replayed draws, and
that check is still in progress (X11b). None of them has been run in Chromium or as deployed. Block descent is the only
method whose browser output has been checked against the validated stream.

The Seed input is used by every method. Browser-drawn runs are reproducible for a given seed, but **they are not the
validated stream**, for any method (see "Uniforms and seeds").

## Files

- `index.html`, `site.css`: the page and its styles (light and dark via `prefers-color-scheme`).
- `app.mjs`: DOM and worker owner. `window.__PP_RUN__(msg)` posts a job and resolves with the worker's reply. The
  buttons call the same function.
- `worker.mjs`: module Web Worker. Loads and verifies models, caches sessions, runs the three methods.
- `config.js`: `MODEL_BASE`, the one line a deployment rewrites to point at a different model location.

## Models layout

`MODEL_BASE` (default `./models`) must contain three manifests and the graphs they name:

```
models/
  scoring/MANIFEST.json   table + energy graphs (score, and the scorer used by pH design)
  ph/MANIFEST.json        pH graphs (pH design)
  decoder/MANIFEST.json   encode + decode graphs (sample)
```

Each manifest has a `buckets` list (256 and 1024). Each bucket's `graphs` entries give a `file` relative to that
manifest's directory and a `sha256`. The worker fetches every `.onnx`, checks its sha256 against the manifest with
`crypto.subtle`, and refuses to load it on a missing digest or a mismatch. The graph files come from the tracked export
gates: `scripts/browser_validation/protonpotts_export_gate.py` (scoring), `protonpotts_ph_export_gate.py` (pH) and
`protonpotts_decode_export_gate.py` (decoder). The manifests themselves are not hashed by the page.

The model files are not in the repository.

## Redistribution

Weights are **not bundled** by default. ProteinMPNN's checkpoint license must be confirmed to cover public
redistribution before weights are published with a site. Shipping them is a deliberate, opt-in act, as in
`.github/workflows/pages.yml`, which builds with no weights unless its manual weights step is run. A deployment that
does not bundle them points `MODEL_BASE` at a release location instead.

## Threads on GitHub Pages

Multithreaded wasm needs cross-origin isolation (`COOP: same-origin` and `COEP: require-corp`). GitHub Pages cannot set
response headers, so a plain Pages deployment runs single-threaded (`threads: 1`). The page shows the thread count and
`crossOriginIsolated` in the status area. To get threads on Pages, vendor `coi-serviceworker.min.js` next to
`index.html` and register it as the first script in `<head>` (the commented-out hook is in `index.html`). Without it,
the page works, just slower.

Session creation is the expensive step, so the worker caches sessions per (method, model base, bucket, thread count).

## Input formats

- **Structure**: a PDB file, uploaded or pasted. Records follow the same reader as the host (`protonpotts_inputs.mjs`):
  a blank chain id, an MSE residue, or a non-standard residue is refused.
- **Protonation labels** (optional, scoring and pH design): comma-separated, one entry per loaded residue, in file
  order. An empty entry means no label.
- **Score variants**: one per line, `name: tokens` or just `tokens`. Tokens are separated by commas or spaces, or,
  with no separators, read one character per residue. Each token is a one-letter code or a v6 protonation token.
  Each row must have one token per kept residue. Row 0 is always the native sequence.
- **pH design**: method, binder chain, temperature, samples per site, max mutations, seed, and the method's own knobs
  (cvPatience and cvMax for the converged and two-phase methods, combinedLambda for the combined method, twoPhaseFrac
  for two_phase). Defaults mirror `DEFAULT_CONFIG` in `ph_design.mjs`. Only the knobs the chosen method reads are sent.
- **Sample**: seed and temperature (default 0.1).

## Uniforms and seeds

The validated pH design replays uniforms that the host generated and passed in. The page does not do that. For
`block_descent`, when no uniforms are supplied, the worker draws them from a seeded generator (`seededStreams`). For the
other four methods the worker passes no uniforms and no replay, so the designer draws from a stream seeded by the Seed
input (`config.seed`). In both cases runs are reproducible in the browser but are **not** the validated stream. The same
applies to sampling without explicit uniforms.

## Building for deployment

The modules under `browser/protonpotts-scorer/` are imported from `worker.mjs` through specifiers the build step
rewrites into a local `lib/` directory, so the built page is self-contained apart from the models and the CDN copy of
onnxruntime-web (pinned at 1.30.0).

## Output

Each result can be downloaded as JSON. Score results hold `names`, `energies` and `l_total`. Design results hold
`designs`, each with `sequence`, `symbols`, `pins`, `final_potts_energy`, `selective_energy` and `n_draws`. Sample
results hold `sequence`, `symbols`, `log_probs` and `decoding_order`.
