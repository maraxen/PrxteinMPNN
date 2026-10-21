---
title: 'Hoisted sampler: encode once per structure, decode per key (xtrax make_hoisted)'
description: 'sample_sequences re-encodes per PRNG key under vmap (false dependence via the noise key split); add make_hoisted_sampler on xtrax make_hoisted, retire the ignored precomputed_* kwargs, encode once per structure in runner.inspect'
status: draft
adversarial_review: 'spec-challenger (sonnet) 261020: NEEDS_WORK, 5 blockers + 6 majors addressed in revision'
task_id: 261020_hoist-consumer-migrations
date: '261020'
backlog_ids: 'xtrax debt #2600 follow-up'
---
# Hoisted sampler

## Purpose

Callers that `jax.vmap` the temperature sampler over PRNG keys (tev_design
`run_design_grid.py`, `generate_tev_conditionals.py`) run the structure encoder once
per key. At `backbone_noise == 0` the encoder output does not depend on the key, but the
jaxpr does, so neither vmap nor XLA can hoist it. Add an encode-once sampler built on
`xtrax.inference.make_hoisted` (xtrax >= 0.4.1b2), and encode once per structure in
`runner.inspect`.

## What exists (V = read at file:line on origin/main f948d2b3, U = unverified)

- V `make_sample_sequences(...).sample_sequences` — `src/aminx/sampling/sample.py:141-214`.
  Key stream: `k_order, prng_key = split(prng_key)` (:172); bundle via
  `build_inference_bundle(..., backbone_noise=noise or 0.0, mode="sample_ar")` (:174);
  `with_decoding_order(bundle, k_order, custom_fn, num_groups)` when no wave schedule
  (:198-200); `sample_autoregressive.kernel(model, prng_key, bundle, config, stage_set)` (:208).
- V `sample_autoregressive.kernel` — `src/aminx/inference/sample_autoregressive.py:124-136`:
  `k_enc, k_dec = split(prng_key)`; `enc = make_encode_fn(model)(bundle, k_enc, config)`;
  `decode_fn(k_dec, enc, bundle, config, stage_set, gumbel_noise=...)`.
- V False dependence: the noise key is split unconditionally and only the value is
  scaled — `src/aminx/utils/coordinates.py:44,48`, `src/aminx/model/ligand_features.py:286,292`.
  Encoding reads `fold_in(k_enc, 0)` (`src/aminx/inference/encode.py:89,193`).
- V `encode.py` reads no `ar_mask`, `wave` or decoding order (rg), so the decoding order
  can be set after encoding. U whether encode reads `fixed_mask`/`fixed_tokens`/`bias`
  (treated as shared args either way; see A2).
- V `precomputed_node_features`, `precomputed_edge_features`, `precomputed_neighbor_indices`
  are parameters of `sample_sequences` (:163-165) and are read nowhere (`rg` over `src/`).
  tev_design passes them believing they skip the encoder (run_design_grid.py:768-772).
- V `runner.inspect` — `src/aminx/host/runner.py:1655-1812`: per structure, per inspection
  feature, rebuilds split-fn factories and re-encodes with identical inputs
  (`backbone_noise=0.0`, same `subkey`); `decoded_node_features` and
  `encoded_node_features`/`edge_features` both encode (:1775-1782, :1805-1812). No cache.
- V xtrax `make_hoisted(shared, per_element, *, batch_size, axis, cache)` →
  `HoistedMap(shared_args, xs, *, reference, check_elements, rtol, atol)`; shared stage
  runs once per call; `axis.varying_inputs` names refused in `shared_args`
  (`HoistVaryingInputError`); parity probe raises `HoistParityError`.
  Not in a release yet: needs xtrax 0.4.1b2 (release PR in flight).
- V aminx pins `xtrax[io,export]==0.4.0a11` and `xtrax[onnx]==0.4.0a11`
  (`pyproject.toml:26,89`).

## API (revised after adversarial review 261020: NEEDS_WORK, B1-B5 + majors addressed)

```python
# src/aminx/sampling/hoisted.py, re-exported from aminx.sampling
def make_hoisted_sampler(
    model, *, decoding_order_fn=_DEFAULT_DECODING_ORDER_FN, num_groups: int | None = None,
    multi_state_strategy="arithmetic_mean", multi_state_temperature: float = 1.0,
    inference_only: bool = False, batch_size: int | None = None,
) -> HoistedSampler: ...

class HoistedSampler:
    def __call__(self, keys,                      # (N,) typed keys or (N, 2) raw
                 *, structure_coordinates, mask, residue_index, chain_index,
                 bias=None, fixed_mask=None, fixed_tokens=None, temperature=1.0,
                 tie_group_map=None, state_weights=None, ligand_coords=None,
                 ligand_atom_types=None, ligand_mask=None,
                 check_elements: int = 0, rtol: float = 1e-5, atol: float = 1e-6,
                 ) -> tuple[Array, Array, Array]:
        """Same (seqs, logits, decoding_order) as vmap(sample_sequences)(keys, ...),
        leading axis N; the encoder runs once per call."""
    def stats(self) -> dict[str, int]: ...
```

**No cross-call cache (B1).** xtrax's memo purity screen refuses `random_bits`, also
inside `cond` branches (`xtrax/inference/memo.py:481-493`), and the encode graph always
contains one: `apply_noise_to_coordinates` draws `jax.random.normal` inside its cond
(`utils/coordinates.py:44-47`), and Dropout draws a Bernoulli mask
(`model/dropout.py:56`). `make_hoisted(..., cache=None)` only; the within-call hoist is
the goal.

**Refactor first (B2).** Split `sample_autoregressive.kernel`
(`inference/sample_autoregressive.py:124-136`) into two helpers in the same module,
and make `kernel` call them so the monolithic path is unchanged:
- `encode_for_sampling(model, bundle, k_enc, config) -> EncoderOutput`
  (`make_encode_fn(model, use_rolling_state=False)(bundle, k_enc, config)`);
- `decode_sampling(model, k_dec, enc, bundle, config, stage_set, *, state_strategy=None,
  inference_only=False, gumbel_noise=None)` (builds `decode_fn` as :131-136 do).

`make_hoisted(shared, per_element, batch_size=batch_size,
axis=AxisSpec(name=<aminx N_SAMPLES axis or "n_samples">, varying_inputs=("keys",)))`:

- `shared(**structure_args)`: `build_inference_bundle(..., backbone_noise=0.0,
  mode="sample_ar", inference=True)` exactly as `sample.py:174-192`;
  `stage_set = make_stage_set(strategy, strategy_temperature, state_weights)` (per call,
  since `state_weights` is a per-call argument, `sample.py:203`);
  `enc = encode_for_sampling(model, bundle, ENCODE_KEY, config)` with fixed
  `ENCODE_KEY = jax.random.key(0)`. Returns `(bundle, config, stage_set, enc)`.
  `config` is all-static and passes through jit (challenger, verified).
- `per_element((bundle, config, stage_set, enc), key)` mirrors the key stream exactly:
  `k_order, k = split(key)` (`sample.py:172`); `bundle = with_decoding_order(bundle,
  k_order, custom_fn, num_groups)` (:198-200); `decoding_order =
  decoding_order_from_wave(bundle.wave, bundle.conditioning.tie_group_map[0])` (:202);
  `_, k_dec = split(k)` (kernel :124); `result = decode_sampling(...)`; return
  `(result.sequence.astype(int8), result.logits, decoding_order)`.
- `temperature` is converted to `jnp.float32` before the call, so distinct Python floats
  do not recompile the shared stage (major: a Python float is a static jit arg under
  HoistedMap's uncached path).
- Parity probe (B3): `reference(shared_args, key) = sample_sequences(key, **shared_args)`
  wrapper built inside `HoistedSampler`; `rtol`/`atol` forwarded to `HoistedMap`
  (defaults 1e-5 / 1e-6, matching A1).

Not supported by the hoisted sampler, each with a reason in its docstring (major):
`backbone_noise` (hoist valid only at 0: a per-sample noise draw is per-element work),
`wave_schedule` (callers with a fixed schedule have no per-key order; use
`sample_sequences`), `use_rolling_state` (not used by the AR kernel, `:127`).

`sample_sequences` with any `precomputed_*` not None: `FutureWarning` ("ignored; use
make_hoisted_sampler"), behaviour unchanged. Removal is a later PR.

**`runner.inspect` (B5).** Only `decoded_node_features`, `encoded_node_features` and
`edge_features` share encoder inputs (same split fn, `backbone_noise=0.0`, same
`subkey`: `runner.py:1775-1782`, `:1805-1812`). Per structure, encode once for those and
reuse; build `make_encoding_conditional_logits_split_fn(model)` once per `inspect` call.
The conditional / batched-conditional / unconditional paths keep their own encode
(different model mode and key split: `sampling/conditional_logits.py:169-227`). Aminx-
native; no xtrax involvement.

**Pin bump.** `xtrax[io,export]==0.4.1b2`, `xtrax[onnx]==0.4.1b2` once 0.4.1b2 is on
PyPI (xtrax PR #223). Before bumping, read the xtrax CHANGELOG sections 0.4.0a12, 0.4.1b1
and 0.4.1b2 and `git diff v0.4.0a11 v0.4.1b2 -- distribution/public_api.toml
'src/xtrax/*/__init__.py'` for removed or renamed names; list any breaking change in
the PR. Until published, develop against xtrax main via an untracked `uv.toml`
`[sources]` override and assert `xtrax.__file__` in tests' session.

## Acceptance (tests under tests/sampling/ and tests/run/; narrow local runs only)

- A0 Refactor is behaviour-preserving: existing `tests/sampling/test_sample.py` and
  `test_sample_call_kw_contract.py` pass unchanged (run those two files only).
- A1 Parity: for N in {1, 5}, `HoistedSampler(keys, **s)` equals
  `jax.vmap(lambda k: sample_sequences(k, **s))(keys)`: sequences and decoding orders
  exactly; logits `allclose(rtol=1e-5, atol=1e-6)`. Cases: single-state protein;
  S=2 multi-state with `state_weights`; one case with tie groups.
- A2 Encoder runs once, counted at execution (B4, revised 261020 after the shape premise failed).
  The original jaxpr-shape check assumed the `(L, K, E)` encoder output is produced only by the
  encoder. That premise is false: the AR decode scan also produces per-key `(N, L, K, E)` values
  (observed as `jit/scan` ops on `float32[4,12,8,32]`), so shape matching cannot isolate the
  encoder. The check is now an execution counter. `encode_for_sampling` is wrapped (in
  `aminx.inference.sample_autoregressive` and in `aminx.sampling.hoisted`, where it is imported by
  name) by a wrapper that calls `jax.debug.callback` and then the real function. The callback adds
  `size(neighbor_indices) // (L*K)`, so a batched call counts once per key. Assertions, after
  `jax.effects_barrier()`: monolithic `jax.vmap(sample_sequences)` over N=5 keys counts N; the
  hoisted sampler counts 1 per call; a mutant whose `per_element` calls `encode_for_sampling` counts
  N+1 (one shared encode from the library stage plus N per-element encodes).
- A3 `HoistVaryingInputError` surfaces through `HoistedSampler` construction/call when
  a caller wires keys into the shared args (this exercises xtrax's name check through
  aminx's axis declaration; stated as such).
- A4 `check_elements=2` passes on A1 inputs; a reference that perturbs logits by 1e-3
  raises `HoistParityError`.
- A5 `sample_sequences(..., precomputed_node_features=x)` warns `FutureWarning`;
  outputs unchanged.
- A6 `runner.inspect` with `decoded_node_features` + `encoded_node_features` +
  `edge_features` on one structure: monkeypatch
  `aminx.sampling.conditional_logits.make_encoding_conditional_logits_split_fn` (as
  imported by `runner`) to return a counting wrapper around the real encode fn; assert
  one encode call per structure (Python calls, not traces) and outputs equal to the
  unpatched pre-change values on the same fixture.
- A7 Two calls with different Python-float temperatures compile the shared stage once
  (`stats()` or `jax.log_compiles` count).
- Public names in `aminx.sampling.__all__`; ruff `--no-fix` + `ruff format --check` + ty
  clean; CHANGELOG entry.

## Out of scope

tev_design changes (next PR, after an aminx release). Removing `precomputed_*`.
Host `runner.sample` (already encodes once per structure: kernel_dispatch.py:445,562).
GPU memory measurements.

## Risks

| Risk | Mitigation |
|---|---|
| Key stream drifts from the monolithic sampler | A1 exact sequence equality; per_element mirrors :172 and kernel :124 splits |
| Encode reads a per-key input not seen here | A1 + A2 + parity probe; fixed_mask read only with atom_37 (encode.py:64), never passed by sample_sequences |
| Unbatched vs batched encoder gives ulp differences that flip an argmax | logits allclose; sequences exact on fixtures; if a flip appears, report rather than loosen |
| xtrax 0.4.1b2 not yet on PyPI | implement against a local xtrax path in an untracked `uv.toml`, pin bump lands when published |
