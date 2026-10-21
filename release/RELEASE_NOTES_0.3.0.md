# aminx 0.3.0

Beta, still largely experimental. This release moves to xtrax 0.4.1b2 and adds an encode-once sampler.

## What's new

- **`make_hoisted_sampler`** (`aminx.sampling`) runs the structure encoder once per call and maps only the decode over the PRNG keys, through `xtrax.inference.make_hoisted`. Output equals `vmap(sample_sequences)`: sequences and decoding orders exactly, logits to `rtol=1e-5, atol=1e-6`. `backbone_noise`, `wave_schedule` and `use_rolling_state` are not supported by it.
- **`runner.inspect`** encodes once per structure for `decoded_node_features`, `encoded_node_features` and `edge_features`. Outputs are unchanged.
- `sample_autoregressive.kernel` is split into `encode_for_sampling` and `decode_sampling`; `kernel` calls both and behaves as before.

## Read before upgrading

- **Breaking: `RunSpec.precision` is now `RunSpec.precision_config`.** xtrax >= 0.4.1b1 requires `RunSpec.precision` to be `str | None`. Replace `RunSpec(precision=PrecisionConfig(...))` with `precision_config=...`, and `spec.run_spec.precision.compute` with `spec.run_spec.precision_config.compute`. The portable JSON wire key `"precision"` is unchanged, so saved specs still load.
- xtrax is pinned to `==0.4.1b2`.
- `sample_sequences` now warns (`FutureWarning`) when a `precomputed_*` argument is passed. Those arguments were always ignored; use `make_hoisted_sampler` to skip repeated encoding.
- No numerical change for any existing path.
