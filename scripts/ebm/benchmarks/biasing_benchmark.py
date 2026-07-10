"""Throughput/latency benchmark: JAX ProteinEBM conformational-biasing path vs PyTorch reference (backlog node **E11c**).

Per the EPIC DAG (`.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`
§2 `E7─>E11c` row, §4.3) and design spec §8.3, this is the throughput/latency
counterpart to E7's accuracy-gated conformational-biasing/multistate logic
(`aminx.ebm.conformational_biasing`, `aminx.ebm.dispatch.
score_state_difference`) -- it measures *speed*, not correctness. Benchmarks
the JAX-ported `aminx.ebm.model.ProteinEBMModel` (via `score_state_difference`'s
Vmap axis dispatch + difference-Fuse, E4/E7) against the original PyTorch
`ProteinEBM` reference (`~/repos/ProteinEBM`), loaded the same way
`scripts/ebm/checkpoint_parity_check.py` does.

**Sibling scripts (E11a/E11b), same conventions, deliberately duplicated
code.** `scripts/ebm/benchmarks/decoy_benchmark.py` (E11a) and
`scripts/ebm/benchmarks/ddg_benchmark.py` (E11b) solve the same
checkpoint-loading/timing/JSON-schema problem for the decoy-ranking and
ΔΔG-stability applications respectively. This script mirrors their
`_build_reference_model`/`_build_jax_model`/`--dry-run`/`--smoke` structure
for consistency across the E11a-c triplet, but duplicates rather than
imports their private (`_`-prefixed) helpers -- each E11x script is a
standalone entry point with no shared package `__init__.py` under
`scripts/ebm/benchmarks/`, and this node must not modify either sibling file.

**Conformational biasing is always the S=2 case (design spec §5, EPIC DAG
§2 E7 row).** Unlike decoy ranking (D decoys) or ΔΔG (M mutants), the
multistate energy-gap application fixes exactly two conformational states of
one protein at one fixed sequence (e.g. LplA open/closed, `~/repos/
ProteinEBM/notebooks/confbiasing.ipynb`) -- there is no cardinality axis to
sweep here beyond the residue-length buckets. :data:`N_STATES` is therefore a
module constant, not a CLI flag.

**Apples-to-apples methodology (design spec §8.3, EPIC risk MINOR-2 -- read
before trusting any number this script produces):**

1. **Bucket-aligned lengths.** Sweep exactly `{64, 128, 256, 512}` residues
   (the EPIC's locked bucket boundaries, decision 6) -- synthetic per-state
   coordinates, zero padding waste at any of the four points. Two states per
   length (the standard S=2 conformational-biasing case), one fixed shared
   sequence.
2. **PyTorch must not retain the autograd graph unnecessarily at inference.**
   Conformational biasing never differentiates energy (it is a plain energy
   *gap*, `E(state_a,s) - E(state_b,s)`, no `-grad(E)` anywhere in the E7
   application, see `aminx.ebm.conformational_biasing.score_conformational_bias`
   -> `aminx.ebm.dispatch.score_state_difference` -> `ProteinEBMModel.energy`,
   never `.score`). So this script's PyTorch timing uses `torch.no_grad()`
   for every call -- strictly stronger than the `create_graph=False`
   discipline E11a/E11b document for their gradient-taking metrics (there is
   no autograd graph at all to retain here, so the discipline applies
   vacuously). Both PyTorch metrics below are energy-only, no backward pass.
3. **Exclude JIT/compile warmup.** One untimed warmup call happens before
   every timed loop, on both implementations -- forces `eqx.filter_jit`
   tracing/compilation on the JAX side (and, for parity, exercises PyTorch's
   own first-call costs -- allocator warmup, algorithm selection -- before
   timing, even though eager-mode PyTorch has no comparable compile step).
4. **Metrics** (`[result_schema]` in `biasing_benchmark.bth.toml`):
   - `energy_evals_per_sec` -- the 2-state Vmap dispatch throughput. JAX side
     dispatches `ProteinEBMModel.energy` over the 2-element state axis via
     `aminx.ebm.plan.plan_axis`/`dispatch_axis` (the same E4 Vmap wiring
     `score_state_difference` uses internally, minus the fuse reduction);
     PyTorch side is "a plain 2-call forward" -- two sequential, unbatched
     `compute_energy` calls (no batch-dim trick, matching the design spec's
     explicit "vs. a plain 2-call PyTorch forward" wording), under
     `torch.no_grad()`.
   - `diff_fuse_wall_clock_ms` -- the **full** difference-Fuse pipeline:
     energy for both states + the fuse (subtraction) reduction, per call,
     in milliseconds. JAX side times `aminx.ebm.dispatch.score_state_difference`
     itself (E4/E7's real composition seam, `eqx.filter_jit`-wrapped);
     PyTorch side times two sequential `compute_energy` calls plus the scalar
     subtraction (`torch.no_grad()`, no batching).
5. Structured JSON output (bathos-compatible; schema declared in
   `biasing_benchmark.bth.toml`), not printed logs.

**SM120 note (not exercised by this script's local CPU runs).** Per
`~/.claude/rules/CLUSTER.md` §2 and design spec §8.3, any GPU run of this
benchmark on `pi_so3` (SM120/Blackwell, node4007/node4008) *must* have
`XLA_FLAGS=--xla_gpu_shard_autotuning=false` set before JAX is imported, or
throughput numbers come out ~1000x wrong (XLA autotuning hangs on that
hardware). This script sets it via `os.environ.setdefault` before importing
`jax` (mirroring `ddg_benchmark.py`/`scripts/benchmarks/bench_aminx_jax.py`'s
established pattern) -- safe/no-op on non-Blackwell hardware, including the
CPU-only environment this script's own `--dry-run`/`--smoke` gates were
validated in. The real `{64,128,256,512}` x GPU sweep is a follow-on cluster
job (L3), NOT run by this script (out of scope for this dispatch -- see the
task brief).

**Scope, honestly bounded.** This is a *mechanism* benchmark: are the two
implementations fast/slow relative to each other on the SAME synthetic
inputs? It says nothing about whether either implementation's conformational-
bias energy gap correlates with real biological activity (that is E7's own
accuracy-gated application + the design spec §8.2 information-ablation probe,
an entirely separate instrument). Per-state coordinates here are i.i.d.
Gaussian in already-scaled ("nm") units, sharing one random sequence -- not
real LplA open/closed structures (E7's `aminx.ebm.conformational_biasing.
load_conformational_states` handles the real-PDB residue-alignment case;
this script never calls it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Must precede `import jax` -- see module docstring's SM120 note. Safe no-op
# on CPU/non-Blackwell hardware (this local environment).
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

  from aminx.ebm.model import ProteinEBMModel

log = logging.getLogger("biasing_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()
DEFAULT_LENGTHS = (64, 128, 256, 512)
DEFAULT_DIFFUSION_TIME = 0.05

# Conformational biasing is always the S=2 case (module docstring) -- not a
# CLI-configurable cardinality axis like decoy_benchmark's --n-decoys or
# ddg_benchmark's --n-mutants.
N_STATES = 2

# --smoke overrides (design spec's smallest locked bucket point; a handful of
# repeats so the whole harness -- checkpoint load, reference-model build, JAX
# port, jit warmup, both timing loops -- proves out well under 60s on CPU;
# mirrors ddg_benchmark.py's proven-fast L2 gate numbers).
_SMOKE_LENGTH = 64
_SMOKE_N_REPEATS = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--lengths",
    type=str,
    default=",".join(str(n) for n in DEFAULT_LENGTHS),
    help="Comma-separated bucket-aligned residue counts (default: the EPIC's locked buckets).",
  )
  parser.add_argument(
    "--n-repeats",
    type=int,
    default=10,
    help="Timed repeats per (length, impl) throughput/latency measurement (after one untimed warmup).",
  )
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--diffusion-time", type=float, default=DEFAULT_DIFFUSION_TIME, help="ProteinEBM-x MVP target t (design spec §9).",
  )
  parser.add_argument("--out", type=Path, default=None, help="JSON output path (required unless --dry-run).")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="L1 gate: validate args/paths/imports only -- no model construction, no timing, no network fetches.",
  )
  parser.add_argument(
    "--smoke",
    action="store_true",
    help=f"L2 gate: tiny end-to-end run (L={_SMOKE_LENGTH}, {N_STATES} states, "
    f"{_SMOKE_N_REPEATS} repeats) -- proves the harness runs correctly in well under 60s on CPU.",
  )
  parser.add_argument(
    "--skip-pytorch",
    action="store_true",
    help="Skip building/timing the PyTorch reference model (JAX-only rows). NOT apples-to-apples -- "
    "harness-debugging escape hatch only. The checkpoint is still loaded via torch (needed for the "
    "JAX model's own construction dims + ported weights).",
  )
  return parser.parse_args(argv)


def _build_reference_model(reference_repo: Path, ckpt: dict[str, Any]) -> "ProteinEBM":
  """Construct + strict-load the reference PyTorch ``ProteinEBM`` from ``ckpt``.

  Mirrors ``scripts/ebm/checkpoint_parity_check.py::_build_reference_model``
  exactly (same construction + ``load_state_dict(strict=False)`` + missing/
  unexpected check), also duplicated in ``decoy_benchmark.py``/
  ``ddg_benchmark.py`` (E11a/E11b) for the same reason: each E11x script is a
  standalone entry point, not import-coupled to any sibling.
  """
  sys.path.insert(0, str(reference_repo))
  from protein_ebm.model.ebm import ProteinEBM  # noqa: PLC0415
  from protein_ebm.model.r3_diffuser import R3Diffuser  # noqa: PLC0415

  cfg = ckpt["hyper_parameters"]["config"]
  diffuser = R3Diffuser(cfg.diffuser)
  model = ProteinEBM(cfg.model, diffuser)
  state_dict = ckpt["state_dict"]
  stripped = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
  missing, unexpected = model.load_state_dict(stripped, strict=False)
  if missing or unexpected:
    msg = f"reference model.load_state_dict was not exact: missing={missing}, unexpected={unexpected}"
    raise RuntimeError(msg)
  model.eval()
  return model


def _build_jax_model(ckpt: dict[str, Any], seed: int) -> "ProteinEBMModel":
  """Construct a ``ProteinEBMModel`` matching the checkpoint's own config, then port weights.

  Mirrors ``checkpoint_parity_check.py::_build_ported_jax_model``: reads
  construction dims straight from ``ckpt["hyper_parameters"]["config"].model``
  (E3.5/checkpoint.py's documented template dims: ``token_s=256, token_z=128,
  dim_fourier=256, conditioning_transition_layers=2, transformer_depth=16,
  transformer_heads=8, num_contact_embeddings=3``), then ports weights via
  the already-done ``aminx.ebm.checkpoint.load_pytorch_checkpoint`` (E3.5,
  not modified here -- only imported).
  """
  from aminx.ebm.checkpoint import load_pytorch_checkpoint  # noqa: PLC0415
  from aminx.ebm.model import ProteinEBMModel  # noqa: PLC0415

  cfg = ckpt["hyper_parameters"]["config"].model
  key = jax.random.PRNGKey(seed)
  model = ProteinEBMModel(
    token_s=cfg.token_s,
    token_z=cfg.token_z,
    dim_fourier=cfg.dim_fourier,
    conditioning_transition_layers=cfg.conditioning_transition_layers,
    transformer_depth=cfg.token_transformer_depth,
    transformer_heads=cfg.token_transformer_heads,
    num_contact_embeddings=cfg.num_contact_embeddings,
    key=key,
  )
  ported, _report = load_pytorch_checkpoint(model, ckpt["state_dict"])
  return ported


def _make_synthetic_states(seed: int, n_residues: int) -> tuple[np.ndarray, np.ndarray]:
  """Fixed synthetic ``(N_STATES, n_residues, 3)`` per-state coords + one shared sequence.

  Standard-normal coordinates (already-scaled space, i.i.d. per state --
  distinct "conformations" of the same synthetic protein) and one random
  shared ``aatype`` (the fixed-sequence contract ``score_state_difference``/
  ``score_conformational_bias`` require -- design spec §5: the multistate gap
  is computed for a *fixed* sequence across states).
  """
  rng = np.random.default_rng(seed)
  coords_states = rng.normal(scale=1.0, size=(N_STATES, n_residues, 3)).astype(np.float32)
  aatype = rng.integers(0, 21, size=(n_residues,)).astype(np.int64)
  return coords_states, aatype


def _time_jax_throughput(
  model: "ProteinEBMModel",
  coords_states: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> tuple[float, float]:
  """Time the raw 2-state Vmap energy dispatch (no fuse) -- the E4 axis-dispatch primitive.

  Uses the same ``aminx.ebm.plan.plan_axis``/``dispatch_axis`` pair
  ``score_state_difference`` calls internally (``EBMAxisNames.N_STATES``),
  minus the difference-Fuse reduction -- isolates pure per-state energy
  throughput from the fuse cost (see :func:`_time_jax_diff_fuse` for the full
  pipeline). One untimed warmup call (forces ``eqx.filter_jit``
  tracing/compilation -- methodology point 3), then ``n_repeats`` timed
  calls. Returns ``(energy_evals_per_sec, elapsed_seconds)``.
  """
  from aminx.ebm.plan import EBMAxisNames, dispatch_axis, plan_axis  # noqa: PLC0415

  decision = plan_axis(EBMAxisNames.N_STATES, N_STATES)

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", cs: jax.Array, a: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    def _score_one(c: jax.Array) -> jax.Array:
      return m.energy(c, a, tt, mk)

    return dispatch_axis(decision.strategy, _score_one, cs)

  result = _run(model, coords_states, aatype, t, mask)
  jax.block_until_ready(result)

  start = time.perf_counter()
  for _ in range(n_repeats):
    result = _run(model, coords_states, aatype, t, mask)
  jax.block_until_ready(result)
  elapsed = time.perf_counter() - start

  energy_evals_per_sec = (N_STATES * n_repeats) / elapsed
  return energy_evals_per_sec, elapsed


def _time_jax_diff_fuse(
  model: "ProteinEBMModel",
  coords_states: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> float:
  """Time the **full** difference-Fuse pipeline: energy for both states + the fuse reduction.

  Times ``aminx.ebm.dispatch.score_state_difference`` itself (E4/E7's real
  composition seam) directly, ``eqx.filter_jit``-wrapped. One untimed warmup
  call, then ``n_repeats`` timed calls. Returns milliseconds per call.
  """
  from aminx.ebm.dispatch import score_state_difference  # noqa: PLC0415

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", cs: jax.Array, a: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    return score_state_difference(m, cs, a, tt, mk)

  result = _run(model, coords_states, aatype, t, mask)
  jax.block_until_ready(result)

  start = time.perf_counter()
  for _ in range(n_repeats):
    result = _run(model, coords_states, aatype, t, mask)
  jax.block_until_ready(result)
  elapsed = time.perf_counter() - start
  return (elapsed / n_repeats) * 1000.0


def _pytorch_energy_for_state(
  torch_model: "ProteinEBM",
  coords_state_np: np.ndarray,
  aatype_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
) -> Any:  # noqa: ANN401 -- duck-typed torch.Tensor, torch is dev-only
  """One unbatched ``compute_energy`` call for a single conformational state, under ``no_grad``.

  No autograd graph is built at all (methodology point 2) -- conformational
  biasing never differentiates energy, so this is strictly cheaper than even
  the ``create_graph=False`` discipline E11a/E11b document for their
  gradient-taking metrics.
  """
  import torch  # noqa: PLC0415

  n = aatype_np.shape[0]
  r_noisy = torch.tensor(coords_state_np, dtype=torch.float32).unsqueeze(0)
  aatype = torch.tensor(aatype_np, dtype=torch.long).unsqueeze(0)
  residue_idx = torch.arange(n).unsqueeze(0)
  chain_id = torch.zeros(1, n, dtype=torch.long)
  contacts = torch.zeros(1, n, dtype=torch.long)
  mask = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0)
  times = torch.tensor([t], dtype=torch.float32)
  feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": mask,
    "t": times,
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }
  with torch.no_grad():
    return torch_model.compute_energy(feats, rescale_input_coords=False)["energy"]


def _time_pytorch_throughput(
  torch_model: "ProteinEBM",
  coords_states_np: np.ndarray,
  aatype_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
) -> tuple[float, float]:
  """Time "a plain 2-call PyTorch forward" -- two sequential, unbatched ``compute_energy`` calls.

  Deliberately NOT a single batched (batch_dim=2) call -- the design spec's
  own methodology wording (§8.3/EPIC §4.3, this script's own docstring point
  4) contrasts the JAX Vmap dispatch against "a plain 2-call PyTorch
  forward". One untimed warmup call (both states), then ``n_repeats`` timed
  repeats of (state_0 call + state_1 call). Returns
  ``(energy_evals_per_sec, elapsed_seconds)``.
  """

  def _one_round() -> None:
    _pytorch_energy_for_state(torch_model, coords_states_np[0], aatype_np, t, mask_np)
    _pytorch_energy_for_state(torch_model, coords_states_np[1], aatype_np, t, mask_np)

  _one_round()  # untimed warmup

  start = time.perf_counter()
  for _ in range(n_repeats):
    _one_round()
  elapsed = time.perf_counter() - start

  energy_evals_per_sec = (N_STATES * n_repeats) / elapsed
  return energy_evals_per_sec, elapsed


def _time_pytorch_diff_fuse(
  torch_model: "ProteinEBM",
  coords_states_np: np.ndarray,
  aatype_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
) -> float:
  """Time the full difference-Fuse pipeline on PyTorch: 2 energy calls + the subtraction.

  Mirrors :func:`_time_jax_diff_fuse`'s scope exactly (energy for both
  states + the fuse reduction), so the two numbers are directly comparable.
  One untimed warmup call, then ``n_repeats`` timed calls. Returns
  milliseconds per call.
  """

  def _one_call() -> None:
    e0 = _pytorch_energy_for_state(torch_model, coords_states_np[0], aatype_np, t, mask_np)
    e1 = _pytorch_energy_for_state(torch_model, coords_states_np[1], aatype_np, t, mask_np)
    (e0 - e1).item()  # force materialization, mirrors jax.block_until_ready

  _one_call()  # untimed warmup

  start = time.perf_counter()
  for _ in range(n_repeats):
    _one_call()
  elapsed = time.perf_counter() - start
  return (elapsed / n_repeats) * 1000.0


def _run_dry_run(args: argparse.Namespace, lengths: list[int]) -> int:
  """L1 gate: validate args/paths/imports only -- no model construction, no timing, no network fetches."""
  log.info("[L1 dry-run] Validating args + imports + paths (no model construction, no timing)...")
  problems: list[str] = []

  for length in lengths:
    if length <= 0:
      problems.append(f"invalid length: {length}")
  if args.n_repeats <= 0:
    problems.append(f"invalid --n-repeats: {args.n_repeats}")

  if not args.checkpoint.exists():
    problems.append(f"checkpoint not found: {args.checkpoint}")
  if not args.skip_pytorch and not args.reference_repo.exists():
    problems.append(f"reference repo not found: {args.reference_repo}")

  try:
    import torch  # noqa: F401, PLC0415
  except ImportError as exc:
    problems.append(f"torch import failed: {exc}")

  try:
    from aminx.ebm.checkpoint import load_pytorch_checkpoint  # noqa: F401, PLC0415
    from aminx.ebm.dispatch import score_state_difference  # noqa: F401, PLC0415
    from aminx.ebm.model import ProteinEBMModel  # noqa: F401, PLC0415
    from aminx.ebm.plan import EBMAxisNames, dispatch_axis, plan_axis  # noqa: F401, PLC0415
  except ImportError as exc:
    problems.append(f"aminx.ebm import failed: {exc}")

  if problems:
    for problem in problems:
      log.error("[L1 FAIL] %s", problem)
    return 1
  log.info("[L1 PASS] paths + imports OK for lengths=%s n_states=%d n_repeats=%d", lengths, N_STATES, args.n_repeats)
  return 0


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args(argv)

  lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

  if args.dry_run:
    return _run_dry_run(args, lengths)

  if args.smoke:
    lengths = [_SMOKE_LENGTH]
    args.n_repeats = min(args.n_repeats, _SMOKE_N_REPEATS)
    log.info(
      "[--smoke] Overriding to lengths=%s n_states=%d n_repeats=%d for a <60s CPU run.",
      lengths,
      N_STATES,
      args.n_repeats,
    )

  if args.out is None:
    log.error("--out is required unless --dry-run")
    return 1

  if not args.checkpoint.exists():
    log.error("Checkpoint not found: %s", args.checkpoint)
    return 1
  if not args.skip_pytorch and not args.reference_repo.exists():
    log.error("Reference repo not found: %s", args.reference_repo)
    return 1

  import torch  # noqa: PLC0415

  jax_device = jax.devices()[0].platform
  log.info("JAX device: %s", jax_device)

  log.info("Loading checkpoint: %s", args.checkpoint)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

  torch_model: ProteinEBM | None = None
  torch_device = "skipped"
  if not args.skip_pytorch:
    log.info("Building + strict-loading reference PyTorch model...")
    torch_model = _build_reference_model(args.reference_repo, ckpt)
    n_params = sum(p.numel() for p in torch_model.parameters())
    log.info("Reference PyTorch model params: %d", n_params)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    log.warning("--skip-pytorch: JAX-only rows. NOT apples-to-apples -- harness-debugging only.")

  log.info("Building + porting JAX ProteinEBMModel...")
  jax_model = _build_jax_model(ckpt, args.seed)

  rows: list[dict[str, Any]] = []

  for length in lengths:
    log.info("=== length=%d (n_states=%d) ===", length, N_STATES)
    coords_states_np, aatype_np = _make_synthetic_states(args.seed, length)
    mask_np = np.ones((length,), dtype=bool)

    coords_states_jax = jnp.asarray(coords_states_np)
    aatype_jax = jnp.asarray(aatype_np, dtype=jnp.int32)
    mask_jax = jnp.ones((length,), dtype=bool)
    t_jax = jnp.asarray(args.diffusion_time)

    log.info("JAX: warmup + timing energy_evals_per_sec (raw 2-state Vmap dispatch, %d repeats)...", args.n_repeats)
    jax_throughput, jax_elapsed = _time_jax_throughput(
      jax_model, coords_states_jax, aatype_jax, t_jax, mask_jax, args.n_repeats,
    )
    log.info("JAX: warmup + timing diff_fuse_wall_clock_ms (full score_state_difference, %d repeats)...", args.n_repeats)
    jax_diff_fuse_ms = _time_jax_diff_fuse(
      jax_model, coords_states_jax, aatype_jax, t_jax, mask_jax, args.n_repeats,
    )

    rows.append(
      {
        "protein_length": length,
        "device": jax_device,
        "impl": "jax",
        "energy_evals_per_sec": jax_throughput,
        "diff_fuse_wall_clock_ms": jax_diff_fuse_ms,
      },
    )
    log.info(
      "[jax]     L=%-4d energy_evals_per_sec=%12.2f diff_fuse_wall_clock_ms=%8.3f (throughput wall=%.3fs)",
      length,
      jax_throughput,
      jax_diff_fuse_ms,
      jax_elapsed,
    )

    if torch_model is not None:
      log.info(
        "PyTorch: warmup + timing energy_evals_per_sec (plain 2-call forward, %d repeats)...", args.n_repeats,
      )
      torch_throughput, torch_elapsed = _time_pytorch_throughput(
        torch_model, coords_states_np, aatype_np, args.diffusion_time, mask_np, args.n_repeats,
      )
      log.info("PyTorch: warmup + timing diff_fuse_wall_clock_ms (2 calls + subtract, %d repeats)...", args.n_repeats)
      torch_diff_fuse_ms = _time_pytorch_diff_fuse(
        torch_model, coords_states_np, aatype_np, args.diffusion_time, mask_np, args.n_repeats,
      )

      rows.append(
        {
          "protein_length": length,
          "device": torch_device,
          "impl": "pytorch",
          "energy_evals_per_sec": torch_throughput,
          "diff_fuse_wall_clock_ms": torch_diff_fuse_ms,
        },
      )
      log.info(
        "[pytorch] L=%-4d energy_evals_per_sec=%12.2f diff_fuse_wall_clock_ms=%8.3f (throughput wall=%.3fs)",
        length,
        torch_throughput,
        torch_diff_fuse_ms,
        torch_elapsed,
      )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("w") as f:
    json.dump(rows, f, indent=2)
  log.info("Wrote %d rows to %s", len(rows), args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
