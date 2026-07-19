"""Throughput/latency benchmark: JAX ProteinEBM ΔΔG-stability path vs PyTorch reference (backlog node **E11b**).

Per the EPIC DAG (`.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`
§2 `E6─>E11b` row, §4.3) and design spec §8.3, this is the throughput/latency
counterpart to E6's accuracy-gated ΔΔG logic (`aminx.ebm.ddg_stability`) --
it measures *speed*, not correctness. Benchmarks the JAX-ported
`aminx.ebm.model.ProteinEBMModel` (via `aminx.ebm.dispatch.
score_mutant_ensemble`'s Vmap/SafeMap axis dispatch, E4) against the
original PyTorch `ProteinEBM` reference (`~/repos/ProteinEBM`), loaded the
same way `scripts/ebm/checkpoint_parity_check.py` does.

**Apples-to-apples methodology (design spec §8.3, EPIC risk MINOR-2 --
read before trusting any number this script produces):**

1. **Bucket-aligned lengths.** Sweep exactly `{64, 128, 256, 512}` residues
   (the EPIC's locked bucket boundaries, decision 6) -- synthetic coordinates,
   zero padding waste at any of the four points.
2. **PyTorch must not retain the autograd graph unnecessarily at inference.**
   The reference's own `ProteinEBM.compute_score` always calls
   `torch.autograd.grad(..., create_graph=True)` (needed elsewhere for its
   *training*-time 2nd-order use) -- at inference that unnecessarily builds
   and retains a second-order-capable graph, unfairly biasing a PyTorch-vs-
   JAX latency comparison in JAX's favor. This script bypasses
   `compute_score` and calls `torch.autograd.grad(energy.sum(), r_noisy,
   create_graph=False)` directly (see `_time_pytorch_score_latency`), mirroring
   `checkpoint_parity_check.py`'s same explicit workaround. The pure-
   throughput measurement (`_time_pytorch_throughput`) goes one step further
   and uses `torch.no_grad()` -- no autograd graph at all -- since no gradient
   is needed for a plain batched forward pass.
3. **Exclude JIT/compile warmup.** One untimed warmup call happens before
   every timed loop, on both implementations -- forces `eqx.filter_jit`
   tracing/compilation on the JAX side, and (for parity, even though PyTorch
   eager mode has no comparable compile step) exercises PyTorch's own
   first-call costs (cuDNN/oneDNN algorithm selection, allocator warmup)
   before timing.
4. **Metrics** (`[result_schema]` in the sidecar):
   `energy_evals_per_sec` -- batched mutant-scoring throughput. JAX side goes
   through `score_mutant_ensemble`'s Vmap/SafeMap dispatch (the real E4/E6
   composition path); PyTorch side is "a plain batched forward" (one
   `compute_energy` call with a batch dimension = mutant count -- no per-
   mutant Python loop), matching the design spec's explicit contrast.
   `score_grad_ms` -- 1st-order conservative-score latency, per mutant
   (single-element JAX `model.score` / PyTorch `create_graph=False` grad).
5. Structured JSON output (bathos-compatible; schema declared in
   `ddg_benchmark.bth.toml`), not printed logs.

**SM120 note (not exercised by this script's local CPU runs).** Per
`~/.claude/rules/CLUSTER.md` §2 and design spec §8.3, any GPU run of this
benchmark on `pi_so3` (SM120 / Blackwell, node4007/node4008) *must* have
`XLA_FLAGS=--xla_gpu_shard_autotuning=false` set before JAX is imported, or
throughput numbers come out ~1000x wrong (XLA autotuning hangs on that
hardware). This script sets it via `os.environ.setdefault` before importing
`jax` (mirroring `scripts/benchmarks/bench_aminx_jax.py`'s established
pattern in this repo) -- safe/no-op on non-Blackwell hardware, including the
CPU-only environment this script's own `--dry-run`/`--smoke` gates were
validated in. This module's smoke test is CPU-only by construction (this
dev environment has no GPU); the real `{64,128,256,512}` x GPU sweep is a
follow-on cluster job (L3), not run by this script.

**Scope, honestly bounded.** This is a *mechanism* benchmark: are the two
implementations fast/slow relative to each other on the SAME inputs? It says
nothing about whether either implementation's ΔΔG predictions are accurate
(that is E6's Spearman-0.686 accuracy gate, `claim.bth.toml` in the design
spec, an entirely separate instrument). The mutant sequences used here are
synthetic single-point substitutions (`aminx.ebm.ddg_stability.
random_point_mutants`) at random positions -- realistic *shape*, not real
ProteinGym/Tsuboyama mutants (E6's module docstring already documents that
the real experimental dataset is not downloaded in this environment).
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
  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

  from aminx.ebm.model import ProteinEBMModel

log = logging.getLogger("ddg_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()
DEFAULT_LENGTHS = (64, 128, 256, 512)
DEFAULT_DIFFUSION_TIME = 0.05

# --smoke overrides (design spec's smallest locked bucket point; a handful of
# mutants/repeats so the whole harness -- checkpoint load, reference-model
# build, JAX port, jit warmup, both timing loops -- proves out well under 60s
# on CPU; see the local L2 gate numbers this script was validated with).
_SMOKE_LENGTH = 64
_SMOKE_N_MUTANTS = 2
_SMOKE_N_REPEATS = 2
DEFAULT_BATCH_SIZES: tuple[int, ...] = (4, 16, 64, 256)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--lengths",
    type=str,
    default=",".join(str(n) for n in DEFAULT_LENGTHS),
    help="Comma-separated bucket-aligned residue counts (default: the EPIC's locked buckets).",
  )
  parser.add_argument("--n-mutants", type=int, default=None, help="Single mutant-axis cardinality (shorthand for --batch-sizes N). Mutually exclusive with --batch-sizes.")
  parser.add_argument(
    "--batch-sizes", type=int, nargs="+", default=None,
    help="Mutant-axis cardinalities to sweep, e.g. '--batch-sizes 4 16 64 256'. Default: (4, 16, 64, 256).",
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
    help=f"L2 gate: tiny end-to-end run (L={_SMOKE_LENGTH}, {_SMOKE_N_MUTANTS} mutants, "
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


def _build_reference_model(reference_repo: Path, ckpt: dict[str, Any], device: "str | torch.device" = "cpu") -> "ProteinEBM":
  """Construct + strict-load the reference PyTorch ``ProteinEBM`` from ``ckpt``.

  Mirrors ``scripts/ebm/checkpoint_parity_check.py::_build_reference_model``
  exactly (same construction + ``load_state_dict(strict=False)`` + missing/
  unexpected check). Duplicated here rather than imported: that script is a
  standalone entry point (no package ``__init__.py`` under ``scripts/ebm/``),
  not a module E11b should import-couple to, and E3.5's sibling script must
  not be modified by this node.

  ``device`` defaults to ``"cpu"`` (matches the checkpoint's own
  ``map_location="cpu"`` load) but callers doing real throughput measurement
  MUST pass the actual compute device -- the model is moved there via
  ``.to(device)`` before returning. Without this, every PyTorch number this
  script reports would silently be a CPU number, no matter what the JSON's
  ``device`` field claims (this was a real, previously-undiscovered bug: see
  ``.praxia/docs/audits/260716_proteinebm-parity-report.md`` §7).
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
  return model.to(device)


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


def _make_synthetic_wildtype(seed: int, n_residues: int) -> tuple[np.ndarray, np.ndarray]:
  """Fixed synthetic backbone: standard-normal coords (already-scaled space) + random sequence.

  Excludes the mask token (index 20, ``aminx.utils.aa_convert.MPNN_ALPHABET``'s
  last letter) from the wildtype draw -- a "wildtype" sequence should never
  itself be the mask/unknown token.
  """
  rng = np.random.default_rng(seed)
  coords = rng.normal(scale=1.0, size=(n_residues, 3)).astype(np.float32)
  aatype = rng.integers(0, 20, size=(n_residues,)).astype(np.int64)
  return coords, aatype


def _make_mutants(
  seed: int, wildtype_aatype: jax.Array, n_mutants: int, n_residues: int,
) -> tuple[jax.Array, list[tuple[int, str]]]:
  """Build ``n_mutants`` synthetic point-mutant rows via E6's ``random_point_mutants``.

  Positions are drawn without replacement up to ``n_residues`` distinct
  positions; if ``n_mutants > n_residues`` positions cycle (duplicate
  positions get the SAME substitution -- ``random_point_mutants`` folds the
  PRNG key by position, so this is deterministic, not a bug). Duplicate
  mutant rows are harmless for a throughput/latency measurement (the axis
  cardinality -- what actually drives dispatch strategy and timing -- is
  unaffected); logged so it's not silently confusing.
  """
  from aminx.ebm.ddg_stability import random_point_mutants  # noqa: PLC0415

  rng = np.random.default_rng(seed)
  n_positions = min(n_mutants, n_residues)
  base_positions = rng.choice(n_residues, size=n_positions, replace=False).tolist()
  if n_mutants > n_residues:
    log.warning(
      "n_mutants=%d > n_residues=%d: %d position(s) will repeat (deterministic duplicate mutant rows; "
      "harmless for throughput/latency timing).",
      n_mutants,
      n_residues,
      n_mutants - n_residues,
    )
    positions = [base_positions[i % n_positions] for i in range(n_mutants)]
  else:
    positions = base_positions
  key = jax.random.PRNGKey(seed + 1)
  return random_point_mutants(key, wildtype_aatype, positions)


def _wall_clock_ms_stats(times_seconds: list[float]) -> tuple[float, float]:
  """(mean_ms, std_ms) from raw per-call wall-clock seconds."""
  arr_ms = np.asarray(times_seconds) * 1000.0
  return float(np.mean(arr_ms)), float(np.std(arr_ms))


def _time_jax_throughput(
  model: "ProteinEBMModel",
  coords: jax.Array,
  mutant_aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> tuple[float, list[float]]:
  """Time ``score_mutant_ensemble``'s Vmap/SafeMap-dispatched batched mutant scoring.

  One untimed warmup call (forces ``eqx.filter_jit`` tracing/compilation --
  methodology point 3), then ``n_repeats`` individually-timed calls. Returns
  ``(energy_evals_per_sec, per_call_seconds)`` -- the per-call list is the raw
  wall-clock data (previously discarded once the aggregate throughput was
  computed).
  """
  from aminx.ebm.dispatch import score_mutant_ensemble  # noqa: PLC0415

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", c: jax.Array, ma: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    return score_mutant_ensemble(m, c, ma, tt, mk)

  jax.block_until_ready(_run(model, coords, mutant_aatype, t, mask))  # untimed warmup

  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    jax.block_until_ready(_run(model, coords, mutant_aatype, t, mask))
    times.append(time.perf_counter() - start)

  n_mutants = mutant_aatype.shape[0]
  energy_evals_per_sec = n_mutants / float(np.mean(times))
  return energy_evals_per_sec, times


def _time_jax_score_latency(
  model: "ProteinEBMModel",
  coords: jax.Array,
  aatype_single: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> tuple[float, list[float]]:
  """Time ``ProteinEBMModel.score`` (1st-order conservative score, ``-jax.grad(E)``) for one mutant.

  One untimed warmup call, then ``n_repeats`` individually-timed calls.
  Returns ``(mean_ms_per_call, per_call_seconds)``.
  """

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", c: jax.Array, a: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    return m.score(c, a, tt, mk)

  jax.block_until_ready(_run(model, coords, aatype_single, t, mask))  # untimed warmup

  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    jax.block_until_ready(_run(model, coords, aatype_single, t, mask))
    times.append(time.perf_counter() - start)
  return float(np.mean(times)) * 1000.0, times


def _time_pytorch_throughput(
  torch_model: "ProteinEBM",
  coords_np: np.ndarray,
  mutant_aatype_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> tuple[float, list[float]]:
  """Time a single plain batched PyTorch forward (batch dim = mutant count, no per-mutant loop).

  Uses ``torch.no_grad()`` -- no autograd graph at all -- since pure
  throughput needs no gradient (methodology point 2, taken to its logical
  conclusion: strictly cheaper than even ``create_graph=False``). One
  untimed warmup call, then ``n_repeats`` individually-timed calls. Returns
  ``(energy_evals_per_sec, per_call_seconds)``.
  """
  import torch  # noqa: PLC0415

  n_mutants, n = mutant_aatype_np.shape
  r_noisy = torch.tensor(np.broadcast_to(coords_np, (n_mutants, n, 3)).copy(), dtype=torch.float32, device=device)
  aatype = torch.tensor(mutant_aatype_np, dtype=torch.long, device=device)
  residue_idx = torch.arange(n, device=device).unsqueeze(0).expand(n_mutants, n).clone()
  chain_id = torch.zeros(n_mutants, n, dtype=torch.long, device=device)
  contacts = torch.zeros(n_mutants, n, dtype=torch.long, device=device)
  mask = torch.tensor(np.broadcast_to(mask_np, (n_mutants, n)).copy(), dtype=torch.bool, device=device)
  times_t = torch.full((n_mutants,), t, dtype=torch.float32, device=device)
  feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": mask,
    "t": times_t,
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }

  with torch.no_grad():
    torch_model.compute_energy(feats, rescale_input_coords=False)  # untimed warmup
    times: list[float] = []
    for _ in range(n_repeats):
      start = time.perf_counter()
      torch_model.compute_energy(feats, rescale_input_coords=False)
      times.append(time.perf_counter() - start)

  energy_evals_per_sec = n_mutants / float(np.mean(times))
  return energy_evals_per_sec, times


def _build_score_feats(coords_np: np.ndarray, aatype_single_np: np.ndarray, t: float, mask_np: np.ndarray, device: "str | torch.device" = "cpu") -> "tuple[dict, torch.Tensor]":
  import torch  # noqa: PLC0415

  n = aatype_single_np.shape[0]
  r_noisy = torch.tensor(coords_np, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
  feats = {
    "r_noisy": r_noisy,
    "aatype": torch.tensor(aatype_single_np, dtype=torch.long, device=device).unsqueeze(0),
    "residue_idx": torch.arange(n, device=device).unsqueeze(0),
    "mask": torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0),
    "t": torch.tensor([t], dtype=torch.float32, device=device),
    "chain_encoding": torch.zeros(1, n, dtype=torch.long, device=device),
    "external_contacts": torch.zeros(1, n, dtype=torch.long, device=device),
  }
  return feats, r_noisy


def _time_pytorch_score_latency(
  torch_model: "ProteinEBM",
  coords_np: np.ndarray,
  aatype_single_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> tuple[float, list[float]]:
  """Time PyTorch's 1st-order conservative-score latency for one mutant (optimized-eager variant).

  **Methodology point 2 (critical):** the reference's own
  ``ProteinEBM.compute_score`` always calls ``torch.autograd.grad(...,
  create_graph=True)``. Called at inference, that unnecessarily builds and
  retains a graph capable of a 2nd backward pass -- pure overhead here, and
  would unfairly bias this latency number against JAX (whose ``jax.grad`` has
  no such flag/cost to begin with). This function bypasses ``compute_score``
  and calls ``torch.autograd.grad(..., create_graph=False)`` directly, so
  both sides pay for exactly one first-order gradient. One untimed warmup
  call, then ``n_repeats`` timed calls. Returns ``(mean_ms, per_call_seconds)``.
  """
  import torch  # noqa: PLC0415

  def _one_call() -> None:
    feats, r_noisy = _build_score_feats(coords_np, aatype_single_np, t, mask_np, device=device)
    energy = torch_model.compute_energy(feats, rescale_input_coords=False)["energy"]
    torch.autograd.grad(energy.sum(), r_noisy, create_graph=False)

  _one_call()  # untimed warmup
  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _one_call()
    times.append(time.perf_counter() - start)
  return float(np.mean(times)) * 1000.0, times


def _time_pytorch_score_shipped(
  torch_model: "ProteinEBM", coords_np: np.ndarray, aatype_single_np: np.ndarray, t: float, mask_np: np.ndarray, n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> tuple[float, list[float]]:
  """PyTorch score latency via the reference's own public ``compute_score`` wrapper, verbatim
  (``create_graph=True`` unconditionally -- "the public code" as literally shipped; see
  ``decoy_benchmark.py::_run_pytorch_score_shipped`` for the identical rationale)."""
  feats, _ = _build_score_feats(coords_np, aatype_single_np, t, mask_np, device=device)

  def _one_call() -> None:
    torch_model.compute_score(feats)

  _one_call()
  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _one_call()
    times.append(time.perf_counter() - start)
  return float(np.mean(times)) * 1000.0, times


def _time_pytorch_score_compiled(
  torch_model: "ProteinEBM", coords_np: np.ndarray, aatype_single_np: np.ndarray, t: float, mask_np: np.ndarray, n_repeats: int,
  device: "str | torch.device" = "cpu",
) -> tuple[float | None, list[float] | None, str | None]:
  """PyTorch score latency via ``torch.compile`` wrapping the optimized-eager path.

  See ``decoy_benchmark.py::_run_pytorch_score_compiled``'s docstring for the precise, isolated-repro
  -confirmed characterization of the ``aminx.training`` stub-module interaction this can hit in this
  process -- not re-derived here, same finding, same honest-failure handling (returns
  ``(None, None, error_message)`` rather than a silently-mislabeled eager fallback).
  """
  import torch  # noqa: PLC0415

  feats, r_noisy = _build_score_feats(coords_np, aatype_single_np, t, mask_np, device=device)

  def _score_fn(f: dict) -> "torch.Tensor":
    energy = torch_model.compute_energy(f, rescale_input_coords=False)["energy"]
    return torch.autograd.grad(energy.sum(), r_noisy, create_graph=False)[0]

  try:
    compiled_fn = torch.compile(_score_fn)
    compiled_fn(feats)  # untimed: forces lazy first-call compilation
  except Exception as exc:  # noqa: BLE001 -- real, reportable compile failure
    return None, None, f"{type(exc).__name__}: {exc}"

  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    compiled_fn(feats)
    times.append(time.perf_counter() - start)
  return float(np.mean(times)) * 1000.0, times, None


def _resolve_batch_sizes(args: argparse.Namespace) -> tuple[int, ...]:
  """Resolve the mutant-count sweep from --n-mutants (single) or --batch-sizes (list)."""
  if args.n_mutants is not None and args.batch_sizes is not None:
    msg = "--n-mutants and --batch-sizes are mutually exclusive"
    raise SystemExit(msg)
  if args.n_mutants is not None:
    return (args.n_mutants,)
  if args.batch_sizes is not None:
    return tuple(args.batch_sizes)
  return DEFAULT_BATCH_SIZES


def _run_dry_run(args: argparse.Namespace, lengths: list[int]) -> int:
  """L1 gate: validate args/paths/imports only -- no model construction, no timing, no network fetches."""
  log.info("[L1 dry-run] Validating args + imports + paths (no model construction, no timing)...")
  problems: list[str] = []
  batch_sizes = _resolve_batch_sizes(args)

  for length in lengths:
    if length <= 0:
      problems.append(f"invalid length: {length}")
  if any(b <= 0 for b in batch_sizes):
    problems.append(f"invalid batch size(s): {batch_sizes}")
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
    from aminx.ebm.ddg_stability import random_point_mutants  # noqa: F401, PLC0415
    from aminx.ebm.dispatch import score_mutant_ensemble  # noqa: F401, PLC0415
    from aminx.ebm.model import ProteinEBMModel  # noqa: F401, PLC0415
  except ImportError as exc:
    problems.append(f"aminx.ebm import failed: {exc}")

  if problems:
    for problem in problems:
      log.error("[L1 FAIL] %s", problem)
    return 1
  log.info("[L1 PASS] paths + imports OK for lengths=%s batch_sizes=%s n_repeats=%d", lengths, batch_sizes, args.n_repeats)
  return 0


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args(argv)

  lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
  batch_sizes = _resolve_batch_sizes(args)

  if args.dry_run:
    return _run_dry_run(args, lengths)

  if args.smoke:
    lengths = [_SMOKE_LENGTH]
    batch_sizes = (min(batch_sizes[0], _SMOKE_N_MUTANTS),)
    args.n_repeats = min(args.n_repeats, _SMOKE_N_REPEATS)
    log.info(
      "[--smoke] Overriding to lengths=%s batch_sizes=%s n_repeats=%d for a <60s CPU run.",
      lengths,
      batch_sizes,
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
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Building + strict-loading reference PyTorch model...")
    torch_model = _build_reference_model(args.reference_repo, ckpt, device=torch_device)
    n_params = sum(p.numel() for p in torch_model.parameters())
    log.info("Reference PyTorch model params: %d", n_params)
  else:
    log.warning("--skip-pytorch: JAX-only rows. NOT apples-to-apples -- harness-debugging only.")

  log.info("Building + porting JAX ProteinEBMModel...")
  jax_model = _build_jax_model(ckpt, args.seed)

  rows: list[dict[str, Any]] = []

  def _write() -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
      json.dump(rows, f, indent=2)

  for length in lengths:
    for n_mutants in batch_sizes:
      log.info("=== length=%d batch=%d ===", length, n_mutants)
      try:
        coords_np, wildtype_np = _make_synthetic_wildtype(args.seed, length)
        wildtype_jax = jnp.asarray(wildtype_np, dtype=jnp.int32)
        mutant_aatype_jax, _mutations = _make_mutants(args.seed, wildtype_jax, n_mutants, length)
        mutant_aatype_np = np.asarray(mutant_aatype_jax)

        coords_jax = jnp.asarray(coords_np)
        mask_jax = jnp.ones((length,), dtype=bool)
        t_jax = jnp.asarray(args.diffusion_time)
        mask_np = np.ones((length,), dtype=bool)

        jax_throughput, jax_energy_times = _time_jax_throughput(
          jax_model, coords_jax, mutant_aatype_jax, t_jax, mask_jax, args.n_repeats,
        )
        jax_energy_ms_mean, jax_energy_ms_std = _wall_clock_ms_stats(jax_energy_times)
        jax_score_ms, jax_score_times = _time_jax_score_latency(
          jax_model, coords_jax, mutant_aatype_jax[0], t_jax, mask_jax, args.n_repeats,
        )
        jax_score_ms_mean, jax_score_ms_std = _wall_clock_ms_stats(jax_score_times)

        if torch_model is not None:
          torch_throughput, torch_energy_times = _time_pytorch_throughput(
            torch_model, coords_np, mutant_aatype_np, args.diffusion_time, mask_np, args.n_repeats, device=torch_device,
          )
          torch_energy_ms_mean, torch_energy_ms_std = _wall_clock_ms_stats(torch_energy_times)

          torch_score_eager_ms, torch_score_eager_times = _time_pytorch_score_latency(
            torch_model, coords_np, mutant_aatype_np[0], args.diffusion_time, mask_np, args.n_repeats, device=torch_device,
          )
          _, torch_score_eager_ms_std = _wall_clock_ms_stats(torch_score_eager_times)

          torch_score_shipped_ms, torch_score_shipped_times = _time_pytorch_score_shipped(
            torch_model, coords_np, mutant_aatype_np[0], args.diffusion_time, mask_np, args.n_repeats, device=torch_device,
          )
          _, torch_score_shipped_ms_std = _wall_clock_ms_stats(torch_score_shipped_times)

          torch_score_compiled_ms, torch_score_compiled_times, compile_err = _time_pytorch_score_compiled(
            torch_model, coords_np, mutant_aatype_np[0], args.diffusion_time, mask_np, args.n_repeats, device=torch_device,
          )
          if compile_err:
            log.warning("[length=%d batch=%d] torch.compile FAILED, honestly recorded: %s", length, n_mutants, compile_err)
            torch_score_compiled_ms_std = None
          else:
            _, torch_score_compiled_ms_std = _wall_clock_ms_stats(torch_score_compiled_times)
      except Exception as e:  # noqa: BLE001 -- a single (length, batch) cell must not lose already-collected rows
        log.error("[length=%d batch=%d] FAILED: %s: %s", length, n_mutants, type(e).__name__, e)
        rows.append({
          "protein_length": length,
          "batch_size": n_mutants,
          "impl": "error",
          "error": f"{type(e).__name__}: {e}",
        })
        _write()
        continue

      rows.append({
        "protein_length": length,
        "batch_size": n_mutants,
        "device": jax_device,
        "impl": "jax",
        "pytorch_variant": None,
        "energy_evals_per_sec": jax_throughput,
        "energy_wall_clock_mean_ms": jax_energy_ms_mean,
        "energy_wall_clock_std_ms": jax_energy_ms_std,
        "score_grad_ms": jax_score_ms,
        "score_wall_clock_mean_ms": jax_score_ms_mean,
        "score_wall_clock_std_ms": jax_score_ms_std,
        "compile_error": None,
      })
      log.info(
        "[jax]     L=%-4d B=%-4d energy_evals_per_sec=%12.2f score_grad_ms=%8.3f",
        length, n_mutants, jax_throughput, jax_score_ms,
      )

      if torch_model is not None:
        for variant, score_ms, score_ms_std in (
          ("eager", torch_score_eager_ms, torch_score_eager_ms_std),
          ("shipped", torch_score_shipped_ms, torch_score_shipped_ms_std),
          ("compiled", torch_score_compiled_ms, torch_score_compiled_ms_std),
        ):
          rows.append({
            "protein_length": length,
            "batch_size": n_mutants,
            "device": torch_device,
            "impl": "pytorch",
            "pytorch_variant": variant,
            "energy_evals_per_sec": torch_throughput,
            "energy_wall_clock_mean_ms": torch_energy_ms_mean,
            "energy_wall_clock_std_ms": torch_energy_ms_std,
            "score_grad_ms": score_ms,
            "score_wall_clock_mean_ms": score_ms,
            "score_wall_clock_std_ms": score_ms_std,
            "compile_error": compile_err if variant == "compiled" else None,
          })
        log.info(
          "[pytorch] L=%-4d B=%-4d energy_evals_per_sec=%12.2f score_grad_ms(eager)=%8.3f",
          length, n_mutants, torch_throughput, torch_score_eager_ms,
        )
      _write()

  log.info("Wrote %d rows to %s", len(rows), args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
