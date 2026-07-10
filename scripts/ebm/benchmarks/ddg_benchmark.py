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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--lengths",
    type=str,
    default=",".join(str(n) for n in DEFAULT_LENGTHS),
    help="Comma-separated bucket-aligned residue counts (default: the EPIC's locked buckets).",
  )
  parser.add_argument("--n-mutants", type=int, default=16, help="Mutant-axis cardinality per length.")
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


def _build_reference_model(reference_repo: Path, ckpt: dict[str, Any]) -> "ProteinEBM":
  """Construct + strict-load the reference PyTorch ``ProteinEBM`` from ``ckpt``.

  Mirrors ``scripts/ebm/checkpoint_parity_check.py::_build_reference_model``
  exactly (same construction + ``load_state_dict(strict=False)`` + missing/
  unexpected check). Duplicated here rather than imported: that script is a
  standalone entry point (no package ``__init__.py`` under ``scripts/ebm/``),
  not a module E11b should import-couple to, and E3.5's sibling script must
  not be modified by this node.
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


def _time_jax_throughput(
  model: "ProteinEBMModel",
  coords: jax.Array,
  mutant_aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> tuple[float, float]:
  """Time ``score_mutant_ensemble``'s Vmap/SafeMap-dispatched batched mutant scoring.

  One untimed warmup call (forces ``eqx.filter_jit`` tracing/compilation --
  methodology point 3), then ``n_repeats`` timed calls. Returns
  ``(energy_evals_per_sec, elapsed_seconds)``.
  """
  from aminx.ebm.dispatch import score_mutant_ensemble  # noqa: PLC0415

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", c: jax.Array, ma: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    return score_mutant_ensemble(m, c, ma, tt, mk)

  result = _run(model, coords, mutant_aatype, t, mask)
  jax.block_until_ready(result)

  start = time.perf_counter()
  for _ in range(n_repeats):
    result = _run(model, coords, mutant_aatype, t, mask)
  jax.block_until_ready(result)
  elapsed = time.perf_counter() - start

  n_mutants = mutant_aatype.shape[0]
  energy_evals_per_sec = (n_mutants * n_repeats) / elapsed
  return energy_evals_per_sec, elapsed


def _time_jax_score_latency(
  model: "ProteinEBMModel",
  coords: jax.Array,
  aatype_single: jax.Array,
  t: jax.Array,
  mask: jax.Array,
  n_repeats: int,
) -> float:
  """Time ``ProteinEBMModel.score`` (1st-order conservative score, ``-jax.grad(E)``) for one mutant.

  One untimed warmup call, then ``n_repeats`` timed calls. Returns
  milliseconds per call.
  """

  @eqx.filter_jit
  def _run(m: "ProteinEBMModel", c: jax.Array, a: jax.Array, tt: jax.Array, mk: jax.Array) -> jax.Array:
    return m.score(c, a, tt, mk)

  result = _run(model, coords, aatype_single, t, mask)
  jax.block_until_ready(result)

  start = time.perf_counter()
  for _ in range(n_repeats):
    result = _run(model, coords, aatype_single, t, mask)
  jax.block_until_ready(result)
  elapsed = time.perf_counter() - start
  return (elapsed / n_repeats) * 1000.0


def _time_pytorch_throughput(
  torch_model: "ProteinEBM",
  coords_np: np.ndarray,
  mutant_aatype_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
) -> tuple[float, float]:
  """Time a single plain batched PyTorch forward (batch dim = mutant count, no per-mutant loop).

  Uses ``torch.no_grad()`` -- no autograd graph at all -- since pure
  throughput needs no gradient (methodology point 2, taken to its logical
  conclusion: strictly cheaper than even ``create_graph=False``). One
  untimed warmup call, then ``n_repeats`` timed calls. Returns
  ``(energy_evals_per_sec, elapsed_seconds)``.
  """
  import torch  # noqa: PLC0415

  n_mutants, n = mutant_aatype_np.shape
  r_noisy = torch.tensor(np.broadcast_to(coords_np, (n_mutants, n, 3)).copy(), dtype=torch.float32)
  aatype = torch.tensor(mutant_aatype_np, dtype=torch.long)
  residue_idx = torch.arange(n).unsqueeze(0).expand(n_mutants, n).clone()
  chain_id = torch.zeros(n_mutants, n, dtype=torch.long)
  contacts = torch.zeros(n_mutants, n, dtype=torch.long)
  mask = torch.tensor(np.broadcast_to(mask_np, (n_mutants, n)).copy(), dtype=torch.bool)
  times = torch.full((n_mutants,), t, dtype=torch.float32)
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
    torch_model.compute_energy(feats, rescale_input_coords=False)  # untimed warmup
    start = time.perf_counter()
    for _ in range(n_repeats):
      torch_model.compute_energy(feats, rescale_input_coords=False)
    elapsed = time.perf_counter() - start

  energy_evals_per_sec = (n_mutants * n_repeats) / elapsed
  return energy_evals_per_sec, elapsed


def _time_pytorch_score_latency(
  torch_model: "ProteinEBM",
  coords_np: np.ndarray,
  aatype_single_np: np.ndarray,
  t: float,
  mask_np: np.ndarray,
  n_repeats: int,
) -> float:
  """Time PyTorch's 1st-order conservative-score latency for one mutant.

  **Methodology point 2 (critical):** the reference's own
  ``ProteinEBM.compute_score`` always calls ``torch.autograd.grad(...,
  create_graph=True)``. Called at inference, that unnecessarily builds and
  retains a graph capable of a 2nd backward pass -- pure overhead here, and
  would unfairly bias this latency number against JAX (whose ``jax.grad`` has
  no such flag/cost to begin with). This function bypasses ``compute_score``
  and calls ``torch.autograd.grad(..., create_graph=False)`` directly, so
  both sides pay for exactly one first-order gradient. One untimed warmup
  call, then ``n_repeats`` timed calls. Returns milliseconds per call.
  """
  import torch  # noqa: PLC0415

  n = aatype_single_np.shape[0]

  def _one_call() -> None:
    r_noisy = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    aatype = torch.tensor(aatype_single_np, dtype=torch.long).unsqueeze(0)
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
    energy = torch_model.compute_energy(feats, rescale_input_coords=False)["energy"]
    torch.autograd.grad(energy.sum(), r_noisy, create_graph=False)

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
  if args.n_mutants <= 0:
    problems.append(f"invalid --n-mutants: {args.n_mutants}")
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
  log.info("[L1 PASS] paths + imports OK for lengths=%s n_mutants=%d n_repeats=%d", lengths, args.n_mutants, args.n_repeats)
  return 0


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args(argv)

  lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

  if args.dry_run:
    return _run_dry_run(args, lengths)

  if args.smoke:
    lengths = [_SMOKE_LENGTH]
    args.n_mutants = min(args.n_mutants, _SMOKE_N_MUTANTS)
    args.n_repeats = min(args.n_repeats, _SMOKE_N_REPEATS)
    log.info(
      "[--smoke] Overriding to lengths=%s n_mutants=%d n_repeats=%d for a <60s CPU run.",
      lengths,
      args.n_mutants,
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
    log.info("=== length=%d ===", length)
    coords_np, wildtype_np = _make_synthetic_wildtype(args.seed, length)
    wildtype_jax = jnp.asarray(wildtype_np, dtype=jnp.int32)
    mutant_aatype_jax, _mutations = _make_mutants(args.seed, wildtype_jax, args.n_mutants, length)
    mutant_aatype_np = np.asarray(mutant_aatype_jax)

    coords_jax = jnp.asarray(coords_np)
    mask_jax = jnp.ones((length,), dtype=bool)
    t_jax = jnp.asarray(args.diffusion_time)
    mask_np = np.ones((length,), dtype=bool)

    log.info(
      "JAX: warmup + timing throughput (score_mutant_ensemble, %d mutants x %d repeats)...",
      args.n_mutants,
      args.n_repeats,
    )
    jax_throughput, jax_elapsed = _time_jax_throughput(
      jax_model, coords_jax, mutant_aatype_jax, t_jax, mask_jax, args.n_repeats,
    )
    log.info("JAX: warmup + timing score_grad_ms (1st-order conservative score, %d repeats)...", args.n_repeats)
    jax_score_ms = _time_jax_score_latency(
      jax_model, coords_jax, mutant_aatype_jax[0], t_jax, mask_jax, args.n_repeats,
    )

    rows.append(
      {
        "protein_length": length,
        "device": jax_device,
        "impl": "jax",
        "energy_evals_per_sec": jax_throughput,
        "score_grad_ms": jax_score_ms,
      },
    )
    log.info(
      "[jax]     L=%-4d energy_evals_per_sec=%12.2f score_grad_ms=%8.3f (throughput wall=%.3fs)",
      length,
      jax_throughput,
      jax_score_ms,
      jax_elapsed,
    )

    if torch_model is not None:
      log.info(
        "PyTorch: warmup + timing throughput (plain batched forward, %d mutants x %d repeats)...",
        args.n_mutants,
        args.n_repeats,
      )
      torch_throughput, torch_elapsed = _time_pytorch_throughput(
        torch_model, coords_np, mutant_aatype_np, args.diffusion_time, mask_np, args.n_repeats,
      )
      log.info("PyTorch: warmup + timing score_grad_ms (create_graph=False explicit, %d repeats)...", args.n_repeats)
      torch_score_ms = _time_pytorch_score_latency(
        torch_model, coords_np, mutant_aatype_np[0], args.diffusion_time, mask_np, args.n_repeats,
      )

      rows.append(
        {
          "protein_length": length,
          "device": torch_device,
          "impl": "pytorch",
          "energy_evals_per_sec": torch_throughput,
          "score_grad_ms": torch_score_ms,
        },
      )
      log.info(
        "[pytorch] L=%-4d energy_evals_per_sec=%12.2f score_grad_ms=%8.3f (throughput wall=%.3fs)",
        length,
        torch_throughput,
        torch_score_ms,
        torch_elapsed,
      )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("w") as f:
    json.dump(rows, f, indent=2)
  log.info("Wrote %d rows to %s", len(rows), args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
