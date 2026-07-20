"""Outer Langevin annealing schedule + E10 structure-prediction throughput (closes a real gap).

`langevin_benchmark.py` (E11d) explicitly scopes itself to the **inner**, single-fixed-`t`
Euler-Maruyama sampler only -- its own module docstring states the **outer** noise-schedule loop and
its model-swap dispatcher (`aminx.ebm.langevin_schedule.run_annealing_schedule`/
`select_model_for_t`), and the full E10 multi-round structure-prediction pipeline
(`aminx.ebm.structure_prediction.run_structure_prediction`), were never throughput-benchmarked --
only correctness-tested (`tests/ebm/test_langevin_schedule.py`, `tests/ebm/test_structure_prediction.py`).
This script is the first throughput measurement for both.

**Model-swap stand-in, stated honestly.** Only one real checkpoint (`model_6_expert_frozen_1m_md.pt`)
is available in this environment -- the same limitation the epic's own plan doc already documented
for E9's correctness tests ("the multi-checkpoint model-swap has never been exercised against two
REAL (non-toy) checkpoints simultaneously resident on one device"). This script uses the SAME ported
model twice as the two swap branches (matching that established stand-in convention exactly, not a
new limitation) -- this measures the swap DISPATCH mechanism's real overhead (the `lax.cond`/
`lax.switch` cost + carrying two live model pytrees), not the cost difference between two distinct
architectures.

**PyTorch comparison.** The reference's `run_dynamics.py` implements the equivalent outer loop
(`get_dynamics_model(t)` eager threshold swap + a Python round loop) but is a full CLI script (global
argparse execution, not an importable module) -- this benchmark reimplements the equivalent minimal
algorithmic shape (per-level Euler-Maruyama steps, eager `if t < threshold` model selection) rather
than importing it, mirroring how every other E11x script already duplicates small reference-side
helpers instead of cross-importing standalone scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.ebm.checkpoint import load_pytorch_checkpoint
from aminx.ebm.langevin_schedule import run_annealing_schedule
from aminx.ebm.model import ProteinEBMModel
from aminx.ebm.structure_prediction import run_structure_prediction

if TYPE_CHECKING:
  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

log = logging.getLogger("langevin_annealing_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()
DEFAULT_LENGTHS: tuple[int, ...] = (64, 128, 256, 512)
SMOKE_LENGTHS: tuple[int, ...] = (64,)
MODEL_SWAP_THRESHOLD = 0.1  # matches run_dynamics.py's real get_dynamics_model(t) threshold.
DEFAULT_NOISE_SCHEDULE = (0.5, 0.3, 0.15, 0.05)  # brackets the swap threshold on both sides.
DEFAULT_N_STEPS_PER_LEVEL = 5
DEFAULT_DT = 1e-3
DEFAULT_E10_ROUNDS = 3
DEFAULT_E10_BATCH_SIZE = 4

SMOKE_TOKEN_S, SMOKE_TOKEN_Z, SMOKE_DIM_FOURIER = 32, 16, 12
SMOKE_TRANSITION_LAYERS, SMOKE_DEPTH, SMOKE_HEADS, SMOKE_NUM_CONTACT_EMBEDDINGS = 1, 2, 2, 2


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--lengths", type=str, default=None)
  parser.add_argument("--n-repeats", type=int, default=None)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--e10-rounds", type=int, default=DEFAULT_E10_ROUNDS)
  parser.add_argument("--e10-batch-size", type=int, default=DEFAULT_E10_BATCH_SIZE)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  return args


def _smoke_model_configs() -> tuple[Any, Any]:
  from types import SimpleNamespace  # noqa: PLC0415

  model_cfg = SimpleNamespace(
    token_s=SMOKE_TOKEN_S, token_z=SMOKE_TOKEN_Z, dim_fourier=SMOKE_DIM_FOURIER,
    conditioning_transition_layers=SMOKE_TRANSITION_LAYERS, token_transformer_depth=SMOKE_DEPTH,
    token_transformer_heads=SMOKE_HEADS, num_contact_embeddings=SMOKE_NUM_CONTACT_EMBEDDINGS,
    aux_score=True, predict_sidechain=False, diffuse_sidechain=False,
    use_self_conditioning=True, use_present_embedding=False, use_attention_mask=False, direct_score=False,
  )
  diffuser_cfg = SimpleNamespace(min_b=0.1, max_b=20.0, coordinate_scaling=0.1)
  return model_cfg, diffuser_cfg


def _build_reference_model(reference_repo: Path, model_cfg: Any, diffuser_cfg: Any, state_dict: Any, device: "str | torch.device" = "cpu") -> "ProteinEBM":  # noqa: ANN401
  """Construct + (optionally) strict-load the reference ``ProteinEBM``.

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

  diffuser = R3Diffuser(diffuser_cfg)
  model = ProteinEBM(model_cfg, diffuser)
  if state_dict is not None:
    stripped = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
      msg = f"reference model.load_state_dict was not exact: missing={missing}, unexpected={unexpected}"
      raise RuntimeError(msg)
  model.eval()
  return model.to(device)


def _build_jax_model(model_cfg: Any, seed: int, state_dict: Any) -> ProteinEBMModel:  # noqa: ANN401
  key = jax.random.PRNGKey(seed)
  model = ProteinEBMModel(
    token_s=model_cfg.token_s, token_z=model_cfg.token_z, dim_fourier=model_cfg.dim_fourier,
    conditioning_transition_layers=model_cfg.conditioning_transition_layers,
    transformer_depth=model_cfg.token_transformer_depth, transformer_heads=model_cfg.token_transformer_heads,
    num_contact_embeddings=model_cfg.num_contact_embeddings, key=key,
  )
  if state_dict is not None:
    model, _report = load_pytorch_checkpoint(model, state_dict)
  return model


def _torch_device() -> str:
  import torch  # noqa: PLC0415

  return "cuda" if torch.cuda.is_available() else "cpu"


def build_models(args: argparse.Namespace) -> tuple[ProteinEBMModel, "ProteinEBM"]:
  """Build (jax_model, reference_model) per ``--smoke``/full-run mode.

  The reference PyTorch model is moved to ``_torch_device()``'s actual device
  -- previously it was left on CPU (the checkpoint's own ``map_location="cpu"``
  load target) regardless of GPU availability, so every reported PyTorch
  number was silently a CPU number.
  """
  device = _torch_device()
  if args.smoke:
    model_cfg, diffuser_cfg = _smoke_model_configs()
    return (
      _build_jax_model(model_cfg, args.seed, state_dict=None),
      _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None, device=device),
    )
  import torch  # noqa: PLC0415

  if not args.checkpoint.exists():
    msg = f"checkpoint not found: {args.checkpoint}"
    raise FileNotFoundError(msg)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model_cfg = ckpt["hyper_parameters"]["config"].model
  diffuser_cfg = ckpt["hyper_parameters"]["config"].diffuser
  return (
    _build_jax_model(model_cfg, args.seed, ckpt["state_dict"]),
    _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, ckpt["state_dict"], device=device),
  )


def _time_jax_outer_annealing(
  model: ProteinEBMModel, length: int, seed: int, n_repeats: int,
) -> list[float]:
  """Time run_annealing_schedule (outer scan + model-swap dispatch) using the same real model
  twice as the two swap branches (see module docstring's stand-in disclosure)."""
  rng = np.random.default_rng(seed)
  coords = jnp.asarray(rng.normal(scale=0.1, size=(length, 3)).astype(np.float32))
  aatype = jnp.asarray(rng.integers(0, 21, size=(length,)).astype(np.int32))
  mask = jnp.ones((length,), dtype=bool)
  noise_schedule = jnp.asarray(DEFAULT_NOISE_SCHEDULE)
  key = jax.random.PRNGKey(seed)

  @eqx.filter_jit
  def _run(m0: ProteinEBMModel, m1: ProteinEBMModel, c: jax.Array, k: jax.Array) -> jax.Array:
    return run_annealing_schedule(
      (m0, m1), (MODEL_SWAP_THRESHOLD,), c, aatype, mask, noise_schedule,
      DEFAULT_N_STEPS_PER_LEVEL, DEFAULT_DT, k,
    )

  jax.block_until_ready(_run(model, model, coords, key))  # untimed warmup
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    jax.block_until_ready(_run(model, model, coords, key))
    times.append(time.perf_counter() - start)
  return times


def _time_pytorch_outer_annealing(
  ref_model: "ProteinEBM", length: int, seed: int, n_repeats: int, device: "str | torch.device" = "cpu",
) -> list[float]:
  """Reimplementation of the reference's equivalent outer loop shape (eager threshold model-swap +
  per-level Euler-Maruyama steps), using the same real model for both swap branches."""
  import torch  # noqa: PLC0415

  rng = np.random.default_rng(seed)
  coords0 = rng.normal(scale=0.1, size=(length, 3)).astype(np.float32)
  aatype_np = rng.integers(0, 21, size=(length,)).astype(np.int64)

  def _one_round() -> None:
    coords = torch.tensor(coords0, dtype=torch.float32, device=device).unsqueeze(0)
    for t in DEFAULT_NOISE_SCHEDULE:
      model_for_t = ref_model  # eager "if t < MODEL_SWAP_THRESHOLD: model = other_model" -- same
      # real model both sides here (stand-in, see module docstring), so no branch needed to reach
      # the SAME numeric outcome as the JAX side's dispatcher; the point is timing the per-level
      # loop overhead, not a real second checkpoint's cost.
      for _ in range(DEFAULT_N_STEPS_PER_LEVEL):
        with torch.no_grad():
          feats = {
            "r_noisy": coords,
            "aatype": torch.tensor(aatype_np, dtype=torch.long, device=device).unsqueeze(0),
            "residue_idx": torch.arange(length, device=device).unsqueeze(0),
            "mask": torch.ones(1, length, dtype=torch.bool, device=device),
            "t": torch.tensor([t], dtype=torch.float32, device=device),
            "chain_encoding": torch.zeros(1, length, dtype=torch.long, device=device),
            "external_contacts": torch.zeros(1, length, dtype=torch.long, device=device),
          }
          out = model_for_t.compute_energy(feats, rescale_input_coords=False)
          aux_score = out["r_update_aux"]
          coords = coords + DEFAULT_DT * aux_score  # matches the aux-score-driven update shape

  _one_round()  # untimed warmup
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _one_round()
    times.append(time.perf_counter() - start)
  return times


def _time_jax_e10(model: ProteinEBMModel, length: int, seed: int, rounds: int, batch_size: int) -> float:
  """Time the full E10 run_structure_prediction pipeline once (single call, not repeated -- this is
  a multi-round pipeline, not a fast single-step primitive; one real timing is the meaningful unit)."""
  rng = np.random.default_rng(seed)
  coords = jnp.asarray(rng.normal(scale=0.1, size=(length, 3)).astype(np.float32))
  aatype = jnp.asarray(rng.integers(0, 21, size=(length,)).astype(np.int32))
  mask = jnp.ones((length,), dtype=bool)
  noise_schedule = jnp.asarray(DEFAULT_NOISE_SCHEDULE)
  key = jax.random.PRNGKey(seed)

  def _run() -> Any:  # noqa: ANN401
    return run_structure_prediction(
      (model, model), (MODEL_SWAP_THRESHOLD,), coords, aatype, mask, noise_schedule,
      DEFAULT_N_STEPS_PER_LEVEL, DEFAULT_DT, key,
      num_rounds=rounds, batch_size=batch_size,
    )

  jax.block_until_ready(_run())  # untimed warmup (also primes jit for every round's shape)
  start = time.perf_counter()
  jax.block_until_ready(_run())
  return time.perf_counter() - start


def _wall_clock_ms_stats(times_seconds: list[float]) -> tuple[float, float]:
  arr_ms = np.asarray(times_seconds) * 1000.0
  return float(np.mean(arr_ms)), float(np.std(arr_ms))


def _run_dry_run(args: argparse.Namespace) -> int:
  log.info("=== L1 dry-run ===")
  import torch  # noqa: PLC0415, F401

  if not args.reference_repo.exists():
    log.error("[FAIL] reference repo not found: %s", args.reference_repo)
    return 1
  model_cfg, diffuser_cfg = _smoke_model_configs()
  jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None)
  _time_jax_outer_annealing(jax_model, length=8, seed=args.seed, n_repeats=1)
  _time_pytorch_outer_annealing(ref_model, length=8, seed=args.seed, n_repeats=1)
  _time_jax_e10(jax_model, length=8, seed=args.seed, rounds=2, batch_size=2)
  log.info("[PASS] outer annealing (JAX+PyTorch) + E10 pipeline all executed once without error.")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"dry_run": True}, indent=2))
  return 0


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if args.dry_run:
    return _run_dry_run(args)

  lengths = (
    SMOKE_LENGTHS if args.smoke
    else (tuple(int(x) for x in args.lengths.split(",")) if args.lengths else DEFAULT_LENGTHS)
  )
  n_repeats = args.n_repeats if args.n_repeats is not None else (2 if args.smoke else 5)
  e10_rounds = min(args.e10_rounds, 2) if args.smoke else args.e10_rounds
  e10_batch = min(args.e10_batch_size, 2) if args.smoke else args.e10_batch_size

  jax_model, ref_model = build_models(args)
  torch_device = _torch_device()
  jax_device = jax.devices()[0].platform
  log.info("JAX device: %s | PyTorch device: %s", jax_device, torch_device)
  results = []
  for length in lengths:
    try:
      log.info("=== length=%d ===", length)
      jax_times = _time_jax_outer_annealing(jax_model, length, args.seed, n_repeats)
      jax_ms_mean, jax_ms_std = _wall_clock_ms_stats(jax_times)
      pt_times = _time_pytorch_outer_annealing(ref_model, length, args.seed, n_repeats, device=torch_device)
      pt_ms_mean, pt_ms_std = _wall_clock_ms_stats(pt_times)

      e10_ms = _time_jax_e10(jax_model, length, args.seed, e10_rounds, e10_batch) * 1000.0

      results.append({
        "protein_length": length,
        "jax_device": jax_device,
        "pytorch_device": torch_device,
        "outer_annealing_jax_ms_mean": jax_ms_mean,
        "outer_annealing_jax_ms_std": jax_ms_std,
        "outer_annealing_pytorch_ms_mean": pt_ms_mean,
        "outer_annealing_pytorch_ms_std": pt_ms_std,
        "outer_annealing_speedup": pt_ms_mean / jax_ms_mean if jax_ms_mean else float("nan"),
        "e10_pipeline_ms": e10_ms,
        "e10_rounds": e10_rounds,
        "e10_batch_size": e10_batch,
      })
      log.info(
        "[length=%d] outer annealing: jax=%.2fms pytorch=%.2fms speedup=%.2fx | e10 pipeline (%d rounds, batch=%d): %.2fms",
        length, jax_ms_mean, pt_ms_mean, pt_ms_mean / jax_ms_mean if jax_ms_mean else float("nan"),
        e10_rounds, e10_batch, e10_ms,
      )
    except Exception as e:  # noqa: BLE001 -- a single length must not lose already-collected results
      log.error("[length=%d] FAILED: %s: %s", length, type(e).__name__, e)
      results.append({"protein_length": length, "error": f"{type(e).__name__}: {e}"})

    _write_payload(args, lengths, n_repeats, e10_rounds, e10_batch, results)

  log.info("Wrote %d result rows to %s", len(results), args.out)
  return 0


def _write_payload(
  args: argparse.Namespace,
  lengths: tuple[int, ...],
  n_repeats: int,
  e10_rounds: int,
  e10_batch: int,
  results: list[dict],
) -> None:
  """Write accumulated results so far -- called after every length so a crash mid-sweep

  (e.g. the documented Blackwell/SM120 XLA-autotuning CUDA_ERROR_ILLEGAL_ADDRESS fault,
  `.praxia/docs/audits/260716_proteinebm-parity-report.md` §7) loses only the in-flight
  length, not every length already timed.
  """
  payload = {
    "meta": {
      "smoke": args.smoke,
      "lengths": list(lengths),
      "noise_schedule": list(DEFAULT_NOISE_SCHEDULE),
      "n_steps_per_level": DEFAULT_N_STEPS_PER_LEVEL,
      "model_swap_threshold": MODEL_SWAP_THRESHOLD,
      "n_repeats": n_repeats,
      "e10_rounds": e10_rounds,
      "e10_batch_size": e10_batch,
      "methodology_notes": [
        "Closes a documented gap: langevin_benchmark.py (E11d) only benchmarks "
        "the INNER fixed-t sampler; this script is the first throughput "
        "measurement for the OUTER noise-schedule+model-swap loop "
        "(run_annealing_schedule) and the full E10 multi-round pipeline "
        "(run_structure_prediction).",
        "Model-swap stand-in: the SAME real checkpoint is used for both swap "
        "branches (no second real checkpoint available in this environment) -- "
        "measures the lax.cond/lax.switch dispatch overhead + carrying two "
        "live model pytrees, not a real cross-checkpoint cost difference. "
        "Matches the epic's own already-documented E9 test stand-in convention.",
        "E10 pipeline timed once per length (not repeated) -- a multi-round "
        "pipeline call is not a fast primitive suited to a tight repeat loop; "
        "one real timing is the meaningful unit here.",
        "No SM120 XLA autotuning workaround applied here -- CPU-only local "
        "run. A cluster (GPU) run MUST set "
        "XLA_FLAGS=--xla_gpu_shard_autotuning=false per ~/.claude/rules/CLUSTER.md.",
        "A row with an 'error' key (no timing fields) means that protein_length "
        "raised during timing -- all other rows in this file completed normally "
        "and are unaffected.",
      ],
    },
    "results": results,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
  raise SystemExit(main())
