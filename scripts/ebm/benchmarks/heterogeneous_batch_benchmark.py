"""Heterogeneous-length batch throughput: JAX bucket+pad+tile vs. PyTorch dynamic-shape strategies.

New benchmark, not a rerun of anything else in this repo. Every E11a-d benchmark to date (decoy/
ddg/biasing/langevin) scores a batch of **one fixed protein length** at a time, at exact bucket
boundaries -- zero padding waste by construction. Real production workloads mix protein lengths in
one batch (the reference's own `score_decoys.py` avoids this by processing one native structure's
decoy set at a time, never mixing lengths within a `DataLoader` batch -- see that script's own
per-`d` loop). This script measures what actually happens when lengths ARE mixed, comparing three
strategies on the identical heterogeneous input:

1. **JAX bucket+pad+tile** (the strategy `aminx.ebm`'s xtrax `Bucket` composition is built for):
   group structures by `xtrax.tiling.select_bucket` bucket assignment (the E4.5-confirmed
   `{64,128,256,512}` boundaries), pad each group to its bucket size, one `jax.vmap`'d
   `ProteinEBMModel.energy` call per bucket group.
2. **PyTorch pad-to-batch-max**: pad every structure in the WHOLE batch to the single largest
   length present, one dense batched `compute_energy` call -- the "naive dynamic-batching" approach
   a PyTorch user would reach for first, with no bucketing concept.
3. **PyTorch per-structure loop**: no padding at all, one `compute_energy` call per structure at its
   own native length -- zero padding waste, zero batching benefit (closest to what `score_decoys.py`
   itself actually does, one native length at a time).

Metrics: total wall-clock for the whole heterogeneous batch per strategy, plus a measured (not
statistically estimated, unlike the E4.5 `bucket_boundary_check.py` padding-waste analysis) padding
overhead: total padded elements processed / total real (unpadded) elements.

Reuses `build_proxy_distribution` from `scripts/ebm/bucket_boundary_check.py` (E4.5's own documented
proxy length distribution: real local PDB/mmCIF lengths + a documented log-normal mixture) for a
realistic mixed-length composition -- no new length-distribution logic.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from bucket_boundary_check import build_proxy_distribution  # noqa: E402

from aminx.ebm.checkpoint import load_pytorch_checkpoint  # noqa: E402
from aminx.ebm.model import ProteinEBMModel  # noqa: E402

if TYPE_CHECKING:
  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

log = logging.getLogger("heterogeneous_batch_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()
DEFAULT_DIFFUSION_TIME = 0.05
BUCKET_BOUNDARIES: tuple[int, ...] = (64, 128, 256, 512)  # E4.5-confirmed boundaries.

SMOKE_TOKEN_S = 32
SMOKE_TOKEN_Z = 16
SMOKE_DIM_FOURIER = 12
SMOKE_TRANSITION_LAYERS = 1
SMOKE_DEPTH = 2
SMOKE_HEADS = 2
SMOKE_NUM_CONTACT_EMBEDDINGS = 2


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--n-structures", type=int, default=32, help="Number of structures in the heterogeneous batch (default 32, or 6 under --smoke).")
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--diffusion-time", type=float, default=DEFAULT_DIFFUSION_TIME)
  parser.add_argument("--n-repeats", type=int, default=5)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--smoke", action="store_true", help="Tiny synthetic-dim models, small batch, <60s on CPU.")
  parser.add_argument("--dry-run", action="store_true", help="L1 gate: imports + one trivial call, no timed loop.")
  return parser.parse_args()


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
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, ckpt["state_dict"], device=device)
  jax_model = _build_jax_model(model_cfg, args.seed, ckpt["state_dict"])
  return jax_model, ref_model


def _build_heterogeneous_batch(n_structures: int, seed: int, max_length: int) -> list[dict[str, np.ndarray]]:
  """Build ``n_structures`` synthetic structures at real/plausible, DISTINCT lengths.

  Lengths come from ``build_proxy_distribution`` (E4.5's real+synthetic proxy, reused verbatim, not
  re-derived), clipped to ``[1, max_length]`` since ``select_bucket`` errors above the largest
  boundary. Each structure gets its own synthetic coords/aatype (i.i.d. Gaussian / random sequence,
  same convention as every other E11x script's ``_synthetic_decoys``) -- distinct sequences, unlike
  the same-sequence-different-coords contract ``score_decoy_batch`` assumes (irrelevant here: this
  script never calls ``score_decoy_batch``, only ``ProteinEBMModel.energy`` directly, which has no
  such same-sequence constraint).
  """
  dist = build_proxy_distribution(n_synthetic=n_structures, seed=seed)
  all_lengths = dist.real_lengths + list(dist.synthetic_lengths)
  rng = np.random.default_rng(seed)
  chosen = rng.choice(all_lengths, size=n_structures, replace=len(all_lengths) < n_structures)
  chosen = np.clip(chosen, 1, max_length).astype(int)

  structures = []
  for i, length in enumerate(chosen):
    struct_rng = np.random.default_rng(seed + 1000 + i)
    structures.append({
      "length": int(length),
      "coords": struct_rng.normal(scale=1.0, size=(length, 3)).astype(np.float32),
      "aatype": struct_rng.integers(0, 21, size=(length,)).astype(np.int64),
      "residue_index": np.arange(length, dtype=np.int64),
      "mask": np.ones((length,), dtype=bool),
    })
  return structures


def _pad_to(struct: dict[str, np.ndarray], target_len: int) -> dict[str, np.ndarray]:
  n = struct["length"]
  pad = target_len - n
  return {
    "coords": np.pad(struct["coords"], ((0, pad), (0, 0))),
    "aatype": np.pad(struct["aatype"], (0, pad)),
    "mask": np.pad(struct["mask"], (0, pad), constant_values=False),
  }


def _jax_bucket_pad_tile(
  model: ProteinEBMModel, structures: list[dict[str, np.ndarray]], t: float, n_repeats: int,
) -> tuple[list[float], int, int]:
  """Strategy 1: group by bucket, pad per-group, one vmap'd energy call per bucket."""
  from xtrax.tiling import select_bucket  # noqa: PLC0415

  groups: dict[int, list[dict[str, np.ndarray]]] = {}
  for s in structures:
    bucket = select_bucket(s["length"], BUCKET_BOUNDARIES)
    groups.setdefault(bucket, []).append(s)

  padded_elements = sum(bucket * len(members) for bucket, members in groups.items())
  real_elements = sum(s["length"] for s in structures)

  batched_inputs = []
  for bucket, members in groups.items():
    padded = [_pad_to(s, bucket) for s in members]
    coords = jnp.asarray(np.stack([p["coords"] for p in padded]))
    aatype = jnp.asarray(np.stack([p["aatype"] for p in padded]), dtype=jnp.int32)
    mask = jnp.asarray(np.stack([p["mask"] for p in padded]))
    batched_inputs.append((coords, aatype, mask))

  t_arr = jnp.asarray(t)

  @eqx.filter_jit
  def _score_group(m: ProteinEBMModel, coords: jax.Array, aatype: jax.Array, mask: jax.Array) -> jax.Array:
    return jax.vmap(lambda c, a, mk: m.energy(c, a, t_arr, mk))(coords, aatype, mask)

  def _call() -> None:
    for coords, aatype, mask in batched_inputs:
      jax.block_until_ready(_score_group(model, coords, aatype, mask))

  _call()  # untimed warmup (forces jit compilation per distinct bucket shape)
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _call()
    times.append(time.perf_counter() - start)
  return times, padded_elements, real_elements


def _pytorch_pad_to_batch_max(
  ref_model: "ProteinEBM", structures: list[dict[str, np.ndarray]], t: float, n_repeats: int, device: "str | torch.device" = "cpu",
) -> tuple[list[float], int, int]:
  """Strategy 2: pad every structure to the single batch-max length, one dense forward."""
  import torch  # noqa: PLC0415

  max_len = max(s["length"] for s in structures)
  padded = [_pad_to(s, max_len) for s in structures]
  n = len(structures)
  r_noisy = torch.tensor(np.stack([p["coords"] for p in padded]), dtype=torch.float32, device=device)
  aatype = torch.tensor(np.stack([p["aatype"] for p in padded]), dtype=torch.long, device=device)
  mask = torch.tensor(np.stack([p["mask"] for p in padded]), dtype=torch.bool, device=device)
  residue_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(n, max_len).clone()
  chain_id = torch.zeros(n, max_len, dtype=torch.long, device=device)
  contacts = torch.zeros(n, max_len, dtype=torch.long, device=device)
  times_t = torch.full((n,), t, dtype=torch.float32, device=device)
  feats = {
    "r_noisy": r_noisy, "aatype": aatype, "residue_idx": residue_idx, "mask": mask,
    "t": times_t, "chain_encoding": chain_id, "external_contacts": contacts,
  }

  def _call() -> None:
    with torch.no_grad():
      ref_model.compute_energy(feats, rescale_input_coords=False)

  _call()
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _call()
    times.append(time.perf_counter() - start)

  padded_elements = max_len * n
  real_elements = sum(s["length"] for s in structures)
  return times, padded_elements, real_elements


def _pytorch_per_structure_loop(
  ref_model: "ProteinEBM", structures: list[dict[str, np.ndarray]], t: float, n_repeats: int, device: "str | torch.device" = "cpu",
) -> tuple[list[float], int, int]:
  """Strategy 3: no padding, one forward call per structure at its native length."""
  import torch  # noqa: PLC0415

  built = []
  for s in structures:
    n = s["length"]
    built.append({
      "r_noisy": torch.tensor(s["coords"], dtype=torch.float32, device=device).unsqueeze(0),
      "aatype": torch.tensor(s["aatype"], dtype=torch.long, device=device).unsqueeze(0),
      "residue_idx": torch.arange(n, device=device).unsqueeze(0),
      "mask": torch.tensor(s["mask"], dtype=torch.bool, device=device).unsqueeze(0),
      "t": torch.tensor([t], dtype=torch.float32, device=device),
      "chain_encoding": torch.zeros(1, n, dtype=torch.long, device=device),
      "external_contacts": torch.zeros(1, n, dtype=torch.long, device=device),
    })

  def _call() -> None:
    with torch.no_grad():
      for feats in built:
        ref_model.compute_energy(feats, rescale_input_coords=False)

  _call()
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _call()
    times.append(time.perf_counter() - start)

  real_elements = sum(s["length"] for s in structures)
  return times, real_elements, real_elements  # zero padding waste by construction


def _run_dry_run(args: argparse.Namespace) -> int:
  log.info("=== L1 dry-run: import + construction + one-shot execution check ===")
  import torch  # noqa: PLC0415, F401

  if not args.reference_repo.exists():
    log.error("[FAIL] reference repo not found: %s", args.reference_repo)
    return 1
  model_cfg, diffuser_cfg = _smoke_model_configs()
  jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None)
  structures = _build_heterogeneous_batch(4, args.seed, max_length=64)
  for s in structures:
    s["length"] = min(s["length"], 8)
    s["coords"] = s["coords"][: s["length"]]
    s["aatype"] = s["aatype"][: s["length"]]
    s["mask"] = s["mask"][: s["length"]]
  _jax_bucket_pad_tile(jax_model, structures, args.diffusion_time, n_repeats=1)
  _pytorch_pad_to_batch_max(ref_model, structures, args.diffusion_time, n_repeats=1)
  _pytorch_per_structure_loop(ref_model, structures, args.diffusion_time, n_repeats=1)
  log.info("[PASS] all three strategies executed once without error.")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"dry_run": True}, indent=2))
  return 0


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if args.dry_run:
    return _run_dry_run(args)

  n_structures = min(args.n_structures, 6) if args.smoke else args.n_structures
  n_repeats = min(args.n_repeats, 2) if args.smoke else args.n_repeats

  jax_model, ref_model = build_models(args)
  torch_device = _torch_device()
  log.info("PyTorch device: %s", torch_device)
  structures = _build_heterogeneous_batch(n_structures, args.seed, max_length=BUCKET_BOUNDARIES[-1])
  lengths = [s["length"] for s in structures]
  log.info("Heterogeneous batch: n=%d lengths=%s", n_structures, sorted(lengths))

  strategy_calls: list[tuple[str, Any]] = [
    ("jax_bucket_pad_tile", lambda: _jax_bucket_pad_tile(jax_model, structures, args.diffusion_time, n_repeats)),
    ("pytorch_pad_to_batch_max", lambda: _pytorch_pad_to_batch_max(ref_model, structures, args.diffusion_time, n_repeats, device=torch_device)),
    ("pytorch_per_structure_loop", lambda: _pytorch_per_structure_loop(ref_model, structures, args.diffusion_time, n_repeats, device=torch_device)),
  ]

  results: list[dict[str, Any]] = []
  for name, call in strategy_calls:
    try:
      times, padded_elements, real_elements = call()
    except Exception as e:  # noqa: BLE001 -- one strategy crashing must not lose the others' already-timed results
      log.error("[%s] FAILED: %s: %s", name, type(e).__name__, e)
      results.append({"strategy": name, "error": f"{type(e).__name__}: {e}"})
      _write_payload(args, n_structures, n_repeats, lengths, results)
      continue
    mean_ms = float(np.mean(times)) * 1000.0
    std_ms = float(np.std(times)) * 1000.0
    padding_overhead = padded_elements / real_elements if real_elements else float("nan")
    results.append({
      "strategy": name,
      "device": torch_device if name.startswith("pytorch") else jax.devices()[0].platform,
      "wall_clock_mean_ms": mean_ms,
      "wall_clock_std_ms": std_ms,
      "padded_elements": padded_elements,
      "real_elements": real_elements,
      "padding_overhead_ratio": padding_overhead,
    })
    log.info(
      "[%s] wall_clock=%.2fms (+/-%.2fms) padding_overhead=%.3fx (%d padded / %d real elements)",
      name, mean_ms, std_ms, padding_overhead, padded_elements, real_elements,
    )
    _write_payload(args, n_structures, n_repeats, lengths, results)

  log.info("Wrote %d strategy results to %s", len(results), args.out)
  return 0


def _write_payload(
  args: argparse.Namespace,
  n_structures: int,
  n_repeats: int,
  lengths: list[Any],
  results: list[dict[str, Any]],
) -> None:
  """Write accumulated results so far -- called after every strategy attempt so a crash mid-run

  loses only the in-flight strategy, not every already-timed one.
  """
  payload = {
    "meta": {
      "smoke": args.smoke,
      "n_structures": n_structures,
      "n_repeats": n_repeats,
      "diffusion_time": args.diffusion_time,
      "seed": args.seed,
      "bucket_boundaries": list(BUCKET_BOUNDARIES),
      "lengths": sorted(int(l) for l in lengths),
      "methodology_notes": [
        "New comparison: no other E11x benchmark tests a batch of MIXED protein "
        "lengths -- every prior benchmark uses one fixed length per call, at "
        "exact bucket boundaries (zero padding waste by construction).",
        "jax_bucket_pad_tile groups structures by xtrax.tiling.select_bucket "
        "assignment (the same E4.5-confirmed {64,128,256,512} boundaries), "
        "pads each group to its bucket size, one jax.vmap'd energy call per "
        "group -- this is what aminx's own bucket+pad+tile composition does.",
        "pytorch_pad_to_batch_max pads every structure to the single largest "
        "length in the WHOLE batch (no bucketing concept) -- the naive "
        "dynamic-batching approach a PyTorch user would reach for first.",
        "pytorch_per_structure_loop has zero padding waste by construction "
        "(one call per structure at its native length) but zero batching "
        "benefit either -- closest to what score_decoys.py itself actually "
        "does (one native length at a time, never mixed).",
        "No SM120 XLA autotuning workaround applied here -- CPU-only local "
        "run. A cluster (GPU) run MUST set "
        "XLA_FLAGS=--xla_gpu_shard_autotuning=false per ~/.claude/rules/CLUSTER.md.",
        "A row with 'error' instead of 'wall_clock_mean_ms' means that strategy "
        "raised during timing -- the other strategies in this file completed "
        "normally and are unaffected.",
      ],
    },
    "results": results,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
  raise SystemExit(main())
