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

Each strategy runs in its own OS subprocess (re-invoking this same script with `--strategy`), not
just in-process sequentially. On Blackwell (SM120), the JAX strategy hits a confirmed, version-
independent XLA:Triton compiler bug (CUDA_ERROR_ILLEGAL_ADDRESS -- see
`.praxia/docs/audits/260716_proteinebm-parity-report.md` sec. 7) that corrupts the CUDA driver
context; once PyTorch shares the same GPU, that corruption previously poisoned both PyTorch
strategies too when everything ran in one process. Separate processes give each strategy its own
CUDA context, so a JAX crash can no longer take the PyTorch numbers down with it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
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

STRATEGY_NAMES: tuple[str, ...] = (
  # jax_bucket_pad_tile runs LAST, not first: in production (jobs 18515871 and 18518766,
  # 2026-07-22, H200/node4009) a ~105GB compute-app entry showed up under this project's own
  # venv path before this script's own strategies had allocated anything meaningful, under a
  # DIFFERENT pid each time. node4009 also happened to be running a third-party job (aaru0302's
  # zinc_finetune_s1_h200_resume) at the time, but SLURM tracks GRES (GPU) allocation at the node
  # level across every partition it belongs to, so that alone shouldn't let two jobs land on the
  # same physical device -- that theory doesn't actually hold up and isn't the explanation. The
  # pragmatic fix regardless of the exact cause (this script's own earlier subprocess in the same
  # job not having released memory yet, driver-level teardown lag, or something else) is
  # `_wait_for_gpu_free` below: gate each strategy's subprocess on the GPU actually reporting
  # enough free memory before it starts touching anything, instead of assuming the moment a
  # subprocess launches its device is ready. Kept this ordering as cheap defense in depth on top
  # of that: whatever residual pressure the free-memory gate doesn't catch, the strategy most
  # likely to fail from it should never be positioned to starve the others that come after it.
  "pytorch_pad_to_batch_max", "pytorch_per_structure_loop", "jax_bucket_pad_tile",
)


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
  parser.add_argument(
    "--strategy", choices=STRATEGY_NAMES, default=None,
    help="Internal: run only this one strategy and write a single-result partial file. Used by "
    "the parent process to isolate each strategy in its own subprocess/CUDA context -- not "
    "meant to be passed by a human caller.",
  )
  parser.add_argument(
    "--compilation-cache-dir", type=Path, default=None,
    help="If set, enables JAX's persistent compilation cache at this directory for the "
    "jax_bucket_pad_tile strategy only (mirrors src/aminx/host/prep.py's production "
    "jax.config.update('jax_compilation_cache_dir', ...) wiring exactly). Each distinct bucket "
    "size (64/128/256/512) present in the batch is its own JIT compile -- passing the SAME "
    "directory across two separate invocations lets the second run load those compiles from "
    "disk instead of recompiling, demonstrating cold-vs-warm compile cost across the bucket "
    "boundaries. Cache state (cold/warm, based on whether the directory already had entries "
    "before this run) is recorded in the output for provenance.",
  )
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


def _log_gpu_processes(label: str) -> None:
  """Diagnostic: dump nvidia-smi's own compute-process list, plus our own pid, so a large memory
  figure can be attributed to THIS process vs. some other one still holding the GPU. Added
  2026-07-22 after jobs 18515871/18518766 left this genuinely ambiguous: nvidia-smi listed a
  ~105GB compute-app entry under this project's own venv path in both runs, under a DIFFERENT
  PID each time -- most likely this script's own JAX preallocation from an earlier subprocess in
  the same job not having been released by the time the next subprocess started (SLURM tracks
  GPU allocation at the node level across partitions, so a same-node third-party job isn't
  expected to explain this). See `_wait_for_gpu_free`, which gates on the GPU actually being
  free rather than relying on this diagnostic alone. Logging our own pid lets a future run
  confirm directly: if the elevated entry's pid never matches ours, something else is holding it.
  Safe no-op if nvidia-smi is absent (e.g. CPU-only local dev) or the call fails for any reason
  -- this must never break a run."""
  try:
    out = subprocess.run(  # noqa: S603, S607 -- fixed argv, no shell, diagnostic-only
      ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv"],
      capture_output=True, text=True, timeout=10, check=False,
    )
    log.info("[gpu-processes %s] (our pid=%d)\n%s", label, os.getpid(), out.stdout.strip() or "(no output)")
  except Exception as e:  # noqa: BLE001 -- diagnostic only, never fatal
    log.info("[gpu-processes %s] nvidia-smi unavailable: %s", label, e)


# Per-strategy minimum free GPU memory (GiB) required before that strategy's subprocess is
# allowed to start building its model, derived from the actual worst-case allocation each one
# has been observed to attempt at production scale (n=256, see BATCH_SIZES_LARGE_L-style ceiling
# in run_cluster_benchmarks.sh for the analogous per-length convention elsewhere in this repo):
# pytorch_pad_to_batch_max pads the WHOLE batch to its single largest length and tried to
# allocate 128GiB doing so (job 18515871's OOM); jax_bucket_pad_tile's own preallocation plus its
# largest bucket's autotuning scratch needed at least 104.85+36.02=~141GB in the worst observed
# case, so it gets the highest bar; pytorch_per_structure_loop never pads and has consistently
# needed only a few GB. These are heuristics tied to the current n_structures=256 production
# config, not a formula derived from --n-structures -- revisit if that default changes materially.
_MIN_FREE_GB_BY_STRATEGY: dict[str, float] = {
  "pytorch_pad_to_batch_max": 130.0,
  "pytorch_per_structure_loop": 10.0,
  "jax_bucket_pad_tile": 110.0,
}


def _wait_for_gpu_free(min_free_gb: float, timeout_s: float = 600.0, poll_interval_s: float = 15.0) -> None:
  """Block until the GPU reports at least ``min_free_gb`` free, or ``timeout_s`` elapses.

  Added 2026-07-22 after jobs 18515871/18518766 both saw a large, unexplained chunk of GPU
  memory already in use the moment a strategy's subprocess started (see _log_gpu_processes) --
  most likely this script's own earlier subprocess in the same job not having released its
  memory yet by the time the next one launched. Rather than trying to fully pin down why the GPU
  isn't free, just wait until it actually is: this is a strict superset fix that also covers any
  other transient reason the device might not be ready yet. Safe no-op if nvidia-smi is absent
  (e.g. CPU-only local dev) -- proceeds immediately rather than blocking forever. If the timeout
  elapses without enough free memory, logs a warning and proceeds anyway -- the retry-with-backoff
  loop in ``_run_one_strategy`` is the last line of defense if this optimistic proceed still OOMs.
  """
  deadline = time.monotonic() + timeout_s
  free_gb = None
  while True:
    try:
      out = subprocess.run(  # noqa: S603, S607 -- fixed argv, no shell, diagnostic-only
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
      )
      free_gb = float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception as e:  # noqa: BLE001 -- diagnostic-only gate, never block a run on this failing
      log.info("[gpu-wait] nvidia-smi unavailable, proceeding without gating: %s", e)
      return
    if free_gb >= min_free_gb:
      log.info("[gpu-wait] %.1fGiB free (>= %.1fGiB required), proceeding", free_gb, min_free_gb)
      return
    if time.monotonic() >= deadline:
      log.warning(
        "[gpu-wait] timed out after %.0fs waiting for %.1fGiB free (last saw %.1fGiB) -- "
        "proceeding anyway", timeout_s, min_free_gb, free_gb,
      )
      return
    log.info("[gpu-wait] only %.1fGiB free (need %.1fGiB), waiting %.0fs...", free_gb, min_free_gb, poll_interval_s)
    time.sleep(poll_interval_s)


def _load_model_configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
  """Load ``(model_cfg, diffuser_cfg, state_dict)`` from ``--checkpoint``, or smoke configs."""
  if args.smoke:
    model_cfg, diffuser_cfg = _smoke_model_configs()
    return model_cfg, diffuser_cfg, None

  import torch  # noqa: PLC0415

  if not args.checkpoint.exists():
    msg = f"checkpoint not found: {args.checkpoint}"
    raise FileNotFoundError(msg)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model_cfg = ckpt["hyper_parameters"]["config"].model
  diffuser_cfg = ckpt["hyper_parameters"]["config"].diffuser
  return model_cfg, diffuser_cfg, ckpt["state_dict"]


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
) -> tuple[list[float], int, int, float]:
  """Strategy 1: group by bucket, pad per-group, one vmap'd energy call per bucket.

  Returns ``(times, padded_elements, real_elements, compile_wall_clock_s)`` -- the 4th element
  is the warmup call's wall-clock time, i.e. one JIT compile per distinct bucket size present in
  the batch (previously discarded; now surfaced so cold-vs-warm persistent-compilation-cache
  runs can be compared directly, see ``--compilation-cache-dir``).
  """
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

  compile_start = time.perf_counter()
  _call()  # warmup -- forces jit compilation per distinct bucket shape; timed separately below
  compile_wall_clock_s = time.perf_counter() - compile_start
  times = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    _call()
    times.append(time.perf_counter() - start)
  return times, padded_elements, real_elements, compile_wall_clock_s


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


def _run_one_strategy(name: str, args: argparse.Namespace) -> dict[str, Any]:
  """Build only the model ``name`` actually needs and time it. Runs inside a single-strategy subprocess.

  Building both frameworks' models regardless of ``name`` (the pre-isolation behavior) meant every
  subprocess touched JAX's GPU backend even to run a pure-PyTorch strategy -- JAX preallocates
  ~75% of device memory the moment it first touches the device (e.g. ``jax.random.PRNGKey`` inside
  ``_build_jax_model``), starving PyTorch's `model.to(device)` in the same process. Only constructing
  the model the requested strategy uses keeps each subprocess's footprint to what it actually needs.
  """
  n_structures = min(args.n_structures, 6) if args.smoke else args.n_structures
  n_repeats = min(args.n_repeats, 2) if args.smoke else args.n_repeats
  structures = _build_heterogeneous_batch(n_structures, args.seed, max_length=BUCKET_BOUNDARIES[-1])
  model_cfg, diffuser_cfg, state_dict = _load_model_configs(args)
  device_label = jax.devices()[0].platform if name == "jax_bucket_pad_tile" else _torch_device()

  cache_state: str | None = None
  cache_dir = getattr(args, "compilation_cache_dir", None)
  if name == "jax_bucket_pad_tile" and cache_dir is not None:
    cache_state = "warm" if cache_dir.exists() and any(cache_dir.iterdir()) else "cold"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))  # mirrors src/aminx/host/prep.py
    # JAX's default 1.0s min-compile-time threshold means fast compiles (small/smoke-scale
    # models, or genuinely quick real ones) silently never get persisted -- 0 guarantees this
    # demo actually reflects cache_state rather than flaking depending on model/scale.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

  _log_gpu_processes(f"before {name}")
  _wait_for_gpu_free(_MIN_FREE_GB_BY_STRATEGY[name])

  def _build_and_run() -> tuple[list[float], int, int, float | None]:
    if name == "jax_bucket_pad_tile":
      jax_model = _build_jax_model(model_cfg, args.seed, state_dict)
      return _jax_bucket_pad_tile(jax_model, structures, args.diffusion_time, n_repeats)
    ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict, device=device_label)
    strategy_fn = _pytorch_pad_to_batch_max if name == "pytorch_pad_to_batch_max" else _pytorch_per_structure_loop
    times, padded_elements, real_elements = strategy_fn(ref_model, structures, args.diffusion_time, n_repeats, device=device_label)
    return times, padded_elements, real_elements, None  # eager PyTorch has no separate compile phase

  # Originally this comment attributed a persistently-elevated nvidia-smi reading (~105GB in use
  # by a dead PID) to node4009 being double-booked across the `pi_so3`/`mit_preemptable`
  # partitions. CORRECTED 2026-07-22: SLURM tracks GPU allocation at the node level across every
  # partition it belongs to, so that theory doesn't actually hold -- a same-node third-party job
  # shouldn't be able to land on the same physical GPU. Most likely explanation is this script's
  # own earlier subprocess in the same job not having released its memory by the time the next
  # one started. Rather than keep chasing the exact mechanism, `_wait_for_gpu_free` above now
  # gates each strategy on the GPU actually reporting enough free memory before touching
  # anything, which helps regardless of cause. The retry backoff below stays as a second layer
  # for whatever it doesn't catch (e.g. memory freed mid-attempt, or a genuinely transient
  # driver-level race like the documented Blackwell/SM120 ghost-listing behavior).
  max_attempts = 6
  for attempt in range(1, max_attempts + 1):
    try:
      times, padded_elements, real_elements, compile_wall_clock_s = _build_and_run()
      break
    except Exception as e:  # noqa: BLE001 -- must still report a result row, not crash the subprocess silently
      is_oom = "out of memory" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e)
      if is_oom:
        _log_gpu_processes(f"after {name} attempt {attempt} OOM")
      if is_oom and attempt < max_attempts:
        wait_s = 20 * attempt
        log.warning(
          "[%s] attempt %d/%d hit an OOM-shaped error, retrying in %ds (likely a still-draining "
          "prior subprocess's CUDA context, not real capacity): %s",
          name, attempt, max_attempts, wait_s, e,
        )
        time.sleep(wait_s)
        continue
      log.error("[%s] FAILED (attempt %d/%d): %s: %s", name, attempt, max_attempts, type(e).__name__, e)
      return {"strategy": name, "error": f"{type(e).__name__}: {e}"}

  mean_ms = float(np.mean(times)) * 1000.0
  std_ms = float(np.std(times)) * 1000.0
  padding_overhead = padded_elements / real_elements if real_elements else float("nan")
  compile_ms = compile_wall_clock_s * 1000.0 if compile_wall_clock_s is not None else None
  log.info(
    "[%s] wall_clock=%.2fms (+/-%.2fms) padding_overhead=%.3fx (%d padded / %d real elements)%s",
    name, mean_ms, std_ms, padding_overhead, padded_elements, real_elements,
    f" compile={compile_ms:.2f}ms cache_state={cache_state}" if compile_ms is not None else "",
  )
  return {
    "strategy": name,
    "device": device_label,
    "wall_clock_mean_ms": mean_ms,
    "wall_clock_std_ms": std_ms,
    "padded_elements": padded_elements,
    "real_elements": real_elements,
    "padding_overhead_ratio": padding_overhead,
    "compile_wall_clock_ms": compile_ms,
    "compilation_cache_dir": str(cache_dir) if cache_dir is not None else None,
    "compilation_cache_state": cache_state,
  }


def _run_strategy_subprocess_mode(args: argparse.Namespace) -> int:
  """Child-process entry point: run exactly one strategy, write its result alone to ``--out``."""
  result = _run_one_strategy(args.strategy, args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"result": result}, indent=2))
  return 0 if "error" not in result else 1


def _child_argv(args: argparse.Namespace, strategy: str, out_path: Path) -> list[str]:
  argv = [
    sys.executable, str(Path(__file__).resolve()),
    "--strategy", strategy,
    "--n-structures", str(args.n_structures),
    "--checkpoint", str(args.checkpoint),
    "--reference-repo", str(args.reference_repo),
    "--seed", str(args.seed),
    "--diffusion-time", str(args.diffusion_time),
    "--n-repeats", str(args.n_repeats),
    "--out", str(out_path),
  ]
  if args.smoke:
    argv.append("--smoke")
  if args.compilation_cache_dir is not None:
    argv.extend(["--compilation-cache-dir", str(args.compilation_cache_dir)])
  return argv


def _child_env(strategy: str) -> dict[str, str]:
  """Env for a strategy's subprocess. Belt-and-suspenders on top of only building the needed
  model: the pytorch strategies never need JAX to touch the GPU at all, so tell it not to
  preallocate (JAX's default ~75% upfront grab would otherwise starve PyTorch in that process
  if anything ever imports/touches JAX there again). The JAX strategy keeps preallocation ON
  (disabling it entirely would slow its own allocator and skew the very throughput being
  measured) but at a REDUCED fraction: confirmed in production (job 18515871, 2026-07-22) that
  the default 0.75 fraction leaves too little headroom for XLA's autotuner, which profiles
  candidate GEMM kernels for the largest bucket group (512) by allocating scratch buffers on top
  of the model's own live activations -- one profiling attempt there needed an extra 36GB, but
  the default fraction only left ~25% (~35GB of a 139.8GB H200) free, so autotuning OOM'd on
  every one of its 6 retries (retrying doesn't help: it's the same process reusing the same
  static arena each time). 0.5 leaves ~70GB of headroom instead, still with preallocation on.
  """
  env = dict(os.environ)
  if strategy == "jax_bucket_pad_tile":
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  else:
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  return env


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if args.dry_run:
    return _run_dry_run(args)
  if args.strategy is not None:
    return _run_strategy_subprocess_mode(args)

  n_structures = min(args.n_structures, 6) if args.smoke else args.n_structures
  n_repeats = min(args.n_repeats, 2) if args.smoke else args.n_repeats
  structures = _build_heterogeneous_batch(n_structures, args.seed, max_length=BUCKET_BOUNDARIES[-1])
  lengths = [s["length"] for s in structures]
  log.info("Heterogeneous batch: n=%d lengths=%s", n_structures, sorted(lengths))

  results: list[dict[str, Any]] = []
  for name in STRATEGY_NAMES:
    partial_path = args.out.with_suffix(f".{name}.partial.json")
    log.info("=== launching isolated subprocess: %s ===", name)
    proc = subprocess.run(_child_argv(args, name, partial_path), env=_child_env(name), check=False)  # noqa: S603 -- fixed argv, no shell
    if partial_path.exists():
      result = json.loads(partial_path.read_text())["result"]
      partial_path.unlink()
    else:
      log.error("[%s] subprocess exited %d with no output file -- recording as crashed", name, proc.returncode)
      result = {"strategy": name, "error": f"subprocess exit code {proc.returncode}, no output written"}
    results.append(result)
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
        "Each strategy runs in its own OS subprocess (this script re-invoked with "
        "--strategy), not just sequentially in-process: on Blackwell (SM120), the "
        "JAX strategy can hit a confirmed XLA:Triton compiler bug that corrupts "
        "the CUDA driver context, which previously also poisoned both PyTorch "
        "strategies once they shared the same GPU (see "
        ".praxia/docs/audits/260716_proteinebm-parity-report.md sec. 7). Separate "
        "processes mean a JAX crash can no longer take the PyTorch numbers with it.",
      ],
    },
    "results": results,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
  raise SystemExit(main())
