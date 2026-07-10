"""ProteinEBM decoy-ranking throughput/latency benchmark: JAX (ported) vs PyTorch (reference).

Backlog node **E11a** (``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md``
§2 E11a row `#3307`, §4.3 bathos benchmark spec; design spec
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §8.3). This
script is the throughput/latency half of the decoy-ranking application
(E5, ``aminx.ebm.decoy_ranking`` / ``aminx.ebm.dispatch.score_decoy_batch``) --
it does **not** compute or re-validate any Spearman correlation. Accuracy
parity is E5/E3.5's job; this script only measures speed.

**Apples-to-apples methodology (design spec §8.3, EPIC risk register
MINOR-2, read before trusting any number this script prints):**

1. **Bucket-aligned lengths.** Benchmarked at exactly the residue-axis bucket
   boundaries the EPIC locked at E4.5: ``{64, 128, 256, 512}``
   (:data:`DEFAULT_LENGTHS`). Coordinates are synthetic (i.i.d. Gaussian) --
   this is a throughput benchmark, not a correctness benchmark; correctness
   was already validated in E3.5 (``scripts/ebm/checkpoint_parity_check.py``)
   and E5 (``tests/ebm/test_decoy_ranking.py``).
2. **PyTorch baseline must not pay for an unnecessary retained autograd
   graph.** The reference's own ``ProteinEBM.compute_score`` always calls
   ``torch.autograd.grad(energy.sum(), r_noisy, create_graph=True)`` even at
   inference -- a correctness-irrelevant but performance-relevant detail
   (nested-differentiable graphs are more expensive to build/retain than a
   plain first-order backward). This script's PyTorch score/grad timing calls
   ``torch.autograd.grad(..., create_graph=False)`` directly (bypassing
   ``compute_score``'s wrapper, mirroring ``scripts/ebm/
   checkpoint_parity_check.py::_reference_energy_and_score``) so the JAX
   comparison isn't unfairly biased by this reference implementation detail.
   JAX has no ``create_graph`` flag to misconfigure -- ``jax.grad`` is
   first-order by construction unless nested.
3. **Exclude JIT/compile warmup.** For JAX, every timed metric is called once
   (untimed) before the timed-repeats loop, forcing ``jax.jit`` compilation
   for that shape; only steady-state (post-compile) calls are timed. For
   PyTorch there is no JIT step, but a matching untimed warmup call is made
   anyway for cache-warming parity (page faults, allocator warm-up, first-call
   BLAS dispatch).
4. **Metrics** (matches the ``[result_schema]`` in the sidecar
   ``decoy_benchmark.bth.toml``):
   - ``energy_evals_per_sec``: pure batched energy forward-pass throughput.
     JAX side batches via ``aminx.ebm.dispatch.score_decoy_batch`` (the same
     Vmap/SafeMap axis-dispatch wiring E5 uses); PyTorch side is a plain
     batched ``ProteinEBM.compute_energy`` forward (the reference model
     natively supports a leading batch dimension -- no vmap needed).
   - ``score_grad_ms``: single-structure conservative-score latency
     (``-grad(E)``, the 1st-order gradient cost). E9's Langevin sampler
     (2nd-order training-adjacent cost) is out of scope for this epic per the
     EPIC DAG's Fork 2/E9 resolution -- this metric is 1st-order only.
5. Every per-(length, impl) row is written to a structured JSON file
   (``--out``), not just logged -- see :data:`_write_report`.

**What this script does NOT do (read before citing any number):**

- It does not apply the mandatory SM120 ``XLA_FLAGS=--xla_gpu_shard_
  autotuning=false`` cluster workaround (``~/.claude/rules/CLUSTER.md``) --
  that flag only matters on the actual `pi_so3` GPU nodes this script never
  touches. A future cluster run of this exact script MUST set that flag or
  its JAX throughput numbers will be ~1000x wrong (per that rule's own
  measured example).
- It does not run on GPU. This environment has no GPU (confirmed: JAX and
  PyTorch both fall back to CPU here). Every number this script has actually
  produced (see the L1/L2 gate report) is a CPU number, reported honestly as
  such via the ``device`` column -- it is NOT the real JAX-vs-PyTorch
  hardware comparison the design spec's target is about
  (``"JAX >= PyTorch throughput on pi_so3 (SM120) at L in {64,128,256,512}"``).
  That comparison is a follow-on cluster (L3) run, out of scope for this
  dispatch.
- ``--smoke`` mode does NOT exercise the real E3.5-ported checkpoint weights
  (430MB, would dominate a smoke-test's runtime budget with I/O + strict
  85M-param ``load_state_dict``/``load_pytorch_checkpoint`` cost, not with
  anything this script's own logic is responsible for). It builds tiny,
  randomly-initialized models of the *same* architecture (both the real
  PyTorch ``protein_ebm.model.ebm.ProteinEBM`` class and the real
  ``aminx.ebm.model.ProteinEBMModel`` class) so the harness's own dispatch/
  timing/JSON-schema logic is exercised end-to-end fast (<60s on CPU) -- see
  the module's ``--smoke`` / L2-gate report for what this validates and does
  not validate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.ebm.checkpoint import load_pytorch_checkpoint
from aminx.ebm.dispatch import score_decoy_batch
from aminx.ebm.model import ProteinEBMModel

if TYPE_CHECKING:
  from collections.abc import Mapping

  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

log = logging.getLogger("decoy_benchmark")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()

# Residue-axis bucket boundaries locked at the E4.5 xtrax HiTL gate (EPIC DAG
# §1 Fork 6, §2 E4.5 row) -- the design spec's own §8.3 sweep points.
DEFAULT_LENGTHS: tuple[int, ...] = (64, 128, 256, 512)
SMOKE_LENGTHS: tuple[int, ...] = (64, 128)

# --- --smoke-mode architecture (tiny, randomly-initialized; NOT the real
# checkpoint -- see module docstring). Mirrors the tiny-dim fixtures already
# used by tests/ebm/test_dispatch.py and tests/ebm/test_decoy_ranking.py's
# synthetic-model tier. ---
SMOKE_TOKEN_S = 32
SMOKE_TOKEN_Z = 16
SMOKE_DIM_FOURIER = 12
SMOKE_TRANSITION_LAYERS = 1
SMOKE_DEPTH = 2
SMOKE_HEADS = 2
SMOKE_NUM_CONTACT_EMBEDDINGS = 2


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--lengths",
    type=str,
    default=None,
    help="Comma-separated residue counts, e.g. '64,128,256,512'. Defaults to "
    "the bucket points (DEFAULT_LENGTHS), or SMOKE_LENGTHS under --smoke.",
  )
  parser.add_argument(
    "--n-decoys",
    type=int,
    default=None,
    help="Decoy batch size for the energy_evals_per_sec metric (default 16, "
    "or 4 under --smoke).",
  )
  parser.add_argument(
    "--n-repeats",
    type=int,
    default=None,
    help="Timed repeats per (length, metric), after the untimed warmup call "
    "(default 5, or 2 under --smoke).",
  )
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--diffusion-time",
    type=float,
    default=0.05,
    help="Fixed VP-SDE diffusion time t (ProteinEBM-x MVP target, design spec §9).",
  )
  parser.add_argument("--out", type=Path, required=True, help="JSON output path.")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Reduced-scale, real-architecture-but-untrained local smoke run (L2 "
    "gate): tiny synthetic dims, no checkpoint load, <60s on CPU. See module docstring.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="L1 gate: import + path + one-shot construct/execute check on a "
    "trivial tiny model. No timed loop, no dependence on the real checkpoint "
    "being present.",
  )
  args = parser.parse_args()
  if args.n_repeats is not None and args.n_repeats < 1:
    parser.error("--n-repeats must be >= 1")
  if args.n_decoys is not None and args.n_decoys < 1:
    parser.error("--n-decoys must be >= 1")
  return args


def _resolve_run_params(args: argparse.Namespace) -> tuple[tuple[int, ...], int, int]:
  """Resolve (lengths, n_decoys, n_repeats), applying --smoke's reduced defaults."""
  if args.lengths is not None:
    lengths = tuple(int(x) for x in args.lengths.split(","))
  else:
    lengths = SMOKE_LENGTHS if args.smoke else DEFAULT_LENGTHS

  if args.smoke:
    n_decoys = args.n_decoys if args.n_decoys is not None else 4
    n_repeats = args.n_repeats if args.n_repeats is not None else 2
  else:
    n_decoys = args.n_decoys if args.n_decoys is not None else 16
    n_repeats = args.n_repeats if args.n_repeats is not None else 5
  return lengths, n_decoys, n_repeats


# ---------------------------------------------------------------------------
# Model construction -- both the real PyTorch reference class and the real
# aminx.ebm.model.ProteinEBMModel class, either from the real E3.5 checkpoint
# (full run) or from tiny random init (--smoke / --dry-run). See module
# docstring point re: why --smoke skips the checkpoint entirely.
# ---------------------------------------------------------------------------


def _smoke_model_configs() -> tuple[SimpleNamespace, SimpleNamespace]:
  """Minimal (model_cfg, diffuser_cfg) namespaces for the tiny --smoke architecture.

  ``protein_ebm.model.ebm.ProteinEBM.__init__``/``forward`` read every field
  via ``getattr(config, name, default)`` (duck-typed ``ml_collections.
  ConfigDict`` in the reference; a plain ``SimpleNamespace`` works identically
  here). ``aux_score`` MUST stay ``True``: ``ProteinEBM.forward`` calls
  ``self.r_update_proj_aux(a)`` **unconditionally** (verified by reading
  ``ebm.py``'s ``forward`` in full) even though the aux output is only
  attached to the returned dict when ``self.aux_score`` is truthy -- setting
  ``aux_score=False`` would make ``__init__`` skip constructing
  ``r_update_proj_aux`` entirely, and ``forward`` would then crash on a
  missing attribute. This is a genuine reference-code wart, not a port bug.
  """
  model_cfg = SimpleNamespace(
    token_s=SMOKE_TOKEN_S,
    token_z=SMOKE_TOKEN_Z,
    dim_fourier=SMOKE_DIM_FOURIER,
    conditioning_transition_layers=SMOKE_TRANSITION_LAYERS,
    token_transformer_depth=SMOKE_DEPTH,
    token_transformer_heads=SMOKE_HEADS,
    num_contact_embeddings=SMOKE_NUM_CONTACT_EMBEDDINGS,
    aux_score=True,
    predict_sidechain=False,
    diffuse_sidechain=False,
    use_self_conditioning=True,
    use_present_embedding=False,
    use_attention_mask=False,
    direct_score=False,
  )
  diffuser_cfg = SimpleNamespace(min_b=0.1, max_b=20.0, coordinate_scaling=0.1)
  return model_cfg, diffuser_cfg


def _build_reference_model(
  reference_repo: Path,
  model_cfg: Any,  # noqa: ANN401 -- duck-typed ml_collections.ConfigDict | SimpleNamespace
  diffuser_cfg: Any,  # noqa: ANN401
  state_dict: "Mapping[str, torch.Tensor] | None",
) -> "ProteinEBM":
  """Construct the real reference ``ProteinEBM``, optionally strict-loading a checkpoint.

  Mirrors ``scripts/ebm/checkpoint_parity_check.py::_build_reference_model``
  (same ``sys.path`` + import + ``load_state_dict(strict=False)`` +
  missing/unexpected check pattern) -- reimplemented here (not imported) so
  this benchmark script has no import-time dependency on another script's
  private (``_``-prefixed) functions, and so it can build the *tiny* --smoke
  architecture that script never needed to support.
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
  return model


def _build_jax_model(
  model_cfg: Any,  # noqa: ANN401
  seed: int,
  state_dict: "Mapping[str, Any] | None",
) -> ProteinEBMModel:
  """Construct the real ``aminx.ebm.model.ProteinEBMModel``, optionally porting a checkpoint."""
  key = jax.random.PRNGKey(seed)
  model = ProteinEBMModel(
    token_s=model_cfg.token_s,
    token_z=model_cfg.token_z,
    dim_fourier=model_cfg.dim_fourier,
    conditioning_transition_layers=model_cfg.conditioning_transition_layers,
    transformer_depth=model_cfg.token_transformer_depth,
    transformer_heads=model_cfg.token_transformer_heads,
    num_contact_embeddings=model_cfg.num_contact_embeddings,
    key=key,
  )
  if state_dict is not None:
    model, _report = load_pytorch_checkpoint(model, state_dict)
  return model


def build_models(
  args: argparse.Namespace,
) -> tuple[ProteinEBMModel, "ProteinEBM", Any]:
  """Build (jax_model, reference_model, model_cfg) per ``--smoke``/full-run mode."""
  if args.smoke:
    log.info("Building SMOKE models: tiny synthetic dims, randomly initialized, no checkpoint load.")
    model_cfg, diffuser_cfg = _smoke_model_configs()
    jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
    ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None)
    return jax_model, ref_model, model_cfg

  import torch  # noqa: PLC0415

  if not args.checkpoint.exists():
    msg = f"checkpoint not found: {args.checkpoint} (see E3.5 task brief for the authorized source)"
    raise FileNotFoundError(msg)
  if not args.reference_repo.exists():
    msg = f"reference repo not found: {args.reference_repo}"
    raise FileNotFoundError(msg)

  log.info("Loading checkpoint: %s", args.checkpoint)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model_cfg = ckpt["hyper_parameters"]["config"].model
  diffuser_cfg = ckpt["hyper_parameters"]["config"].diffuser

  log.info("Building + strict-loading reference PyTorch model...")
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, ckpt["state_dict"])

  log.info("Building + porting JAX ProteinEBMModel...")
  jax_model = _build_jax_model(model_cfg, args.seed, ckpt["state_dict"])

  return jax_model, ref_model, model_cfg


# ---------------------------------------------------------------------------
# Synthetic decoy inputs
# ---------------------------------------------------------------------------


def _synthetic_decoys(n_decoys: int, length: int, seed: int) -> dict[str, np.ndarray]:
  """Build one fixed synthetic decoy batch at a given residue length.

  Coordinates are i.i.d. Gaussian in already-scaled ("nm") units -- this is a
  throughput benchmark, not a correctness benchmark (design spec §8.3
  methodology point 1); a real correctness comparison against the paper's
  Rosetta decoy set is E5's job (``tests/ebm/test_decoy_ranking.py``'s
  ``TestRealCheckpointDecoyRankingProxy``), not this script's.

  ``aatype``/``residue_index``/``chain_id``/``contacts``/``mask`` are shared
  across the ``n_decoys`` axis (matching ``score_decoy_batch``'s own contract:
  each decoy has distinct coordinates but a shared sequence/mask) -- only
  ``coords`` varies per decoy.
  """
  rng = np.random.default_rng(seed)
  coords = rng.normal(scale=1.0, size=(n_decoys, length, 3)).astype(np.float32)
  aatype = rng.integers(0, 21, size=(length,)).astype(np.int64)
  residue_index = np.arange(length, dtype=np.int64)
  chain_id = np.zeros((length,), dtype=np.int64)
  contacts = np.zeros((length,), dtype=np.int64)
  mask = np.ones((length,), dtype=bool)
  return {
    "coords": coords,
    "aatype": aatype,
    "residue_index": residue_index,
    "chain_id": chain_id,
    "contacts": contacts,
    "mask": mask,
  }


def _timed_calls(fn: Any, n_repeats: int) -> list[float]:  # noqa: ANN401 -- Callable[[], object]
  """Run ``fn`` (already warmed up) ``n_repeats`` times, returning per-call wall-clock seconds."""
  times: list[float] = []
  for _ in range(n_repeats):
    start = time.perf_counter()
    fn()
    times.append(time.perf_counter() - start)
  return times


# ---------------------------------------------------------------------------
# JAX timing (energy_evals_per_sec via score_decoy_batch; score_grad_ms via
# ProteinEBMModel.score). Both wrapped in eqx.filter_jit -- the untimed
# warmup call forces XLA compilation for that shape; timed calls hit the
# compiled cache (methodology point 3).
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _jax_energy_batch(
  model: ProteinEBMModel,
  coords: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
) -> jax.Array:
  return score_decoy_batch(model, coords, aatype, t, mask)


@eqx.filter_jit
def _jax_score_one(
  model: ProteinEBMModel,
  coords: jax.Array,
  aatype: jax.Array,
  t: jax.Array,
  mask: jax.Array,
) -> jax.Array:
  return model.score(coords, aatype, t, mask)


def _run_jax_energy(
  model: ProteinEBMModel, decoys: dict[str, np.ndarray], t: float, n_repeats: int,
) -> list[float]:
  coords = jnp.asarray(decoys["coords"])
  aatype = jnp.asarray(decoys["aatype"], dtype=jnp.int32)
  mask = jnp.asarray(decoys["mask"])
  t_arr = jnp.asarray(t)

  def call() -> None:
    jax.block_until_ready(_jax_energy_batch(model, coords, aatype, t_arr, mask))

  call()  # untimed: forces XLA compilation for this (n_decoys, length) shape.
  return _timed_calls(call, n_repeats)


def _run_jax_score(
  model: ProteinEBMModel, decoys: dict[str, np.ndarray], t: float, n_repeats: int,
) -> list[float]:
  coords0 = jnp.asarray(decoys["coords"][0])
  aatype = jnp.asarray(decoys["aatype"], dtype=jnp.int32)
  mask = jnp.asarray(decoys["mask"])
  t_arr = jnp.asarray(t)

  def call() -> None:
    jax.block_until_ready(_jax_score_one(model, coords0, aatype, t_arr, mask))

  call()  # untimed: forces XLA compilation for this length's shape.
  return _timed_calls(call, n_repeats)


# ---------------------------------------------------------------------------
# PyTorch timing (batched compute_energy for throughput; explicit
# create_graph=False autograd.grad for score_grad_ms -- methodology point 2).
# Non-r_noisy input tensors are built once, outside the timed closure, so the
# timed region measures only forward(+backward) compute, matching the JAX
# side (which never rebuilds its device arrays per call either).
# ---------------------------------------------------------------------------


def _run_pytorch_energy(
  ref_model: "ProteinEBM", decoys: dict[str, np.ndarray], t: float, n_repeats: int,
) -> list[float]:
  import torch  # noqa: PLC0415

  n_decoys, length = decoys["coords"].shape[0], decoys["coords"].shape[1]
  r_noisy = torch.tensor(decoys["coords"], dtype=torch.float32)
  aatype = torch.tensor(np.tile(decoys["aatype"], (n_decoys, 1)), dtype=torch.long)
  residue_idx = torch.tensor(np.tile(decoys["residue_index"], (n_decoys, 1)), dtype=torch.long)
  chain_id = torch.tensor(np.tile(decoys["chain_id"], (n_decoys, 1)), dtype=torch.long)
  contacts = torch.tensor(np.tile(decoys["contacts"], (n_decoys, 1)), dtype=torch.long)
  mask = torch.tensor(np.tile(decoys["mask"], (n_decoys, 1)), dtype=torch.bool)
  times = torch.full((n_decoys,), t, dtype=torch.float32)
  input_feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": mask,
    "t": times,
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }
  del length  # unused; kept for readability of the tile shapes above.

  def call() -> None:
    # Pure forward energy eval -- no grad graph needed (matches the JAX side,
    # which never builds one for a plain score_decoy_batch/.energy() call).
    with torch.no_grad():
      ref_model.compute_energy(input_feats, rescale_input_coords=False)

  call()  # untimed: no JIT step in PyTorch, but a first call still primes
  # allocator/cache state (cache-warming parity with the JAX compile call).
  return _timed_calls(call, n_repeats)


def _run_pytorch_score(
  ref_model: "ProteinEBM", decoys: dict[str, np.ndarray], t: float, n_repeats: int,
) -> list[float]:
  import torch  # noqa: PLC0415

  coords0 = decoys["coords"][0]
  r_noisy = torch.tensor(coords0, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
  aatype = torch.tensor(decoys["aatype"], dtype=torch.long).unsqueeze(0)
  residue_idx = torch.tensor(decoys["residue_index"], dtype=torch.long).unsqueeze(0)
  chain_id = torch.tensor(decoys["chain_id"], dtype=torch.long).unsqueeze(0)
  contacts = torch.tensor(decoys["contacts"], dtype=torch.long).unsqueeze(0)
  mask = torch.tensor(decoys["mask"], dtype=torch.bool).unsqueeze(0)
  times = torch.tensor([t], dtype=torch.float32)
  input_feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": mask,
    "t": times,
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }

  def call() -> None:
    model_out = ref_model.compute_energy(input_feats, rescale_input_coords=False)
    energy = model_out["energy"]
    # Methodology point 2 (see module docstring): the reference's own
    # compute_score always uses create_graph=True even at inference. We call
    # autograd.grad directly with create_graph=False so PyTorch isn't
    # unfairly penalized by that correctness-irrelevant reference detail.
    # r_noisy is a persistent leaf (built once, above); each call rebuilds a
    # fresh graph from it via the forward pass, so re-using it across
    # repeated timed backward calls is safe.
    torch.autograd.grad(energy.sum(), r_noisy, create_graph=False)[0]

  call()  # untimed cache-warming call (no JIT step in PyTorch).
  return _timed_calls(call, n_repeats)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _jax_device() -> str:
  return jax.devices()[0].platform


def _torch_device() -> str:
  import torch  # noqa: PLC0415

  return "cuda" if torch.cuda.is_available() else "cpu"


def _run_dry_run(args: argparse.Namespace) -> int:
  """L1 gate: imports + paths + one real (but trivial) end-to-end call, no timed loop.

  Per ``~/.claude/rules/CLUSTER.md``'s L1 definition ("verify paths, imports,
  and that no network fetches will fail"): checks the checkpoint/reference-
  repo paths exist (skipped under --smoke, since that mode never touches
  them), then builds the tiny --smoke-dim models (regardless of whether
  --smoke was also passed) and runs each of the four timing functions once
  with ``n_repeats=1`` on a trivially small (2 decoys, 8 residues) input --
  enough to prove every import/construction/dispatch path actually executes,
  without spending time on a real timing loop.
  """
  log.info("=== L1 dry-run: import + construction + one-shot execution check ===")
  import torch  # noqa: PLC0415, F401 -- import-availability check

  if not args.reference_repo.exists():
    log.error("[FAIL] reference repo not found: %s", args.reference_repo)
    return 1
  log.info("[PASS] reference repo path exists: %s", args.reference_repo)

  if not args.smoke:
    if not args.checkpoint.exists():
      log.error("[FAIL] checkpoint not found: %s", args.checkpoint)
      return 1
    log.info("[PASS] checkpoint path exists: %s", args.checkpoint)

  log.info("Constructing tiny synthetic-dim models (real classes, random init) to prove the wiring...")
  model_cfg, diffuser_cfg = _smoke_model_configs()
  jax_model = _build_jax_model(model_cfg, args.seed, state_dict=None)
  ref_model = _build_reference_model(args.reference_repo, model_cfg, diffuser_cfg, state_dict=None)

  decoys = _synthetic_decoys(n_decoys=2, length=8, seed=args.seed)
  _run_jax_energy(jax_model, decoys, args.diffusion_time, n_repeats=1)
  _run_jax_score(jax_model, decoys, args.diffusion_time, n_repeats=1)
  _run_pytorch_energy(ref_model, decoys, args.diffusion_time, n_repeats=1)
  _run_pytorch_score(ref_model, decoys, args.diffusion_time, n_repeats=1)
  log.info("[PASS] JAX energy/score + PyTorch energy/score all executed once without error.")

  payload = {
    "dry_run": True,
    "note": "L1 gate only -- tiny synthetic dims, no real checkpoint/GPU timing performed.",
    "smoke_dims": {
      "token_s": SMOKE_TOKEN_S,
      "token_z": SMOKE_TOKEN_Z,
      "transformer_depth": SMOKE_DEPTH,
      "transformer_heads": SMOKE_HEADS,
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  log.info("[PASS] L1 dry-run OK. Wrote stub report to %s", args.out)
  return 0


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if args.dry_run:
    return _run_dry_run(args)

  lengths, n_decoys, n_repeats = _resolve_run_params(args)
  log.info(
    "=== decoy_benchmark: lengths=%s n_decoys=%d n_repeats=%d smoke=%s ===",
    lengths, n_decoys, n_repeats, args.smoke,
  )

  jax_model, ref_model, model_cfg = build_models(args)
  jax_device = _jax_device()
  torch_device = _torch_device()
  log.info("JAX device: %s | PyTorch device: %s", jax_device, torch_device)

  results: list[dict[str, Any]] = []
  wall_start = time.perf_counter()
  for length in lengths:
    decoys = _synthetic_decoys(n_decoys, length, args.seed)

    log.info("[length=%d] JAX warmup + timed energy batch (n_decoys=%d)...", length, n_decoys)
    jax_energy_times = _run_jax_energy(jax_model, decoys, args.diffusion_time, n_repeats)
    jax_energy_eps = n_decoys / float(np.mean(jax_energy_times))

    log.info("[length=%d] JAX warmup + timed score/grad...", length)
    jax_score_times = _run_jax_score(jax_model, decoys, args.diffusion_time, n_repeats)
    jax_score_ms = float(np.mean(jax_score_times)) * 1000.0

    log.info("[length=%d] PyTorch warmup + timed energy batch...", length)
    pt_energy_times = _run_pytorch_energy(ref_model, decoys, args.diffusion_time, n_repeats)
    pt_energy_eps = n_decoys / float(np.mean(pt_energy_times))

    log.info("[length=%d] PyTorch warmup + timed score/grad (create_graph=False)...", length)
    pt_score_times = _run_pytorch_score(ref_model, decoys, args.diffusion_time, n_repeats)
    pt_score_ms = float(np.mean(pt_score_times)) * 1000.0

    results.append({
      "protein_length": length,
      "impl": "jax",
      "device": jax_device,
      "energy_evals_per_sec": jax_energy_eps,
      "score_grad_ms": jax_score_ms,
    })
    results.append({
      "protein_length": length,
      "impl": "pytorch",
      "device": torch_device,
      "energy_evals_per_sec": pt_energy_eps,
      "score_grad_ms": pt_score_ms,
    })

    speedup = jax_energy_eps / pt_energy_eps if pt_energy_eps else float("nan")
    log.info(
      "[length=%d] jax: %.1f evals/s, %.3f ms/grad | pytorch: %.1f evals/s, %.3f ms/grad | "
      "jax/pytorch evals speedup=%.3fx",
      length, jax_energy_eps, jax_score_ms, pt_energy_eps, pt_score_ms, speedup,
    )

  wall_elapsed = time.perf_counter() - wall_start

  payload = {
    "meta": {
      "smoke": args.smoke,
      "n_decoys": n_decoys,
      "n_repeats": n_repeats,
      "diffusion_time": args.diffusion_time,
      "seed": args.seed,
      "checkpoint": None if args.smoke else str(args.checkpoint),
      "reference_repo": str(args.reference_repo),
      "model_dims": {
        "token_s": model_cfg.token_s,
        "token_z": model_cfg.token_z,
        "transformer_depth": model_cfg.token_transformer_depth,
        "transformer_heads": model_cfg.token_transformer_heads,
        "num_contact_embeddings": model_cfg.num_contact_embeddings,
      },
      "wall_clock_seconds": wall_elapsed,
      "methodology_notes": [
        "PyTorch score_grad_ms uses torch.autograd.grad(..., create_graph=False) "
        "explicitly -- the reference's own compute_score always uses "
        "create_graph=True even at inference (design spec §8.3 / EPIC MINOR-2).",
        "Both impls get one untimed warmup call per (length, metric) before "
        "timing (JAX: forces jit compilation; PyTorch: cache/allocator warm-up).",
        "No SM120 XLA autotuning workaround applied here -- CPU-only local "
        "run. A cluster (GPU) run of this script MUST set "
        "XLA_FLAGS=--xla_gpu_shard_autotuning=false per ~/.claude/rules/CLUSTER.md "
        "or JAX throughput numbers will be wrong by orders of magnitude.",
      ],
    },
    "results": results,
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  log.info("Wrote %d result rows to %s (wall clock %.1fs)", len(results), args.out, wall_elapsed)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
