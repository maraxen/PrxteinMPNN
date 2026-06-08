#!/usr/bin/env python3
"""GPU benchmark for SafeMap heterogeneous batch (mixed-length) inference.

Measures the performance of prxteinmpnn's SafeMap dispatcher on a heterogeneous batch
containing structures with variable sequence lengths, compared to PyTorch baselines:
  - Padded to max length (batch_size=4, all sequences padded to L=max)
  - Sequential per-length (4 separate model calls, one per length)

Hypothesis: SafeMap heterogeneous batch avoids padding-induced computation overhead,
delivering higher per-residue throughput than max-length-padded baseline.

Usage:
    uv run python scripts/benchmarks/bench_mixed_length.py --dry-run
    uv run python scripts/benchmarks/bench_mixed_length.py --smoke
    uv run python scripts/benchmarks/bench_mixed_length.py \\
        --hardware A100 \\
        --lengths 76 150 300 500 \\
        --n-warmup 10 \\
        --n-timed 20 \\
        --pdb-dir tests/data \\
        --output-json results.json

Exit codes:
    0: SUCCESS
    1: FAILURE (missing fixtures, model load error, or benchmark error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import random

# Suppress JAX warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    150: "1mbn.pdb",
    300: "3pgk.pdb",
    500: "1SMD.pdb",
}

# Default PDB directory relative to the prxteinmpnn package root
_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults() -> None:
    """Set JAX configuration before importing models."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")


class _BenchmarkSpec:
    """Minimal spec for make_inference_plan."""
    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = [1.0]
    average_node_features = False


def _make_benchmark_spec_with_temperatures(temperature_list: list[float] | None = None) -> _BenchmarkSpec:
    """Create a benchmark spec with custom temperature list."""
    spec = _BenchmarkSpec()
    if temperature_list is not None:
        spec.temperature = temperature_list
    return spec


# ============================================================================
# PDB Loading
# ============================================================================


def load_pdb_as_arrays(pdb_path: str) -> dict[str, Any]:
    """Load a PDB file and return coordinate arrays using biopython."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords_list = []
    mask_list = []
    seq_list = []
    residue_index_list = []
    chain_index_list = []

    model_obj = next(iter(structure))
    chain_ids = sorted(chain.id for chain in model_obj)
    chain_to_idx = {cid: i for i, cid in enumerate(chain_ids)}

    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}

    for chain in model_obj:
        chain_idx = chain_to_idx[chain.id]
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue

            res_coord = np.zeros((4, 3), dtype=np.float32)
            atom_found = [False, False, False, False]

            for atom_name, atom_idx in atom_order.items():
                if atom_name in residue:
                    res_coord[atom_idx] = residue[atom_name].get_vector().get_array()
                    atom_found[atom_idx] = True

            res_mask = 1.0 if atom_found[1] else 0.0

            resname = residue.get_resname().strip()
            try:
                from Bio.Data.IUPACData import protein_letters_3to1
                one_letter = protein_letters_3to1.get(resname.capitalize(), "X")
                aa_idx = _AA_TO_IDX.get(one_letter, 20)
            except Exception:  # noqa: BLE001  # graceful degradation: default to unknown amino acid
                aa_idx = 20

            res_idx = residue.get_id()[1]

            coords_list.append(res_coord)
            mask_list.append(res_mask)
            seq_list.append(aa_idx)
            residue_index_list.append(res_idx)
            chain_index_list.append(chain_idx)

    L = len(coords_list)
    return {
        "coords": np.array(coords_list, dtype=np.float32),
        "mask": np.array(mask_list, dtype=np.float32),
        "sequence": np.array(seq_list, dtype=np.int32),
        "residue_index": np.array(residue_index_list, dtype=np.int32),
        "chain_index": np.array(chain_index_list, dtype=np.int32),
        "actual_len": L,
    }


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int
]:
    """Load PDB fixture for a given nominal seq_len."""
    if seq_len not in _PDB_MAP:
        raise FileNotFoundError(
            f"No PDB fixture for seq_len={seq_len}. "
            f"Available: {list(_PDB_MAP.keys())}"
        )

    pdb_file = pdb_dir / _PDB_MAP[seq_len]
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB fixture not found: {pdb_file}")

    data = load_pdb_as_arrays(str(pdb_file))

    return (
        jnp.asarray(data["coords"], dtype=jnp.float32),
        jnp.asarray(data["mask"], dtype=jnp.float32),
        jnp.asarray(data["sequence"], dtype=jnp.int32),
        jnp.asarray(data["residue_index"], dtype=jnp.int32),
        jnp.asarray(data["chain_index"], dtype=jnp.int32),
        data["actual_len"],
    )


# ============================================================================
# Model & Plan Setup
# ============================================================================


_DEFAULT_CHECKPOINT_ID = "proteinmpnn_v_48_020"


def load_model(
    checkpoint_path: Path | None = None,
    checkpoint_id: str | None = None,
) -> Any:
    """Load pre-trained model via io.weights.load_model."""
    from prxteinmpnn.io.weights import load_model as _load

    effective_id = checkpoint_id or _DEFAULT_CHECKPOINT_ID
    key = random.PRNGKey(42)
    local_path = str(checkpoint_path) if checkpoint_path is not None else None

    model = _load(
        checkpoint_id=effective_id,
        local_path=local_path,
        key=key,
    )
    if local_path:
        logger.info("Loaded checkpoint: %s (id=%s)", local_path, effective_id)
    else:
        logger.info("Loaded bundled checkpoint: %s", effective_id)
    return model


def create_inference_plan(model: Any, spec: _BenchmarkSpec | None = None) -> Any:
    """Create InferencePlan from model for score_conditional task."""
    from prxteinmpnn.host.plan import make_inference_plan

    if spec is None:
        spec = _BenchmarkSpec()

    # score_conditional uses ConditionalMode (make_inference_plan default)
    return make_inference_plan(model, spec)


# ============================================================================
# ColabDesign Sequential Baseline
# ============================================================================


def benchmark_colabdesign_sequential(
    pdb_dir: Path,
    lengths: list[int],
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float | None:
    """ColabDesign sequential baseline: one model.sample() call per length, summed.

    ColabDesign doesn't natively batch heterogeneous lengths, so we run
    prep_inputs() + sample() for each length separately and sum the times.

    Returns
    -------
    float | None
        Mean latency in milliseconds, or None if ColabDesign unavailable or fixture missing.
    """
    try:
        from colabdesign.mpnn.model import mk_mpnn_model
    except ImportError:
        logger.warning("ColabDesign not available; skipping baseline")
        return None

    import jax  # noqa: PLC0415  # lazy import for optional dependency

    # Verify all fixtures exist
    for length in lengths:
        if length not in _PDB_MAP:
            logger.warning("No ColabDesign fixture for length=%s", length)
            return None
        pdb_file = pdb_dir / _PDB_MAP[length]
        if not pdb_file.exists():
            logger.warning("ColabDesign fixture not found: %s", pdb_file)
            return None

    try:
        cd_model = mk_mpnn_model(
            model_name="v_48_020",
            backbone_noise=0.0,
            dropout=0.0,
            seed=42,
            verbose=False,
            weights="original",
        )

        def run_all_lengths() -> float:
            """Run model.sample() on each length and sum times."""
            total = 0.0
            for length in lengths:
                pdb_file = str(pdb_dir / _PDB_MAP[length])
                cd_model.prep_inputs(pdb_filename=pdb_file)
                t0 = time.perf_counter()
                result = cd_model.sample(num=1, temperature=1.0)
                jax.block_until_ready(jax.tree_util.tree_leaves(result))
                total += time.perf_counter() - t0
            return total

        # Warmup
        logger.info("ColabDesign warmup (%s iterations)...", n_warmup)
        for _ in range(n_warmup):
            run_all_lengths()

        # Timed
        logger.info("ColabDesign timed runs (%s iterations)...", n_timed)
        times = []
        for _ in range(n_timed):
            times.append(run_all_lengths())

        return float(np.mean(times)) * 1000.0  # ms, mean

    except Exception as e:  # noqa: BLE001  # graceful degradation: baseline skipped if unavailable
        logger.warning("ColabDesign sequential baseline failed: %s", e)
        return None


# ============================================================================
# PyTorch Baselines
# ============================================================================


def _load_pytorch_model(reference_path: Path, device: Any) -> tuple[Any, int]:
    """Load ProteinMPNN PyTorch model from reference path.

    Parameters
    ----------
    reference_path : Path
        Path to ligandmpnn_reference_assets directory.
    device : Any
        Torch device to load model onto.

    Returns
    -------
    tuple
        (model, atom_context_num)
    """
    import sys  # noqa: PLC0415  # lazy import for optional dependency
    import warnings  # noqa: PLC0415  # lazy import for optional dependency

    import torch  # noqa: PLC0415  # lazy import for optional dependency

    sys.path.insert(0, str(reference_path))
    from model_utils import ProteinMPNN

    checkpoint_path = reference_path / "model_params" / "proteinmpnn_v_48_020.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 1)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type="protein_mpnn",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, atom_context_num


def _cuda_sync() -> None:
    """Synchronize CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001  # graceful degradation: CUDA sync optional
        pass


def benchmark_pytorch_sequential(
    pdb_dir: Path,
    lengths: list[int],
    reference_path: Path | None,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float | None:
    """PyTorch baseline: one model.sample() call per length, summed.

    Processes each structure at its native length sequentially,
    measuring total time for all lengths.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB fixtures.
    lengths : list[int]
        Sequence lengths to benchmark.
    reference_path : Path | None
        Path to ligandmpnn_reference_assets. If None or missing, returns None.
    n_warmup : int
        Warmup iterations.
    n_timed : int
        Timed iterations.

    Returns
    -------
    float | None
        Mean latency in milliseconds, or None if reference_path missing or error.
    """
    if reference_path is None or not reference_path.exists():
        logger.warning("REFERENCE_PATH not set or not found; skipping PyTorch sequential baseline")
        return None

    try:
        import sys  # noqa: PLC0415  # lazy import for optional dependency

        import torch  # noqa: PLC0415  # lazy import for optional dependency

        sys.path.insert(0, str(reference_path))
        from data_utils import featurize  # noqa: PLC0415, E402  # lazy import for optional dependency

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt_model, atom_context_num = _load_pytorch_model(reference_path, device)

        # Load and prepare all structures
        structures = []
        for length in lengths:
            coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
                pdb_dir, length
            )

            coords_np = np.array(coords)
            mask_np = np.array(mask)
            seq_np = np.array(sequence)
            ri_np = np.array(residue_index)
            ci_np = np.array(chain_index)
            L = actual_len

            protein_dict = {
                "X": torch.tensor(coords_np, dtype=torch.float32, device=device),
                "mask": torch.tensor(mask_np, dtype=torch.float32, device=device),
                "R_idx": torch.tensor(ri_np, dtype=torch.long, device=device),
                "chain_labels": torch.tensor(ci_np, dtype=torch.long, device=device),
                "S": torch.tensor(seq_np, dtype=torch.long, device=device),
                "chain_mask": torch.ones(L, dtype=torch.float32, device=device),
            }
            fd = featurize(protein_dict, model_type="protein_mpnn", number_of_ligand_atoms=atom_context_num)
            fd["temperature"] = 1.0
            fd["bias"] = torch.zeros([1, fd["X"].shape[1], 21], device=device)
            fd["symmetry_residues"] = [[]]
            fd["symmetry_weights"] = [[]]
            structures.append(fd)

        def run_sequential() -> float:
            """Run sample on each structure and sum times."""
            total = 0.0
            for fd in structures:
                L = fd["X"].shape[1]
                fd["batch_size"] = 1
                fd["randn"] = torch.randn([1, L], device=device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    pt_model.sample(fd)
                _cuda_sync()
                total += time.perf_counter() - t0
            return total

        # Warmup
        logger.info("PyTorch sequential warmup (%s iterations)...", n_warmup)
        for _ in range(n_warmup):
            run_sequential()

        # Timed
        logger.info("PyTorch sequential timed runs (%s iterations)...", n_timed)
        times = [run_sequential() for _ in range(n_timed)]
        return float(np.mean(times)) * 1000.0  # ms

    except Exception as e:  # noqa: BLE001  # graceful degradation: baseline skipped if unavailable
        logger.warning("PyTorch sequential baseline failed: %s: %s", e.__class__.__name__, e)
        return None


def benchmark_pytorch_padded(
    pdb_dir: Path,
    lengths: list[int],
    reference_path: Path | None,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float | None:
    """PyTorch baseline: pad all structures to max length, run sequentially on padded.

    All structures are padded to L_max (the maximum actual length), then each is
    run separately through the model. This shows the cost of padding overhead.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB fixtures.
    lengths : list[int]
        Sequence lengths to benchmark.
    reference_path : Path | None
        Path to ligandmpnn_reference_assets. If None or missing, returns None.
    n_warmup : int
        Warmup iterations.
    n_timed : int
        Timed iterations.

    Returns
    -------
    float | None
        Mean latency in milliseconds, or None if reference_path missing or error.
    """
    if reference_path is None or not reference_path.exists():
        logger.warning("REFERENCE_PATH not set or not found; skipping PyTorch padded baseline")
        return None

    try:
        import sys  # noqa: PLC0415  # lazy import for optional dependency

        import torch  # noqa: PLC0415  # lazy import for optional dependency

        sys.path.insert(0, str(reference_path))
        from data_utils import featurize  # noqa: PLC0415, E402  # lazy import for optional dependency

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt_model, atom_context_num = _load_pytorch_model(reference_path, device)

        # Load all structures and find max length
        structures = []
        actual_lens = []
        for length in lengths:
            coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
                pdb_dir, length
            )
            structures.append((coords, mask, sequence, residue_index, chain_index))
            actual_lens.append(actual_len)

        L_max = max(actual_lens)
        logger.info("PyTorch padded: L_max=%s", L_max)

        # Pad all structures to L_max
        def pad_to(arr: np.ndarray, L_max: int, pad_value: float | int) -> np.ndarray:
            """Pad array to length L_max along first axis."""
            L = arr.shape[0]
            if L >= L_max:
                return arr[:L_max]
            pad_shape = (L_max - L,) + arr.shape[1:]
            return np.concatenate([arr, np.full(pad_shape, pad_value, dtype=arr.dtype)], axis=0)

        padded_structures = []
        for (coords, mask, sequence, residue_index, chain_index) in structures:
            coords_np = np.array(coords)
            mask_np = np.array(mask)
            seq_np = np.array(sequence)
            ri_np = np.array(residue_index)
            ci_np = np.array(chain_index)

            c_pad = pad_to(coords_np, L_max, 0.0)
            m_pad = pad_to(mask_np, L_max, 0.0)
            s_pad = pad_to(seq_np, L_max, 20)  # unknown aa = 20
            ri_pad = pad_to(ri_np, L_max, 0)
            ci_pad = pad_to(ci_np, L_max, 0)

            protein_dict = {
                "X": torch.tensor(c_pad, dtype=torch.float32, device=device),
                "mask": torch.tensor(m_pad, dtype=torch.float32, device=device),
                "R_idx": torch.tensor(ri_pad, dtype=torch.long, device=device),
                "chain_labels": torch.tensor(ci_pad, dtype=torch.long, device=device),
                "S": torch.tensor(s_pad, dtype=torch.long, device=device),
                "chain_mask": torch.ones(L_max, dtype=torch.float32, device=device),
            }
            fd = featurize(protein_dict, model_type="protein_mpnn", number_of_ligand_atoms=atom_context_num)
            fd["temperature"] = 1.0
            fd["bias"] = torch.zeros([1, L_max, 21], device=device)
            fd["symmetry_residues"] = [[]]
            fd["symmetry_weights"] = [[]]
            padded_structures.append(fd)

        def run_padded() -> float:
            """Run sample on each padded structure and sum times."""
            total = 0.0
            for fd in padded_structures:
                fd["batch_size"] = 1
                fd["randn"] = torch.randn([1, L_max], device=device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    pt_model.sample(fd)
                _cuda_sync()
                total += time.perf_counter() - t0
            return total

        # Warmup
        logger.info("PyTorch padded warmup (%s iterations)...", n_warmup)
        for _ in range(n_warmup):
            run_padded()

        # Timed
        logger.info("PyTorch padded timed runs (%s iterations)...", n_timed)
        times = [run_padded() for _ in range(n_timed)]
        return float(np.mean(times)) * 1000.0  # ms

    except Exception as e:  # noqa: BLE001  # graceful degradation: baseline skipped if unavailable
        logger.warning("PyTorch padded baseline failed: %s: %s", e.__class__.__name__, e)
        return None


# ============================================================================
# SafeMap Heterogeneous Batch Benchmark
# ============================================================================


def benchmark_safe_map_mixed_batch(
    model: Any,
    plan: Any,
    pdb_dir: Path,
    lengths: list[int],
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any]:
    """Benchmark SafeMap on heterogeneous batch with mixed sequence lengths.

    Loads one structure per length, measures latency for processing all structures
    together in a single timing run.

    Returns metrics:
      - mixed_batch_latency_ms: total latency
      - per_residue_throughput: residues/second
    """
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.tiling.bucketing import BucketingConfig

    _BUCKET_CFG = BucketingConfig()

    # Load one structure per length and build bundles
    bundles = []
    configs = []
    total_residues = 0

    for length in lengths:
        coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
            pdb_dir, length
        )
        total_residues += actual_len
        logger.info("Loaded %s: actual_len=%s", length, actual_len)

        # Build bundle for score_conditional
        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            ligand_coords=None,
            ligand_atom_types=None,
            ligand_mask=None,
            temperature=1.0,
            mode="score_conditional",
            inference=True,
            bucket_config=_BUCKET_CFG,
        )
        bundles.append(bundle)
        configs.append(config)

    logger.info("Total residues in heterogeneous batch: %s", total_residues)

    key = random.PRNGKey(42)

    # Create JIT function for scoring a single bundle/config pair
    @eqx.filter_jit
    def score_one(bundle: Any, config: Any) -> Any:
        """Score a single structure."""
        return plan.score(bundle, key, config)

    # Warmup
    logger.info("Warmup (%s iterations)...", n_warmup)
    for _ in range(n_warmup):
        for bundle, config in zip(bundles, configs):
            _ = score_one(bundle, config)

    # Timed runs (measure total time for all structures)
    logger.info("Timed runs (%s iterations)...", n_timed)
    times = []
    for _ in range(n_timed):
        start = time.perf_counter()
        for bundle, config in zip(bundles, configs):
            _ = score_one(bundle, config)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    latency_ms = np.mean(times)
    latency_s = latency_ms / 1000.0
    per_residue_throughput = total_residues / latency_s

    return {
        "mixed_batch_latency_ms": float(latency_ms),
        "per_residue_throughput": float(per_residue_throughput),
        "total_residues": int(total_residues),
        "latency_samples_ms": [float(t) for t in times],
    }


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Main entry point."""
    _set_jax_defaults()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark SafeMap heterogeneous batch (mixed-length) inference."
    )

    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Hardware name (e.g., A100, H100, H200, blackwell)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[76, 150, 300, 500],
        help="Sequence lengths in heterogeneous batch (default: 76 150 300 500)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB fixture files (defaults to tests/data)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke-test mode (1 warmup, 3 timed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved parameters without executing",
    )
    # Unused args for compatibility
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="[UNUSED] For CLI compatibility",
    )

    args = parser.parse_args()

    # Resolve pdb_dir
    if args.pdb_dir is None:
        args.pdb_dir = _DEFAULT_PDB_DIR

    # Handle --smoke
    if args.smoke:
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke mode: n_warmup=1, n_timed=3")

    logger.info("Hardware: %s", args.hardware)
    logger.info("Batch lengths: %s", args.lengths)
    logger.info("PDB directory: %s", args.pdb_dir)

    if args.dry_run:
        logger.info("Dry-run complete")
        return 0

    # Load model
    logger.info("Loading prxteinmpnn model...")
    try:
        model = load_model()
    except Exception as e:  # noqa: BLE001  # graceful degradation: benchmark skipped if unavailable
        logger.error("Failed to load model: %s", e)
        return 1

    # Create inference plan
    logger.info("Creating inference plan...")
    try:
        spec = _make_benchmark_spec_with_temperatures()
        plan = create_inference_plan(model, spec)
    except Exception as e:  # noqa: BLE001  # graceful degradation: benchmark skipped if unavailable
        logger.error("Failed to create inference plan: %s", e)
        return 1

    # Run SafeMap mixed-batch benchmark
    logger.info("=" * 70)
    logger.info("SafeMap Mixed-Length Batch Benchmark")
    logger.info("=" * 70)
    try:
        safemap_results = benchmark_safe_map_mixed_batch(
            model=model,
            plan=plan,
            pdb_dir=args.pdb_dir,
            lengths=args.lengths,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )
        logger.info("SafeMap results: %s", safemap_results)
    except Exception as e:  # noqa: BLE001  # graceful degradation: benchmark skipped if unavailable
        logger.error("SafeMap benchmark failed: %s", e)
        import traceback  # noqa: PLC0415  # lazy import for debugging
        traceback.print_exc()
        return 1

    # Run ColabDesign sequential baseline
    logger.info("=" * 70)
    logger.info("ColabDesign Sequential Baseline")
    logger.info("=" * 70)
    cd_latency_ms = benchmark_colabdesign_sequential(
        pdb_dir=args.pdb_dir,
        lengths=args.lengths,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )
    if cd_latency_ms is not None:
        logger.info("ColabDesign sequential: %fms", cd_latency_ms)
    else:
        logger.info("ColabDesign sequential: skipped or unavailable")

    # Resolve reference_path for PyTorch baselines
    reference_path_str = os.environ.get("REFERENCE_PATH", "")
    reference_path = Path(reference_path_str) if reference_path_str else None

    # Run PyTorch padded baseline
    logger.info("=" * 70)
    logger.info("PyTorch Padded Baseline")
    logger.info("=" * 70)
    pt_padded_ms = benchmark_pytorch_padded(
        pdb_dir=args.pdb_dir,
        lengths=args.lengths,
        reference_path=reference_path,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )
    if pt_padded_ms is not None:
        logger.info("PyTorch padded: %fms", pt_padded_ms)
    else:
        logger.info("PyTorch padded: skipped or unavailable")

    # Run PyTorch sequential baseline
    logger.info("=" * 70)
    logger.info("PyTorch Sequential Baseline")
    logger.info("=" * 70)
    pt_sequential_ms = benchmark_pytorch_sequential(
        pdb_dir=args.pdb_dir,
        lengths=args.lengths,
        reference_path=reference_path,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )
    if pt_sequential_ms is not None:
        logger.info("PyTorch sequential: %fms", pt_sequential_ms)
    else:
        logger.info("PyTorch sequential: skipped or unavailable")

    # Compute throughput improvement (SafeMap vs PyTorch padded)
    throughput_improvement = None
    if (
        pt_padded_ms is not None
        and pt_padded_ms > 0
        and safemap_results["mixed_batch_latency_ms"] > 0
    ):
        throughput_improvement = pt_padded_ms / safemap_results["mixed_batch_latency_ms"]

    # Assemble output
    output = {
        "schema_version": "1",
        "hardware": args.hardware,
        "batch_lengths": args.lengths,
        "mixed_latency_ms": safemap_results["mixed_batch_latency_ms"],
        "per_residue_throughput": safemap_results["per_residue_throughput"],
        "pytorch_padded_latency_ms": pt_padded_ms,
        "pytorch_sequential_latency_ms": pt_sequential_ms,
        "per_residue_throughput_improvement_vs_padded": throughput_improvement,
        "colabdesign_sequential_latency_ms": cd_latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results written to %s", args.output_json)
    return 0


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


if __name__ == "__main__":
    sys.exit(main())
