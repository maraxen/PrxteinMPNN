#!/usr/bin/env python3
"""GPU benchmark adapter for LigandMPNN (PyTorch) inference.

Measures cold first-call overhead, warm latency, and GPU memory usage for each
(seq_len, batch_size, precision, task) cell. Produces JSON output conforming to
the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py --dry-run
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py --smoke
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py \
        --task score_conditional \
        --seq-lens 76 500 \
        --batch-sizes 1 4 16 \
        --precision fp32 \
        --hardware A100 \
        --n-warmup 10 \
        --n-timed 20 \
        --pdb-dir tests/data \
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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Element list matching LigandMPNN data_utils.py — index 0 = unknown, 1-indexed thereafter
_ELEMENT_LIST: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
_ELEMENT_LIST_UPPER: list[str] = [e.upper() for e in _ELEMENT_LIST]
_ELEMENT_TO_IDX: dict[str, int] = {e: i + 1 for i, e in enumerate(_ELEMENT_LIST_UPPER)}

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    94: "1BC8.cif",
    500: "1SMD.pdb",
}

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _get_cuda_version() -> str | None:
    """Try to get CUDA version, return None if unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "unknown"  # Can't easily extract full version; caller can query nvidia-smi
    except Exception:
        pass
    return None


def _get_torch_version() -> str:
    """Get PyTorch version."""
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ============================================================================
# Reference Path Resolution
# ============================================================================


def resolve_reference_path(reference_path: str | None = None) -> Path:
    """Resolve reference path from argument or environment."""
    if reference_path:
        path = Path(reference_path)
    else:
        env_path = os.environ.get("REFERENCE_PATH", "/home/marielle/repos/LigandMPNN")
        path = Path(env_path)

    if not path.exists():
        raise ValueError(f"REFERENCE_PATH does not exist: {path}")

    return path


# ============================================================================
# PDB Loading
# ============================================================================


def load_pdb_as_arrays(pdb_path: str) -> dict[str, Any]:
    """Load a PDB file and return coordinate arrays using biopython.

    Returns
    -------
    dict with keys:
        coords:         float32 (L, 4, 3)
        mask:           float32 (L,)
        sequence:       int32   (L,)         - amino acid indices 0-19 (unk=20)
        residue_index:  int32   (L,)
        chain_index:    int32   (L,)
        actual_len:     int
    """
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
            except Exception:
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
        "residue_index": np.array(residue_index_list, dtype=np.int64),
        "chain_index": np.array(chain_index_list, dtype=np.int64),
        "actual_len": L,
    }


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
]:
    """Load PDB fixture for a given nominal seq_len.

    Returns
    -------
    tuple
        (coords, mask, sequence, residue_index, chain_index, actual_len)
    """
    if seq_len not in _PDB_MAP:
        raise FileNotFoundError(
            f"No PDB fixture for seq_len={seq_len}. "
            f"Available: {list(_PDB_MAP.keys())}"
        )

    pdb_file = pdb_dir / _PDB_MAP[seq_len]
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB fixture not found: {pdb_file}")

    # Check if CIF file — dispatch to ligand loader
    if str(pdb_file).endswith(".cif"):
        result = _load_ligand_cif_fixture(pdb_dir, seq_len)
        if result is None:
            raise FileNotFoundError(f"CIF fixture not found: {pdb_file}")
        coords, mask, sequence, residue_index, chain_index, _, _, actual_len = result
        return coords, mask, sequence, residue_index, chain_index, actual_len

    data = load_pdb_as_arrays(str(pdb_file))
    return (
        data["coords"],
        data["mask"],
        data["sequence"].astype(np.int64),
        data["residue_index"],
        data["chain_index"],
        data["actual_len"],
    )


def _load_ligand_cif_fixture(
    pdb_dir: Path, seq_len: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    int,
] | None:
    """Load protein + ligand arrays from 1BC8.cif.

    Returns (coords, mask, sequence, residue_index, chain_index,
             ligand_xyz, ligand_elements, actual_len) or None if not a CIF fixture.
    Protein = chain C, ligand = chains A+B (non-water).
    """
    filename = _PDB_MAP.get(seq_len)
    if filename is None or not filename.endswith(".cif"):
        return None

    cif_path = pdb_dir / filename
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF fixture not found: {cif_path}")

    protein_atoms: list[tuple[str, int, float, float, float]] = []
    ligand_xyz: list[tuple[float, float, float]] = []
    ligand_elements: list[str] = []

    lines = cif_path.read_text().splitlines()
    in_loop = False
    col_names: list[str] = []
    col_map: dict[str, int] = {}

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            col_names = []; col_map = {}; in_loop = True; continue
        if in_loop and stripped.startswith("_atom_site."):
            col_names.append(stripped); col_map[stripped] = len(col_names) - 1; continue
        if in_loop and col_names and not stripped.startswith("_") and stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) < len(col_names): continue
            group = parts[col_map.get("_atom_site.group_PDB", 0)]
            if group not in ("ATOM", "HETATM"): continue
            element = parts[col_map["_atom_site.type_symbol"]].upper()
            resname = parts[col_map["_atom_site.label_comp_id"]]
            auth_chain = parts[col_map["_atom_site.auth_asym_id"]]
            atom_name = parts[col_map["_atom_site.label_atom_id"]].strip('"')
            try:
                auth_seq = int(parts[col_map["_atom_site.auth_seq_id"]])
                x = float(parts[col_map["_atom_site.Cartn_x"]])
                y = float(parts[col_map["_atom_site.Cartn_y"]])
                z = float(parts[col_map["_atom_site.Cartn_z"]])
            except (ValueError, KeyError): continue
            if auth_chain == "C" and resname not in ("HOH",):
                protein_atoms.append((atom_name, auth_seq, x, y, z))
            elif auth_chain in ("A", "B") and resname not in ("HOH",):
                ligand_xyz.append((x, y, z)); ligand_elements.append(element)
        elif in_loop and col_names and (stripped.startswith("_") or stripped == "loop_" or stripped == "#"):
            in_loop = False

    if not protein_atoms:
        raise ValueError(f"No protein atoms in chain C of {cif_path}")

    from collections import defaultdict
    residue_atoms: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for atom_name, res_seq, x, y, z in protein_atoms:
        if atom_name in ("N", "CA", "C", "O"):
            residue_atoms[res_seq][atom_name] = (x, y, z)

    sorted_resids = sorted(residue_atoms.keys())
    BACKBONE = ["N", "CA", "C", "O"]
    coords_list, mask_list = [], []
    for res in sorted_resids:
        row = [residue_atoms[res].get(a, (0.0, 0.0, 0.0)) for a in BACKBONE]
        coords_list.append(row)
        mask_list.append(1.0 if all(a in residue_atoms[res] for a in BACKBONE) else 0.0)

    actual_len = len(sorted_resids)
    coords = np.array(coords_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.float32)
    sequence = np.zeros(actual_len, dtype=np.int64)
    residue_index = np.array(sorted_resids, dtype=np.int64)
    chain_index = np.zeros(actual_len, dtype=np.int64)

    return (
        coords, mask, sequence, residue_index, chain_index,
        np.array(ligand_xyz, dtype=np.float32),
        np.array(ligand_elements),
        actual_len,
    )


def _compute_ligand_features_pytorch(
    coords: np.ndarray,
    ligand_xyz: np.ndarray,
    ligand_elements: np.ndarray,
    num_neighbors: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Y/Y_t/Y_m nearest-neighbor ligand features (mirrors LigandMPNN featurize).

    Returns:
        Y   : (L, A, 3)  float32 — ligand atom coordinates per residue
        Y_t : (L, A)     int32   — element type indices (1-indexed, 0=unknown)
        Y_m : (L, A)     float32 — validity mask
    """
    from scipy.spatial import cKDTree

    ca_coords = coords[:, 1, :]
    L = ca_coords.shape[0]
    A = num_neighbors
    n_query = min(A, len(ligand_xyz))

    tree = cKDTree(ligand_xyz)
    dists, idxs = tree.query(ca_coords, k=n_query)

    Y = np.zeros((L, A, 3), dtype=np.float32)
    Y_t = np.zeros((L, A), dtype=np.int32)
    Y_m = np.zeros((L, A), dtype=np.float32)

    idx_arr = idxs if n_query > 1 else idxs[:, None]
    for i in range(L):
        for j in range(n_query):
            idx = idx_arr[i, j]
            Y[i, j] = ligand_xyz[idx]
            elem = ligand_elements[idx].upper()
            Y_t[i, j] = _ELEMENT_TO_IDX.get(elem, 0)
            Y_m[i, j] = 1.0

    return Y, Y_t, Y_m


# ============================================================================
# Model & Feature Setup (PyTorch)
# ============================================================================


def load_model(
    reference_path: Path,
    device: Any,
    model_type: str = "protein_mpnn",
) -> Any:
    """Load pre-trained ProteinMPNN model (no-ligand variant for apples-to-apples comparison).

    Uses proteinmpnn_v_48_020.pt with model_type="protein_mpnn" for the no-ligand
    benchmark path. This avoids requiring Y/Y_t/Y_m ligand arrays in featurize().
    """
    import sys

    import torch

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
        model_type=model_type,
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing or unexpected:
        warnings.warn(
            f"Checkpoint key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys. "
            f"Model loaded with strict=False.",
            UserWarning,
            stacklevel=2,
        )

    model.to(device).eval()

    return model, atom_context_num


def create_protein_dict(
    coords: np.ndarray,
    mask: np.ndarray,
    sequence: np.ndarray,
    residue_index: np.ndarray,
    chain_index: np.ndarray,
    device: Any,
) -> dict[str, Any]:
    """Create protein_dict for featurize() call."""
    import torch

    L = coords.shape[0]
    return {
        "X": torch.tensor(coords, dtype=torch.float32, device=device),           # (L, 4, 3)
        "mask": torch.tensor(mask, dtype=torch.float32, device=device),           # (L,)
        "R_idx": torch.tensor(residue_index, dtype=torch.long, device=device),   # (L,)
        "chain_labels": torch.tensor(chain_index, dtype=torch.long, device=device),  # (L,)
        "S": torch.tensor(sequence, dtype=torch.long, device=device),            # (L,) native seq
        "chain_mask": torch.ones(L, dtype=torch.float32, device=device),         # (L,) design all
    }


def prepare_feature_dict(
    protein_dict: dict[str, Any],
    reference_path: Path,
    model_type: str,
    atom_context_num: int,
    device: Any,
) -> dict[str, Any]:
    """Prepare feature_dict for model.sample() or model.score()."""
    import sys

    import torch

    sys.path.insert(0, str(reference_path))
    from data_utils import featurize

    feature_dict = featurize(protein_dict, model_type=model_type, number_of_ligand_atoms=atom_context_num)

    L = feature_dict["X"].shape[1]
    feature_dict["temperature"] = 1.0
    feature_dict["bias"] = torch.zeros([1, L, 21], device=device)
    feature_dict["symmetry_residues"] = [[]]  # [[]] = no symmetry; [] causes IndexError in model.sample()
    feature_dict["symmetry_weights"] = [[]]

    return feature_dict


# ============================================================================
# Timing
# ============================================================================


def _cuda_sync(device: Any) -> None:
    """Synchronize CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure_cold_overhead_sample(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
) -> tuple[float, str]:
    """Measure cold first-call overhead for ar_sample."""
    import torch

    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = batch_size
    feature_dict["randn"] = torch.randn([batch_size, L], device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.sample(feature_dict)
    _cuda_sync(device)
    overhead_s = time.perf_counter() - t0

    note = "PyTorch eager mode: CUDA kernel warmup; no torch.compile; ar_sample uses model.sample()"
    return overhead_s, note


def measure_cold_overhead_score(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
    native_sequence: "Any",
) -> tuple[float, str]:
    """Measure cold first-call overhead for score_conditional.

    Uses model.score(feature_dict, use_sequence=True) to score the native sequence.
    """
    import torch

    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = batch_size
    # Set native sequence for scoring
    feature_dict["S"] = native_sequence.unsqueeze(0) if native_sequence.dim() == 1 else native_sequence

    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            out = model.score(feature_dict, use_sequence=True)
        except TypeError:
            # Fallback: some versions don't accept use_sequence
            out = model.score(feature_dict)
    _cuda_sync(device)
    overhead_s = time.perf_counter() - t0

    note = "PyTorch eager mode: CUDA kernel warmup; score_conditional uses model.score(use_sequence=True)"
    return overhead_s, note


def measure_warm_latency_sample(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for ar_sample."""
    import torch

    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = batch_size
    feature_dict["randn"] = torch.randn([batch_size, L], device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model.sample(feature_dict)
        _cuda_sync(device)

    times = []
    for _ in range(n_timed):
        feature_dict["randn"] = torch.randn([batch_size, L], device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.sample(feature_dict)
        _cuda_sync(device)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


def measure_warm_latency_score(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
    native_sequence: "Any",
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for score_conditional."""
    import torch

    feature_dict["batch_size"] = batch_size
    feature_dict["S"] = native_sequence.unsqueeze(0) if native_sequence.dim() == 1 else native_sequence

    def _score():
        try:
            return model.score(feature_dict, use_sequence=True)
        except TypeError:
            return model.score(feature_dict)

    for _ in range(n_warmup):
        with torch.no_grad():
            _score()
        _cuda_sync(device)

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        with torch.no_grad():
            _score()
        _cuda_sync(device)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


# ============================================================================
# Benchmark Single Cell
# ============================================================================


def benchmark_cell(
    model: Any,
    pdb_dir: Path,
    seq_len: int,
    batch_size: int,
    precision: str,
    task: str,
    device: Any,
    reference_path: Path,
    atom_context_num: int,
    ligand_conditioning: bool = False,
    model_type: str = "ligand_mpnn",
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any] | None:
    """Benchmark a single (seq_len, batch_size, precision, task) cell."""
    import torch

    try:
        if seq_len not in _PDB_MAP:
            logger.info(
                f"  Skipping L={seq_len} (no PDB fixture; have L={list(_PDB_MAP.keys())})"
            )
            return None

        coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
            pdb_dir, seq_len
        )

        if seq_len != actual_len:
            logger.info(
                f"  Note: nominal L={seq_len}, loaded L={actual_len} from {_PDB_MAP[seq_len]}"
            )

        if precision != "fp32":
            logger.info(f"  Skipping precision={precision}: LigandMPNN reference only supports fp32")
            return None

        # Build tensors
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        seq_t = torch.tensor(sequence, dtype=torch.long, device=device)
        residue_index_t = torch.tensor(residue_index, dtype=torch.long, device=device)
        chain_index_t = torch.tensor(chain_index, dtype=torch.long, device=device)

        protein_dict = {
            "X": coords_t,
            "mask": mask_t,
            "R_idx": residue_index_t,
            "chain_labels": chain_index_t,
            "S": seq_t,  # native sequence; overwritten below per task
            "chain_mask": torch.ones(actual_len, dtype=torch.float32, device=device),
        }

        # Ligand conditioning: load CIF and compute NN features
        if ligand_conditioning:
            lig_result = _load_ligand_cif_fixture(pdb_dir, seq_len)
            if lig_result is None:
                logger.info(f"  Skipping ligand (no CIF fixture for L={seq_len})")
                return None
            _, _, _, _, _, lig_xyz, lig_elems, _ = lig_result
            import torch
            Y, Y_t, Y_m = _compute_ligand_features_pytorch(
                np.array(coords), lig_xyz, lig_elems, num_neighbors=16
            )
            protein_dict["Y"] = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)
            protein_dict["Y_t"] = torch.tensor(Y_t, dtype=torch.long, device=device).unsqueeze(0)
            protein_dict["Y_m"] = torch.tensor(Y_m, dtype=torch.float32, device=device).unsqueeze(0)

        # Prepare feature dict
        model_type_arg = model_type
        if ligand_conditioning:
            model_type = "ligand_mpnn"
        try:
            feature_dict = prepare_feature_dict(
                protein_dict,
                reference_path=reference_path,
                model_type=model_type,
                atom_context_num=atom_context_num,
                device=device,
            )
            model_type = model_type_arg
        except Exception as e:
            logger.error(f"  Failed to prepare feature dict: {e}")
            return {
                "schema_version": "1",
                "model": "ligandmpnn_pytorch",
                "hardware": "unknown",
                "seq_len": actual_len,
                "batch_size": batch_size,
                "task": task,
                "precision": precision,
                "ligand_conditioning": False,
                "axis_strategy": None,
                "average_encoding_mode": None,
                "compile_time_cold_s": None,
                "compile_time_warm_s": None,
                "compile_time_note": f"featurize_failed: {e}",
                "latency_median_ms": None,
                "latency_p95_ms": None,
                "latency_per_residue_us": None,
                "throughput_seq_per_s": None,
                "peak_gpu_memory_gb": None,
                "n_warmup": n_warmup,
                "n_timed": n_timed,
                "jax_version": None,
                "torch_version": _get_torch_version(),
                "cuda_version": _get_cuda_version(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "skipped_reason": f"featurize_failed",
            }

        if task == "ar_sample":
            # For sampling, S is zeros (not fixed sequence)
            L = feature_dict["X"].shape[1]
            feature_dict["S"] = torch.zeros(1, L, dtype=torch.long, device=device)

            logger.info(
                f"  Computing cold overhead (seq_len={actual_len}, batch_size={batch_size}, task={task})..."
            )
            overhead_s, overhead_note = measure_cold_overhead_sample(
                model, feature_dict, batch_size, device
            )

            logger.info("  Computing warm latency...")
            median_s, p95_s, times = measure_warm_latency_sample(
                model, feature_dict, batch_size, device, n_warmup, n_timed
            )
        else:  # score_conditional
            # For scoring, use native sequence
            logger.info(
                f"  Computing cold overhead (seq_len={actual_len}, batch_size={batch_size}, task={task})..."
            )
            overhead_s, overhead_note = measure_cold_overhead_score(
                model, feature_dict, batch_size, device, seq_t
            )

            logger.info("  Computing warm latency...")
            median_s, p95_s, times = measure_warm_latency_score(
                model, feature_dict, batch_size, device, seq_t, n_warmup, n_timed
            )

        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (actual_len * batch_size)
        throughput_seq_per_s = batch_size / median_s

        peak_gpu_memory_gb = 0.0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                mb = int(result.stdout.strip().split("\n")[0].strip())
                peak_gpu_memory_gb = mb / 1024.0
        except Exception:
            pass

        result = {
            "schema_version": "1",
            "model": "ligandmpnn_pytorch",
            "hardware": "unknown",  # Set by caller
            "seq_len": actual_len,
            "batch_size": batch_size,
            "task": task,
            "precision": precision,
            "ligand_conditioning": ligand_conditioning,
            "axis_strategy": None,
            "average_encoding_mode": None,
            "compile_time_cold_s": float(overhead_s),
            "compile_time_warm_s": 0.0,
            "compile_time_note": overhead_note,
            "latency_median_ms": float(latency_median_ms),
            "latency_p95_ms": float(latency_p95_ms),
            "latency_per_residue_us": float(latency_per_residue_us),
            "throughput_seq_per_s": float(throughput_seq_per_s),
            "peak_gpu_memory_gb": float(peak_gpu_memory_gb),
            "n_warmup": n_warmup,
            "n_timed": n_timed,
            "jax_version": None,
            "torch_version": _get_torch_version(),
            "cuda_version": _get_cuda_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"  ✓ seq_len={actual_len}, batch_size={batch_size}, task={task}: "
            f"cold={overhead_s:.3f}s, warm_median={latency_median_ms:.2f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return None


# ============================================================================
# Main
# ============================================================================


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description="GPU benchmark for LigandMPNN PyTorch adapter",
    )
    parser.add_argument(
        "--task",
        choices=["score_conditional", "ar_sample"],
        default="score_conditional",
        help="Task to benchmark (default: score_conditional)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[76],
        help="Sequence lengths to benchmark (default: [76])",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: [1])",
    )
    parser.add_argument(
        "--precision",
        type=str,
        nargs="+",
        default=["fp32"],
        choices=["bf16", "fp32"],
        help="Precisions to benchmark (default: [fp32])",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="unknown",
        help="Hardware identifier (default: unknown)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Timed iterations (default: 20)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB fixture files 1ubq.pdb and 1SMD.pdb",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("outputs/benchmark_fixtures"),
        help="[DEPRECATED] Use --pdb-dir instead.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="Path to LigandMPNN reference repo (default: $REFERENCE_PATH env, /home/marielle/repos/LigandMPNN)",
    )
    parser.add_argument(
        "--ligand-conditioning",
        action="store_true",
        default=False,
        help="Enable ligand conditioning (requires 1BC8.cif fixture and ligand checkpoint).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running benchmarks",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal benchmark: seq_lens=[76], batch_sizes=[1]",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve pdb_dir: --pdb-dir > --fixture-dir (deprecated)
    if args.pdb_dir is None:
        if args.fixture_dir != Path("outputs/benchmark_fixtures"):
            warnings.warn(
                "--fixture-dir is deprecated; use --pdb-dir instead.",
                DeprecationWarning,
                stacklevel=1,
            )
            args.pdb_dir = args.fixture_dir
        else:
            # Default: auto-detect from script location
            args.pdb_dir = Path(__file__).parents[2] / "tests" / "data"

    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        logger.info("Smoke test mode: minimal configuration")

    # Resolve reference path
    try:
        reference_path = resolve_reference_path(args.reference_path)
    except ValueError as e:
        if args.dry_run:
            reference_path = Path(args.reference_path or "/home/marielle/repos/LigandMPNN")
            logger.warning(f"Reference path warning (dry-run): {e}")
        else:
            logger.error(f"Reference path error: {e}")
            return 1

    # Use protein_mpnn for no-ligand apples-to-apples comparison (unless ligand_conditioning)
    if args.ligand_conditioning:
        model_type = "ligand_mpnn"
    else:
        model_type = "protein_mpnn"

    if args.dry_run:
        config = {
            "task": args.task,
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "pdb_dir": str(args.pdb_dir),
            "reference_path": str(reference_path),
            "model_type": model_type,
            "cuda_available": _check_cuda_available(),
            "torch_version": _get_torch_version(),
        }
        logger.info("DRY RUN - Configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    if not args.pdb_dir.exists():
        logger.error(f"PDB directory not found: {args.pdb_dir}")
        return 1

    try:
        import torch
        logger.info(f"PyTorch {_get_torch_version()} loaded")
    except ImportError as e:
        logger.error(f"Failed to import PyTorch: {e}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available, falling back to CPU (benchmark will be slow)")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading LigandMPNN model (model_type={model_type})...")
    try:
        model, atom_context_num = load_model(reference_path, device, model_type=model_type)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    all_results = []
    total_cells = len(args.seq_lens) * len(args.batch_sizes) * len(args.precision)
    current_cell = 0

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            for precision in args.precision:
                current_cell += 1
                logger.info(
                    f"Benchmarking cell {current_cell}/{total_cells}: "
                    f"seq_len={seq_len}, batch_size={batch_size}, "
                    f"precision={precision}, task={args.task}"
                )

                result = benchmark_cell(
                    model,
                    args.pdb_dir,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    precision=precision,
                    task=args.task,
                    device=device,
                    reference_path=reference_path,
                    atom_context_num=atom_context_num,
                    ligand_conditioning=args.ligand_conditioning,
                    model_type=model_type,
                    n_warmup=args.n_warmup,
                    n_timed=args.n_timed,
                )

                if result is not None:
                    result["hardware"] = args.hardware
                    all_results.append(result)
                else:
                    logger.warning(f"Skipped cell {current_cell}")

    if not all_results:
        logger.error("No benchmarks completed successfully")
        return 1

    output = {
        "schema_version": "1",
        "results": all_results,
        "total_cells": len(all_results),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))

    logger.info(f"Benchmark complete: {len(all_results)}/{total_cells} cells successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
