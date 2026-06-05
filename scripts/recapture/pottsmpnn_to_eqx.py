"""Recapture PottsMPNN checkpoint weights into Equinox .eqx.zst format.

This script loads a PottsMPNN checkpoint from PyTorch, extracts the dense h, J, W
tensors (Potts energy matrices), and saves them as Equinox leaves with metadata.

Key conventions:
- x2 scale: directed-slot PottsMPNN convention; see ADR 260605_potts-parallel-not-stageset
- k_neighbors: read from loaded model config (not CLI args — k is baked in checkpoint)
- Checkpoint metadata: k_neighbors, checkpoint_path, wt_seq, vocab size
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import zstandard as zstd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def etab_to_dense_h_j_w(
    etab: np.ndarray,
    e_idx: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map PottsMPNN ``(etab, E_idx)`` to dense ``h``, ``J``, ``W`` (PottsMPNN index space).

    Returns arrays shaped for ``potts_log_unnormalized`` with ``q = etab.shape[-1]``,
    i.e. typically ``q=22`` after upstream padding. Caller slices to ``q=21`` for Gate 7.

    **Convention (checked in tests):**

    - For each slot ``k`` with neighbor ``j = e_idx[i, k]``: if ``i == j``, add ``diag(etab[i,k])``
      to ``h[i]``; otherwise accumulate ``etab[i,k]`` into ``J[i, j]``, then symmetrize ``J``.
    - ``W[i, j] = 1`` for every distinct neighbor pair ``(i, j)``, ``i != j`` (symmetrized).

    **Scale:** multiply ``h`` and ``J`` by **2** so ``potts_log_unnormalized`` matches
    PottsMPNN ``calc_eners`` (directed-slot vs symmetric ``0.5`` bookkeeping; verified in tests).
    """
    if etab.ndim != 5 or e_idx.ndim != 3:
        msg = f"expected etab (1,L,K,q,q) and e_idx (1,L,K); got {etab.shape}, {e_idx.shape}"
        raise ValueError(msg)
    etab = np.asarray(etab, dtype=np.float64)
    e_idx = np.asarray(e_idx, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.float64)
    if etab.shape[0] != 1 or e_idx.shape[0] != 1:
        raise ValueError("batch size must be 1")
    etab0 = etab[0]
    e0 = e_idx[0]
    m = mask[0] if mask.ndim == 2 else mask
    n_res, k, q, q2 = etab0.shape
    if q != q2:
        raise ValueError("etab last two dims must be square")
    h = np.zeros((n_res, q), dtype=np.float64)
    j = np.zeros((n_res, n_res, q, q), dtype=np.float64)
    w = np.zeros((n_res, n_res), dtype=np.float64)
    for i in range(n_res):
        if m[i] <= 0:
            continue
        for kk in range(k):
            jn = int(e0[i, kk])
            if jn < 0 or jn >= n_res or m[jn] <= 0:
                continue
            block = etab0[i, kk].astype(np.float64)
            if i == jn:
                # ``calc_eners`` uses ``etab[i,k,s_i,s_{E[i,k]}]`` with ``E[i,k]=i`` → matrix diagonal.
                h[i] += np.diag(block)
            else:
                j[i, jn] += block
                w[i, jn] = 1.0
                w[jn, i] = 1.0
    j = 0.5 * (j + np.transpose(j, (1, 0, 3, 2)))
    w = np.maximum(w, w.T)
    # x2 scale: directed-slot PottsMPNN convention; see ADR 260605_potts-parallel-not-stageset
    h *= 2.0
    j *= 2.0
    return h.astype(np.float32), j.astype(np.float32), w.astype(np.float32)


def slice_potts_to_mpnn_q(
    h: np.ndarray,
    j: np.ndarray,
    *,
    q_mpnn: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the leading ``q_mpnn × q_mpnn`` interaction block (standard AAs + X)."""
    if h.shape[-1] < q_mpnn or j.shape[-1] < q_mpnn:
        msg = f"h,j last dims must be >= {q_mpnn}, got h={h.shape}, j={j.shape}"
        raise ValueError(msg)
    h21 = np.asarray(h[..., :q_mpnn], dtype=np.float32)
    j21 = np.asarray(j[:, :, :q_mpnn, :q_mpnn], dtype=np.float32)
    return h21, j21


def load_and_convert_checkpoint(
    checkpoint_path: Path,
    pdb_path: Path,
    pottsmpnn_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load PottsMPNN checkpoint and convert to dense h, J, W tensors.

    Returns a dict with keys:
    - 'h': (n_res, 21) field tensor
    - 'j': (n_res, n_res, 21, 21) pairwise interaction tensor
    - 'w': (n_res, n_res) adjacency tensor
    - 'mask': (n_res,) residue mask
    - 'k_neighbors': from loaded model config
    - 'checkpoint_path': path to source checkpoint
    - 'pdb_path': path to source PDB
    - 'wt_seq': wild-type sequence string
    - 'vocab': vocab size (from state dict)

    Requires torch and the PottsMPNN repository.
    """
    try:
        import torch
    except ImportError as e:
        msg = "load_and_convert_checkpoint requires torch (e.g. uv run --group experiments)"
        raise ImportError(msg) from e

    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    pdb_file = Path(pdb_path).resolve()
    if not pdb_file.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_file}")

    # Inject pottsmpnn_root into sys.path
    root = Path(pottsmpnn_root).resolve() if pottsmpnn_root else Path.home() / ".local" / "PottsMPNN"
    if not root.is_dir():
        msg = f"POTTSMPNN root not found: {root}. Set --pottsmpnn-root or check environment."
        raise FileNotFoundError(msg)

    sys.path.insert(0, str(root))
    try:
        import run_utils
        from omegaconf import OmegaConf
        from potts_mpnn_utils import PottsMPNN, parse_PDB
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)

    log.info("Loading checkpoint from %s", ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]

    # Infer vocab size from state dict
    from mistypotts.pottsmpnn_prxtein_etab import infer_vocab_from_state_dict

    vocab = infer_vocab_from_state_dict(state)
    potts_dim = int(state["etab_out.weight"].shape[0])
    log.info("Inferred vocab=%d, potts_dim=%d", vocab, potts_dim)

    # Load checkpoint config to get k_neighbors
    checkpoint_cfg = payload.get("model_cfg", {})
    if isinstance(checkpoint_cfg, str):
        try:
            checkpoint_cfg = json.loads(checkpoint_cfg)
        except json.JSONDecodeError:
            checkpoint_cfg = {}

    k_neighbors = checkpoint_cfg.get("k_neighbors", 48)
    log.info("k_neighbors from config: %d", k_neighbors)

    # Construct model
    model = PottsMPNN(
        ca_only=False,
        num_letters=vocab,
        vocab=vocab,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        potts_dim=potts_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        augment_eps=0.0,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    log.info("Loaded model, processing PDB %s", pdb_file)

    cfg_cpu = OmegaConf.create({"dev": "cpu", "model": {"vocab": vocab}})
    pdb_data = parse_PDB(str(pdb_file), input_chain_list=None, ca_only=False, skip_gaps=False)
    if not pdb_data:
        raise ValueError(f"parse_PDB returned empty for {pdb_file}")

    # Extract etab via run_utils.get_etab
    model_d = model.to("cpu")
    etab_t, e_idx_t, wt_seq = run_utils.get_etab(model_d, pdb_data, cfg_cpu, partition=None)
    etab = etab_t.detach().cpu().numpy().astype(np.float64)
    e_idx = e_idx_t.detach().cpu().numpy().astype(np.int64)

    log.info("Extracted etab shape: %s, e_idx shape: %s", etab.shape, e_idx.shape)
    log.info("Wild-type sequence: %s", wt_seq)

    # Rebuild mask
    n_res = etab.shape[1]
    mask_np = np.ones((1, n_res), dtype=np.float32)

    # Convert to dense h, J, W
    h_full, j_full, w = etab_to_dense_h_j_w(etab, e_idx, mask_np)

    # Slice to q=21
    h, j = slice_potts_to_mpnn_q(h_full, j_full, q_mpnn=21)

    if dry_run:
        log.info("DRY RUN: would save %d×21 h, %d×%d×21×21 j, %d×%d w", h.shape[0], j.shape[0], j.shape[1], w.shape[0], w.shape[1])
        return {
            "h": h,
            "j": j,
            "w": w,
            "mask": mask_np[0].astype(np.float32),
            "k_neighbors": k_neighbors,
            "checkpoint_path": str(ckpt),
            "pdb_path": str(pdb_file),
            "wt_seq": wt_seq,
            "vocab": vocab,
        }

    return {
        "h": h,
        "j": j,
        "w": w,
        "mask": mask_np[0].astype(np.float32),
        "k_neighbors": k_neighbors,
        "checkpoint_path": str(ckpt),
        "pdb_path": str(pdb_file),
        "wt_seq": wt_seq,
        "vocab": vocab,
    }


def save_checkpoint_eqx_zst(data: dict[str, Any], output_path: Path) -> None:
    """Save converted checkpoint as Equinox leaves with zstd compression.

    Uses equinox.tree_serialise_leaves + zstandard for compression.
    Metadata (k_neighbors, checkpoint_path, pdb_path, wt_seq, vocab) stored as JSON in a leaf.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    metadata = {
        "k_neighbors": int(data["k_neighbors"]),
        "checkpoint_path": data["checkpoint_path"],
        "pdb_path": data["pdb_path"],
        "wt_seq": data["wt_seq"],
        "vocab": int(data["vocab"]),
    }

    # Build PyTree: tensors as leaves, metadata as a leaf
    pytree = {
        "h": jax.numpy.asarray(data["h"]),
        "j": jax.numpy.asarray(data["j"]),
        "w": jax.numpy.asarray(data["w"]),
        "mask": jax.numpy.asarray(data["mask"]),
        "metadata": metadata,
    }

    # Serialize via Equinox
    log.info("Serializing to %s", output_path)
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, pytree)
    serialized_bytes = stream.getvalue()

    # Compress with zstd
    log.info("Compressing with zstd")
    cctx = zstd.ZstdCompressor(level=10)
    compressed = cctx.compress(serialized_bytes)

    # Write
    output_path.write_bytes(compressed)
    log.info("Saved checkpoint to %s (%d bytes compressed)", output_path, len(compressed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to PottsMPNN .pt checkpoint")
    parser.add_argument("--pdb", type=Path, required=True, help="Path to structure PDB file")
    parser.add_argument("--pottsmpnn-root", type=Path, default=None, help="Path to PottsMPNN repo root")
    parser.add_argument("--out", type=Path, required=True, help="Output .eqx.zst file path")
    parser.add_argument("--dry-run", action="store_true", help="Load checkpoint but don't save")

    args = parser.parse_args()

    try:
        data = load_and_convert_checkpoint(
            checkpoint_path=args.checkpoint,
            pdb_path=args.pdb,
            pottsmpnn_root=args.pottsmpnn_root,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            log.info("DRY RUN completed successfully")
            print(json.dumps({
                "h_shape": list(data["h"].shape),
                "j_shape": list(data["j"].shape),
                "w_shape": list(data["w"].shape),
                "mask_shape": list(data["mask"].shape),
                "k_neighbors": data["k_neighbors"],
                "wt_seq": data["wt_seq"],
                "vocab": data["vocab"],
            }, indent=2))
        else:
            save_checkpoint_eqx_zst(data, args.out)
            log.info("Checkpoint saved successfully")

    except Exception as e:
        log.exception("Failed to convert checkpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
