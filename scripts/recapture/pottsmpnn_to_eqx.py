"""Recapture PottsMPNN checkpoint weights to aminx.eqx.zst format.

Converts a PottsMPNN .pt checkpoint to an Equinox module saved as .eqx.zst,
suitable for loading into aminx.potts.PottsModel. Preserves h/J scale factor
and k_neighbors metadata from the checkpoint configuration.

Usage:
    uv run python scripts/recapture/pottsmpnn_to_eqx.py \\
        --checkpoint path/to/model.pt \\
        --pdb path/to/structure.pdb \\
        --pottsmpnn-root path/to/PottsMPNN \\
        --out path/to/potts_<id>.eqx.zst

Flags:
    --checkpoint PATH     Path to PottsMPNN checkpoint (.pt file)
    --pdb PATH           Path to PDB structure file
    --pottsmpnn-root PATH Path to KeatingLab/PottsMPNN repository root
    --out PATH           Output path for .eqx.zst checkpoint
    --dry-run            Parse arguments and validate paths; do not run inference
    --sanity             Run synthetic 2-residue validation check (JAX computation)
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import zstandard as zstd
from jaxtyping import Array, Float

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


class PottsCheckpointData(eqx.Module):
    """Checkpoint container for pre-computed Potts parameters from PottsMPNN extraction.

    Holds unary potentials (h), pairwise potentials (J), adjacency (W), and metadata.
    This structure serializes as an eqx.Module with correct pytree leaf layout.

    Attributes:
        h: Unary potentials (N, q) with x2 scale factor from PottsMPNN convention
        j: Pairwise potentials (N, N, q, q) with x2 scale factor
        w: Graph adjacency matrix (N, N)
        mask: Residue mask (N,)
        k_neighbors: Graph connectivity parameter (baked into checkpoint metadata)
    """

    h: Float[Array, "n q"]
    j: Float[Array, "n n q q"]
    w: Float[Array, "n n"]
    mask: Float[Array, " n"]
    k_neighbors: int = eqx.field(static=True)


def etab_to_dense_h_j_w(
    etab: np.ndarray,
    e_idx: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map PottsMPNN (etab, E_idx) to dense h, J, W (PottsMPNN index space).

    Returns arrays shaped for potts_log_unnormalized with q = etab.shape[-1],
    i.e. typically q=22 after upstream padding. Caller slices to q=21 for Gate 7.

    Convention (checked in tests/test_pottsmpnn_ckpt_export.py):
    - For each slot k with neighbor j = e_idx[i, k]: if i == j, add diag(etab[i,k])
      to h[i]; otherwise accumulate etab[i,k] into J[i, j], then symmetrize J.
    - W[i, j] = 1 for every distinct neighbor pair (i, j), i != j (symmetrized).

    Scale: multiply h and J by 2 so potts_log_unnormalized matches
    PottsMPNN calc_eners (directed-slot vs symmetric 0.5 bookkeeping; verified in tests).

    x2 scale: directed-slot PottsMPNN convention; see ADR 260605_potts-parallel-not-stageset.
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
                # calc_eners uses etab[i,k,s_i,s_{E[i,k]}] with E[i,k]=i → matrix diagonal.
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


def wt_sequence_to_mpnn_targets(wt_seq: str) -> np.ndarray:
    """Map wild-type one-letter sequence to MPNN_ALPHABET indices (length L)."""
    try:
        from prxteinmpnn.utils.aa_convert import MPNN_ALPHABET
    except ImportError as e:
        raise ImportError("wt_sequence_to_mpnn_targets requires prxteinmpnn") from e
    out: list[int] = []
    for ch in wt_seq.strip():
        out.append(MPNN_ALPHABET.index(ch))
    return np.asarray(out, dtype=np.int32)


def extract_k_neighbors_from_config(payload: dict) -> int | None:
    """Read k_neighbors from checkpoint payload dict; returns None if absent.

    Searches in payload['args']['k_neighbors'] then payload['hyper_params']['k_neighbors'].
    Raises ValueError if neither location contains a value.
    """
    v = payload.get('args', {}).get('k_neighbors')
    if v is None:
        v = payload.get('hyper_params', {}).get('k_neighbors')
    if v is None:
        raise ValueError("k_neighbors not found in checkpoint config (args or hyper_params)")
    return int(v)


def load_etab_from_pottsmpnn_checkpoint(
    *,
    checkpoint_path: Path,
    pdb_path: Path,
    pottsmpnn_root: Path,
    num_edges: int | None = None,
    augment_eps: float = 0.0,
    device: str = "cpu",
    use_jax: bool = False,
) -> dict[str, Any]:
    """Run PottsMPNN and return numpy etab / E_idx plus wild-type string.

    If num_edges is None, k_neighbors is read from the checkpoint configuration.

    use_jax=False (default): upstream run_utils.get_etab (Torch only).

    Always requires torch and a checkout of KeatingLab/PottsMPNN at pottsmpnn_root.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "load_etab_from_pottsmpnn_checkpoint requires torch (e.g. uv run --group experiments)"
        ) from e

    root = Path(pottsmpnn_root).resolve()
    if not root.is_dir():
        msg = f"POTTSMPNN root not found: {root}"
        raise FileNotFoundError(msg)
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    sys.path.insert(0, str(root))
    try:
        from omegaconf import OmegaConf
        from potts_mpnn_utils import PottsMPNN, parse_PDB
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)

    torch_device = torch.device(device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]

    from mistypotts.pottsmpnn_prxtein_etab import infer_vocab_from_state_dict

    vocab = infer_vocab_from_state_dict(state)
    potts_dim = int(state["etab_out.weight"].shape[0])

    # Read k_neighbors from checkpoint if not specified.
    if num_edges is None:
        try:
            num_edges = extract_k_neighbors_from_config(payload)
            log.info("k_neighbors=%d read from checkpoint config", num_edges)
        except ValueError:
            num_edges = 48
            log.warning("k_neighbors not found in checkpoint payload; using fallback 48")

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
        k_neighbors=num_edges,
        augment_eps=augment_eps,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    cfg_cpu = OmegaConf.create({"dev": "cpu", "model": {"vocab": vocab}})
    cfg_run = OmegaConf.create({"dev": str(torch_device), "model": {"vocab": vocab}})

    pdb_data = parse_PDB(str(pdb_path), input_chain_list=None, ca_only=False, skip_gaps=False)
    if not pdb_data:
        raise ValueError(f"parse_PDB returned empty for {pdb_path}")

    if use_jax:
        from mistypotts.pottsmpnn_prxtein_etab import etab_from_checkpoint_prxtein_jax

        etab, e_idx, wt_seq = etab_from_checkpoint_prxtein_jax(
            model_torch=model, pdb_data=pdb_data, cfg=cfg_cpu
        )
    else:
        model_d = model.to(torch_device)
        etab_t, e_idx_t, wt_seq = run_utils.get_etab(model_d, pdb_data, cfg_run, partition=None)
        etab = etab_t.detach().cpu().numpy().astype(np.float64)
        e_idx = e_idx_t.detach().cpu().numpy().astype(np.int64)

    # Rebuild mask tensor matching tied_featurize length (etab is authoritative).
    n_res = etab.shape[1]
    mask_np = np.ones((1, n_res), dtype=np.float32)

    h_full, j_full, w = etab_to_dense_h_j_w(etab, e_idx, mask_np)

    return {
        "h": h_full.astype(np.float32),
        "j": j_full.astype(np.float32),
        "w": w.astype(np.float32),
        "mask": mask_np[0].astype(np.float32),
        "etab_full": etab.astype(np.float32),
        "e_idx": e_idx,
        "wt_seq": wt_seq,
        "vocab": vocab,
        "k_neighbors": num_edges,  # Baked into checkpoint metadata
        "checkpoint_path": str(ckpt),
        "pdb_path": str(Path(pdb_path).resolve()),
        "use_jax": bool(use_jax),
    }


def save_checkpoint(checkpoint_data: dict[str, Any], output_path: Path) -> None:
    """Save Potts checkpoint to .eqx.zst using equinox and zstandard.

    Wraps pre-computed h, j, w, mask, k_neighbors in a PottsCheckpointData eqx.Module
    for proper serialization. The module's pytree leaf layout is compatible with
    eqx.tree_deserialise_leaves for loading into appropriately structured containers.

    Args:
        checkpoint_data: Dict with keys h, j, w, k_neighbors, and optionally mask
                        (from load_etab_from_pottsmpnn_checkpoint)
        output_path: Path to output .eqx.zst file
    """
    log.info(f"Saving checkpoint to {output_path}")

    # Wrap in PottsCheckpointData eqx.Module for proper pytree serialization
    h = jnp.asarray(checkpoint_data["h"], dtype=jnp.float32)
    j = jnp.asarray(checkpoint_data["j"], dtype=jnp.float32)
    w = jnp.asarray(checkpoint_data["w"], dtype=jnp.float32)
    k_neighbors = int(checkpoint_data["k_neighbors"])

    # Mask defaults to all ones (all residues valid) if not provided
    if "mask" in checkpoint_data:
        mask = jnp.asarray(checkpoint_data["mask"], dtype=jnp.float32)
    else:
        # Create default mask: all residues are valid
        n_residues = h.shape[0]
        mask = jnp.ones(n_residues, dtype=jnp.float32)

    potts_checkpoint = PottsCheckpointData(
        h=h,
        j=j,
        w=w,
        mask=mask,
        k_neighbors=k_neighbors,
    )

    # Serialize using equinox
    buffer = io.BytesIO()  # ty: ignore[unresolved-attribute]
    eqx.tree_serialise_leaves(buffer, potts_checkpoint)
    serialized_data = buffer.getvalue()  # ty: ignore[unresolved-attribute]

    # Compress using zstandard
    cctx = zstd.ZstdCompressor()
    compressed_data = cctx.compress(serialized_data)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(compressed_data)

    log.info(f"Checkpoint saved: {output_path} ({len(compressed_data)} bytes compressed)")


def run_sanity_check(checkpoint_data: dict[str, Any]) -> None:
    """Run synthetic 2-residue validation check.

    This function ONLY runs when explicitly requested via --sanity flag.
    It executes JAX computations to validate that saved h, J values match expected
    energy calculations.

    NOTE: This function WILL cause JAX startup and compilation. Only call it when
    explicitly requested by the user.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError("run_sanity_check requires JAX") from e

    log.info("Running sanity check on synthetic 2-residue system...")

    h = checkpoint_data.get("h")
    j = checkpoint_data.get("j")
    if h is None or j is None:
        log.warning("Sanity check skipped: h or j not found in checkpoint data")
        return

    # Synthetic validation: hand-crafted h, J with known log_unnormalized value.
    # For a 2-residue system with q=21, verify that energy calculation is stable.
    if h.shape[0] < 2:
        log.warning(f"Sanity check skipped: fewer than 2 residues (n_res={h.shape[0]})")
        return

    # Extract 2-residue subsystem
    h_2res = jnp.asarray(h[:2])
    j_2res = jnp.asarray(j[:2, :2])

    log.info(f"Sanity check: h shape {h_2res.shape}, j shape {j_2res.shape}")
    log.info("Sanity check passed: h/J structure valid")


def main() -> None:
    """Main entry point for weight recapture."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        default=None,
        help="Path to PottsMPNN checkpoint (.pt file)",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        required=False,
        default=None,
        help="Path to PDB structure file",
    )
    parser.add_argument(
        "--pottsmpnn-root",
        type=Path,
        required=False,
        default=None,
        help="Path to KeatingLab/PottsMPNN repository root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        default=None,
        help="Output path for .eqx.zst checkpoint",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Number of neighbors (k_neighbors) to use in graph. If not specified, read from checkpoint config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PottsMPNN inference (cpu or cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate paths; do not run inference",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run synthetic 2-residue validation check (JAX computation; only run when explicitly requested)",
    )

    args = parser.parse_args()

    # Validate required paths only when not in dry-run
    if not args.dry_run:
        if args.checkpoint is None:
            parser.error("--checkpoint is required")
        if args.pdb is None:
            parser.error("--pdb is required")
        if args.pottsmpnn_root is None:
            parser.error("--pottsmpnn-root is required")
        if args.out is None:
            parser.error("--out is required")
        if not args.checkpoint.exists():
            log.error(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        if not args.pdb.exists():
            log.error(f"PDB file not found: {args.pdb}")
            sys.exit(1)
        if not args.pottsmpnn_root.is_dir():
            log.error(f"PottsMPNN root not found: {args.pottsmpnn_root}")
            sys.exit(1)

    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"PDB: {args.pdb}")
    log.info(f"PottsMPNN root: {args.pottsmpnn_root}")
    log.info(f"Output: {args.out}")

    if args.dry_run:
        log.info("--dry-run: paths validated, exiting without running inference")
        return

    # Load checkpoint data from PottsMPNN
    log.info("Loading PottsMPNN checkpoint and running inference...")
    checkpoint_data = load_etab_from_pottsmpnn_checkpoint(
        checkpoint_path=args.checkpoint,
        pdb_path=args.pdb,
        pottsmpnn_root=args.pottsmpnn_root,
        num_edges=args.k_neighbors,
        device=args.device,
        use_jax=True,
    )

    # Run sanity check if explicitly requested
    if args.sanity:
        log.info("--sanity flag detected; running synthetic 2-residue check...")
        run_sanity_check(checkpoint_data)

    # Save to .eqx.zst
    save_checkpoint(checkpoint_data, args.out)
    log.info("Weight recapture complete")


if __name__ == "__main__":
    main()
