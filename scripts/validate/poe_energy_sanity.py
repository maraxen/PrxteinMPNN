#!/usr/bin/env python3
"""Sanity check for PoE (Product of Experts) energy computation.

Validates two critical invariants:
  1. Identity-backbone invariant: PoeModel([params, params]).joint_energy(seq) ≈ 2*single_energy(seq, params)
  2. x2 scale factor: potts_log_unnormalized(2*h, 2*J, W, seq) matches expected

Alphabet: Potts uses canonical MPNN alphabet (q=21, indices 0-20 for A-Y-X).
DANGEROUS: This script WILL execute JAX computations on non-dry-run paths.
Always verify with --dry-run before running on cluster.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Add src/ to path
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

from aminx.model import ProteinFeatures
from aminx.potts.model import PottsModel, PottsParams
from aminx.potts.poe import PoeModel
from aminx.potts.sampling import _potts_log_unnormalized


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging to stderr with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PoE energy sanity check: validate identity-backbone and x2 scale invariants."
    )
    parser.add_argument(
        "--n-residues",
        type=int,
        default=6,
        help="Number of residues (sequence length)",
    )
    parser.add_argument(
        "--num-aa",
        type=int,
        default=20,
        help="Number of amino acids (should be 20 for standard, plus X=gap at idx 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/dev/null",
        help="Output file path (not used in this version)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all JAX computation, just validate args and imports then exit 0",
    )

    args = parser.parse_args()

    # Validate args
    if args.n_residues <= 0:
        parser.error(f"--n-residues must be > 0, got {args.n_residues}")
    if args.num_aa <= 0:
        parser.error(f"--num-aa must be > 0, got {args.num_aa}")

    # DRY-RUN: validate args and imports, then exit immediately
    if args.dry_run:
        print("dry-run: args OK", file=sys.stderr)
        return 0

    # ============================================================================
    # BEGIN COMPUTATION (JAX startup, model instantiation, forward passes)
    # ============================================================================

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting PoE energy sanity check: n_residues={args.n_residues}, "
        f"num_aa={args.num_aa}, seed={args.seed}"
    )

    # Sanity check: num_aa should align with MPNN alphabet
    if args.num_aa != 20:
        logger.warning(
            f"num_aa={args.num_aa} (expected 20 standard amino acids). "
            f"Full MPNN alphabet is 21 including gap (X) at index 20."
        )

    # Fixed parameters for tiny synthetic check
    n = args.n_residues
    num_aa = 21  # Always use full alphabet (20 + X at idx 20)
    hidden_dim = 16
    k_neighbors = min(3, n - 1)  # Tiny k for synthetic data
    edge_features_dim = 128  # Must match ProteinFeatures output dimension
    trw_iters = 2

    try:
        # Initialize JAX key and split for reproducibility
        key = jax.random.PRNGKey(args.seed)
        key, key_model1, key_model2 = jax.random.split(key, 3)

        logger.info(
            f"Creating PottsModel: n={n}, num_aa={num_aa}, hidden_dim={hidden_dim}, "
            f"k_neighbors={k_neighbors}, trw_iters={trw_iters}"
        )

        # Construct two identical PottsModel instances
        model1 = PottsModel(
            hidden_dim=hidden_dim,
            num_aa=num_aa,
            k_neighbors=k_neighbors,
            edge_features_dim=edge_features_dim,
            trw_iters=trw_iters,
            key=key_model1,
        )

        model2 = PottsModel(
            hidden_dim=hidden_dim,
            num_aa=num_aa,
            k_neighbors=k_neighbors,
            edge_features_dim=edge_features_dim,
            trw_iters=trw_iters,
            key=key_model2,
        )

        logger.info("Creating PoeModel with 2 identical backbones")
        poe = PoeModel((model1, model2))

        # Create synthetic random coordinates, mask, residue/chain indices
        key, *subkeys = jax.random.split(key, 4)
        coords = jax.random.normal(subkeys[0], (n, 37, 3), dtype=jnp.float32)
        mask = jnp.ones(n, dtype=jnp.float32)  # All residues valid
        residue_index = jnp.arange(n, dtype=jnp.int32)
        chain_index = jnp.zeros(n, dtype=jnp.int32)  # Single chain

        logger.info(
            f"Running ProteinFeatures to compute k-NN graph: coords shape {coords.shape}, mask shape {mask.shape}"
        )
        key, key_features = jax.random.split(key)

        # Compute k-NN graph features via ProteinFeatures
        features = ProteinFeatures(
            node_features=128,
            edge_features=128,
            k_neighbors=model1.k_neighbors,
            key=key_features,
        )

        edge_knn, nei, _, _ = features(
            key_features,
            coords,
            mask,
            residue_index,
            chain_index,
            backbone_noise=None,
        )

        # Cast to JAX arrays to ensure proper types
        edge_knn = jnp.asarray(edge_knn, dtype=jnp.float32)
        nei = jnp.asarray(nei, dtype=jnp.int32)

        logger.info(
            f"Running infer_params on model1: edge_knn shape {edge_knn.shape}, nei shape {nei.shape}"
        )
        key, key_infer = jax.random.split(key)
        params1 = model1.infer_params(
            key=key_infer,
            edge_knn=edge_knn,
            nei=nei,
            mask=mask,
        )

        logger.info(f"Inferred PottsParams: h shape {params1.h.shape}, J shape {params1.J.shape}")

        # Extract h, J, W from params
        h_raw = params1.h
        j_raw = params1.J
        w = params1.W

        # Fixed test sequence (all zeros -> all amino acid 0)
        seq = jnp.zeros(n, dtype=jnp.int32)

        # ========================================================================
        # ASSERTION 1: Identity-backbone invariant
        # PoeModel(n_backbones=2, identical backbones).joint_energy(seq, [params, params])
        # should equal 2 * potts_log_unnormalized(h, J, W, seq, mask)
        # within tolerance ±1e-5
        # ========================================================================

        logger.info("Testing ASSERTION 1: Identity-backbone invariant")
        logger.info("Computing single-backbone energy via PottsModel.log_prob")

        single_energy = PottsModel.log_prob(seq=seq, h=h_raw, j=j_raw, w=w, mask=mask)
        logger.info(f"  single_energy (PottsModel.log_prob) = {float(single_energy):.6e}")

        # Now compute via PoE with identical params
        logger.info("Computing joint energy via PoeModel with [params, params]")
        params_list = ((h_raw, j_raw, w), (h_raw, j_raw, w))
        joint_energy = poe.joint_energy(seq=seq, params_list=params_list)
        logger.info(f"  joint_energy (2x identical) = {float(joint_energy):.6e}")

        expected_joint = 2.0 * single_energy
        logger.info(f"  expected (2 * single) = {float(expected_joint):.6e}")

        diff_1 = float(jnp.abs(joint_energy - expected_joint))
        logger.info(f"  difference = {diff_1:.6e}")

        tol_1 = 1e-5
        assertion_1_pass = diff_1 < tol_1
        logger.info(
            f"ASSERTION 1: {'PASS' if assertion_1_pass else 'FAIL'} "
            f"(diff {diff_1:.6e} {'<' if assertion_1_pass else '>='} tol {tol_1:.6e})"
        )

        if not assertion_1_pass:
            logger.error(
                f"ASSERTION 1 FAILED: Identity-backbone invariant violated. "
                f"joint_energy={float(joint_energy):.6e} vs expected={float(expected_joint):.6e}, "
                f"diff={diff_1:.6e} >= tol={tol_1:.6e}"
            )

        # ========================================================================
        # ASSERTION 2: x2 scale factor verification
        # Verify that potts_log_unnormalized(2*h_raw, 2*J_raw, W, seq, mask)
        # matches the expected scaling. Given h and J carry x2 from PottsMPNN,
        # we verify that doubling them scales the energy accordingly.
        # ========================================================================

        logger.info("Testing ASSERTION 2: x2 scale factor consistency")

        # Compute energy with original h, J
        energy_1x = _potts_log_unnormalized(h=h_raw, j=j_raw, w=w, seq=seq, mask=mask)
        logger.info(f"  energy (1x h, J) = {float(energy_1x):.6e}")

        # Compute energy with 2x h, J
        h_2x = 2.0 * h_raw
        j_2x = 2.0 * j_raw
        energy_2x = _potts_log_unnormalized(h=h_2x, j=j_2x, w=w, seq=seq, mask=mask)
        logger.info(f"  energy (2x h, J) = {float(energy_2x):.6e}")

        # Expected: 2x scaling should give 2x energy
        expected_2x = 2.0 * energy_1x
        logger.info(f"  expected (2 * energy_1x) = {float(expected_2x):.6e}")

        diff_2 = float(jnp.abs(energy_2x - expected_2x))
        logger.info(f"  difference = {diff_2:.6e}")

        tol_2 = 1e-6
        assertion_2_pass = diff_2 < tol_2
        logger.info(
            f"ASSERTION 2: {'PASS' if assertion_2_pass else 'FAIL'} "
            f"(diff {diff_2:.6e} {'<' if assertion_2_pass else '>='} tol {tol_2:.6e})"
        )

        if not assertion_2_pass:
            logger.error(
                f"ASSERTION 2 FAILED: x2 scale factor mismatch. "
                f"energy_2x={float(energy_2x):.6e} vs expected={float(expected_2x):.6e}, "
                f"diff={diff_2:.6e} >= tol={tol_2:.6e}"
            )

        # ========================================================================
        # Final verdict
        # ========================================================================

        logger.info("")
        logger.info("=" * 70)
        if assertion_1_pass and assertion_2_pass:
            logger.info("ALL ASSERTIONS PASSED")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("ONE OR MORE ASSERTIONS FAILED")
            logger.error("=" * 70)
            return 1

    except NotImplementedError as e:
        logger.error(f"NotImplementedError (model not yet fully implemented): {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
