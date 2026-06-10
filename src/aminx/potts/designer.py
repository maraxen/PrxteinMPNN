"""MPNN-Potts Designer: Coordinator wrapper for MPNN-guided Potts design.

This module implements MpnnPottsDesigner(eqx.Module), the sole permitted cross-pipeline
orchestrator for combining Aminx (MPNN) and PottsModel inference. This is the ONLY file
in aminx.potts permitted to import from aminx.inference.*

Architecture:
  - MpnnPottsDesigner holds both Aminx (MPNN encoder) and PottsModel (TRW inference)
  - Provides unified run_design() API that orchestrates both models
  - MPNN supplies unconditional logits to seed Gibbs sampling in Potts
  - Potts alphabet_map remaps MPNN logits to Potts vocabulary (identity by design)
  - Separates concerns: MPNN logits (per-position) → Potts energies (global) → Gibbs

Reference: ADR 260605_potts-parallel-not-stageset (MpnnPottsDesigner is Approach D)

Risks:
  - alphabet_map index 20 collision (X/gap): mitigated by static identity map + warning
  - seed_mode 'unconditional' vs 'order_averaged': flag controls MPNN AR order bias

References
----------
.. [PottsMPNN] Birnbaum and Keating. "Beyond native sequence recovery: Improved
   modeling of the sequence-energy landscape of protein structures." bioRxiv, 2026.
   https://doi.org/10.64898/2026.01.14.699067
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.model.mpnn import Aminx  # noqa: TC001 (used at runtime in __init__)
from aminx.potts.model import POTTS_TO_MPNN_ALPHABET_MAP, PottsModel
from aminx.potts.sampling import gibbs_sweep
from aminx.utils.aa_convert import MPNN_ALPHABET


@dataclass(frozen=True)
class DesignResult:
  """Result from MpnnPottsDesigner.run_design().

  Attributes:
      mpnn_logits: Per-position logits from MPNN (L, 21) unconditional scoring
      potts_marginals: Node marginals from Potts TRW inference (N, 21)
      gibbs_seq: Final designed sequence after Gibbs sweeps (N,)
      sequence: Sequence as string (convenience)
  """

  mpnn_logits: Float[Array, "n 21"]
  potts_marginals: Float[Array, "n 21"]
  gibbs_seq: Int[Array, " n"]
  sequence: str


class MpnnPottsDesigner(eqx.Module):
  """Coordinator wrapper for MPNN-guided Potts design.

  Holds both Aminx (MPNN encoder for unconditional logits) and PottsModel (TRW inference
  engine for global probabilistic design). Provides a unified run_design() API that:

  1. Runs Aminx unconditional scoring → per-position logits (L, 21)
  2. Applies alphabet_map: potts_logits = logits[:, alphabet_map] (gather, not Python loop)
  3. Initializes sequence via argmax or Gumbel sampling
  4. Calls potts.infer_params() → PottsParams (marginals, h, J, W)
  5. Runs n_gibbs_sweeps with optional order averaging
  6. Returns DesignResult with MPNN logits, Potts marginals, final sequence

  Design invariant: alphabet_map must be a JAX array of shape (21,) with dtype int32.
  Index 20 collision (X/gap) is safely identity-mapped; warnings flag non-identity indices 0-19.

  Attributes:
      mpnn: Aminx model instance (ProteinMPNN encoder)
      potts: PottsModel instance (TRW inference engine)
      alphabet_map: Int[Array, '21'] mapping Potts indices to MPNN indices (static=False)
      seed_mode: Literal['unconditional', 'order_averaged'] (static=True)
                 Controls MPNN AR order bias during Gibbs initialization
  """

  mpnn: Aminx = eqx.field(static=False)
  potts: PottsModel = eqx.field(static=False)
  alphabet_map: Int[Array, " 21"] = eqx.field(static=False)
  seed_mode: Literal["unconditional", "order_averaged"] = eqx.field(static=True)

  def __init__(
    self,
    mpnn: Aminx,
    potts: PottsModel,
    alphabet_map: Int[Array, " 21"] | None = None,
    seed_mode: Literal["unconditional", "order_averaged"] = "unconditional",
  ) -> None:
    """Initialize MpnnPottsDesigner.

    Args:
        mpnn: Aminx model instance
        potts: PottsModel instance
        alphabet_map: Shape (21,) int32 array mapping Potts indices to MPNN indices
                      (default: identity permutation from POTTS_TO_MPNN_ALPHABET_MAP)
        seed_mode: 'unconditional' (default, no AR order bias) or 'order_averaged' (future)

    Raises:
        ValueError: If alphabet_map.shape != (21,)
        ValueError: If any index 0-19 is non-identity (logged as warning, not error)
    """
    self.mpnn = mpnn
    self.potts = potts
    self.seed_mode = seed_mode

    # Use default identity map if not provided
    if alphabet_map is None:
      alphabet_map = POTTS_TO_MPNN_ALPHABET_MAP

    # Ensure JAX array
    alphabet_map = jnp.asarray(alphabet_map, dtype=jnp.int32)

    # __post_init__ validation
    if alphabet_map.shape != (21,):
      msg = f"alphabet_map must have shape (21,), got {alphabet_map.shape}"
      raise ValueError(msg)

    # Warn if any index 0-19 is non-identity (risky for seq alignment)
    for i in range(20):
      if alphabet_map[i] != i:
        warnings.warn(
          f"alphabet_map[{i}] = {alphabet_map[i]} (non-identity). "
          f"This breaks amino acid order at position {i}. "
          f"If intentional, document why; otherwise use identity permutation.",
          stacklevel=2,
        )

    self.alphabet_map = alphabet_map

  def run_design(
    self,
    key: PRNGKeyArray,
    coords: Float[Array, "n 37 3"],
    mask: Float[Array, " n"],
    residue_index: Int[Array, " n"],
    chain_index: Int[Array, " n"],
    n_gibbs_sweeps: int = 100,
  ) -> DesignResult:
    """Design a sequence via MPNN-guided Potts inference.

    Step 1: Run MPNN unconditional scoring → per-position logits (L, 21)
    Step 2: Apply alphabet_map: potts_logits = logits[:, alphabet_map]
    Step 3: Initialize sequence from argmax (deterministic for reproducibility)
    Step 4: Call potts.infer_params() → PottsParams (marginals, h, J, W)
    Step 5: Run n_gibbs_sweeps with Gibbs sampling
    Step 6: Return DesignResult with all intermediate outputs

    Args:
        key: JAX random key
        coords: Backbone coordinates (N, 37, 3) padded with B-factor
        mask: Residue mask (N,)
        residue_index: Residue index array (N,)
        chain_index: Chain assignment (N,)
        n_gibbs_sweeps: Number of full Gibbs sweeps (default: 100)

    Returns:
        DesignResult: (mpnn_logits, potts_marginals, gibbs_seq, sequence)
    """
    k_mpnn, k_gibbs = jax.random.split(key)

    # Step 1: Run MPNN unconditional scoring
    # Get per-position logits (L, 21) from Aminx encoder + final projection
    # This is the ONLY cross-pipeline import in potts.designer

    # Call Aminx encoder to get node and edge features
    node_features, _edge_features, _neighbor_indices = self.mpnn(
      coords,
      mask,
      residue_index,
      chain_index,
      prng_key=k_mpnn,
      inference=True,
    )

    # Decode to per-position logits using the w_out projection
    # node_features is shape (L, D), w_out maps D -> 21
    mpnn_logits = jax.vmap(self.mpnn.w_out)(node_features)  # (L, 21)

    # Step 2: Apply alphabet_map to remap MPNN logits to Potts vocabulary
    # potts_logits[i, j] = mpnn_logits[i, alphabet_map[j]]
    # This is a gather operation (efficient in JAX)
    potts_logits = mpnn_logits[:, self.alphabet_map]

    # Step 3: Initialize sequence from argmax of Potts logits (deterministic)
    # argmax gives highest probability amino acid at each position
    initial_seq = jnp.argmax(potts_logits, axis=-1)

    # Step 4: Infer Potts parameters (marginals, h, J, W)
    params = self.potts.infer_params(
      key=k_gibbs,
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
    )

    # Step 5: Run Gibbs sweeps
    # Start from initial_seq, refine via n_gibbs_sweeps full sweeps
    seq_current = initial_seq
    k_sweep = k_gibbs

    for _ in range(n_gibbs_sweeps):
      k_sweep, k_step = jax.random.split(k_sweep)
      seq_current, _ = gibbs_sweep(
        key=k_step,
        seq=seq_current,
        h=params.h,
        j=params.J,
        w=params.W,
        mask=mask,
        site_order=None,  # Use default order [0, 1, ..., n-1]
        neighbor_scale=1.0,
        inverse_temp=1.0,
      )

    # Step 6: Convert final sequence to string
    sequence_str = "".join([MPNN_ALPHABET[int(aa)] for aa in seq_current])

    return DesignResult(
      mpnn_logits=mpnn_logits,
      potts_marginals=params.marginals,
      gibbs_seq=seq_current,
      sequence=sequence_str,
    )
