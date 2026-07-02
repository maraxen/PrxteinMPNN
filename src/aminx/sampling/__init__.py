"""Sampling utilities for PrXteinMPNN."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import jax

from aminx.sampling.conditional_logits import (
  make_conditional_logits_fn,
  make_encoding_conditional_logits_split_fn,
)
from aminx.sampling.sample import make_sample_sequences
from aminx.utils import ste

if TYPE_CHECKING:
  from aminx.model import Aminx

__all__ = [
  "make_conditional_logits_fn",
  "make_encoding_conditional_logits_split_fn",
  "make_sample_sequences",
  "sample",
  "ste",
]


def sample(
  prng_key: jax.Array,
  model: Aminx,
  structure_coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  bias: jax.Array | None = None,
  fixed_mask: jax.Array | None = None,
  fixed_tokens: jax.Array | None = None,
  backbone_noise: float | None = None,
  temperature: float = 1.0,
  tie_group_map: jax.Array | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
  state_weights: jax.Array | None = None,
  use_rolling_state: bool = False,
  ligand_coords: jax.Array | None = None,
  ligand_atom_types: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
  inference_only: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sample sequences from a structure using the default temperature sampler.

  This is a convenience wrapper around `make_sample_sequences`.

  Args:
    prng_key: JAX random key.
    model: A Aminx Equinox model instance.
    structure_coordinates: Atomic coordinates (N, 4, 3).
    mask: Alpha carbon mask indicating valid residues.
    residue_index: Residue indices.
    chain_index: Chain indices.
    bias: Sequence bias.
    fixed_mask: Mask for fixed positions.
    fixed_tokens: Fixed token values.
    backbone_noise: Noise level for backbone coordinates.
    temperature: Sampling temperature.
    tie_group_map: Groups of tied positions.
    multi_state_strategy: How to combine multi-state logits.
    multi_state_temperature: Temperature for multi-state combination.
    state_weights: Weights for each state.
    use_rolling_state: Use rolling state scan vs vmap.
    ligand_coords: Ligand coordinates.
    ligand_atom_types: Ligand atom types.
    ligand_mask: Ligand atom mask.
    inference_only: When True, the AR wave axis uses lax.while_loop instead
      of lax.scan -- much faster to compile, not reverse-mode differentiable.
      Safe for any sampling use (AR decoding is never used in a training/grad
      path in aminx). Leave False if unsure.

  Returns:
    Tuple of (sampled sequence, logits, decoding order).

  """
  sampler = make_sample_sequences(model, sampling_strategy="temperature")
  return cast(
    "tuple[jax.Array, jax.Array, jax.Array]",
    sampler(
      prng_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      bias=bias,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
      backbone_noise=backbone_noise,
      temperature=temperature,
      tie_group_map=tie_group_map,
      multi_state_strategy=multi_state_strategy,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      use_rolling_state=use_rolling_state,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
      inference_only=inference_only,
    ),
  )
