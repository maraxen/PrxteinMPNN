"""Score a given sequence on a structure using the ProteinMPNN model."""

from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from aminx.inference import score_conditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.types.protocols import ModelProtocol, ScoreFn
from aminx.utils.autoregression import generate_ar_mask
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)

SCORE_EPS = 1e-8


def make_score_fn(
  model: ModelProtocol,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  inference: bool = True,  # noqa: FBT001, FBT002
) -> ScoreFn:
  """Create a function to score a sequence on a structure using Aminx.

  Args:
    model: Protein or Ligand Equinox checkpoint.
    decoding_order_fn: Decoding order.
    inference: Use ``eqx.nn.inference_mode`` when True.

  Returns:
    JIT scoring function.

  """
  del _num_encoder_layers, _num_decoder_layers

  if inference and isinstance(model, eqx.Module):
    model = eqx.nn.inference_mode(model, value=True)

  n_aa = (
    int(getattr(model, "w_s_embed", None).num_embeddings) if hasattr(model, "w_s_embed") else 21
  )

  @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state"))
  def score_sequence(
    prng_key: jax.Array,
    sequence: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    backbone_noise: float | None = None,
    ar_mask: jax.Array | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean", "geometric_mean", "product",
    ] = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
    state_weights: jax.Array | None = None,
    bias: jax.Array | None = None,
    use_rolling_state: bool = False,
    ligand_coords: jax.Array | None = None,
    ligand_atom_types: jax.Array | None = None,
    ligand_mask: jax.Array | None = None,
    **kwargs,  # Accept but ignore extra kwargs (e.g., _k_neighbors for backward compat)
  ) -> tuple[jax.Array, jax.Array, jax.Array]:

    L = sequence.shape[0]
    S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

    decoding_order, prng_key = decoding_order_fn(prng_key, L, None, None)
    if ar_mask is None:
      ar_mask_single = generate_ar_mask(decoding_order)
    else:
      ar_mask_single = ar_mask[0] if ar_mask.ndim == 3 else ar_mask

    bundle, config = build_inference_bundle(
      coords=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=sequence,
      backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
      ar_mask=ar_mask_single,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      state_weights=state_weights,
      bias=bias,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
      mode="score_conditional",
      inference=True,
    )
    stage_set = make_stage_set(
      strategy=multi_state_strategy,
      strategy_temperature=multi_state_temperature,
      state_weights=state_weights,
    )

    logits = score_conditional.kernel(model, prng_key, bundle, config, stage_set)

    # Compute score
    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    score = -(sequence[..., :20] * log_probability).sum(-1)

    # Use average mask across states for scoring? Or just mask[0]?
    # In modernized architecture, we usually score against the combined logits.
    # We use the first state's mask as a proxy for the system mask.
    mask_flat = mask[0]
    masked_score_sum = (score * mask_flat).sum(-1)
    mask_sum = mask_flat.sum() + 1e-8  # epsilon guards the division below

    return masked_score_sum / mask_sum, logits, decoding_order

  return cast("ScoreFn", score_sequence)


make_score_sequence = make_score_fn


def score(
  prng_key: PRNGKeyArray,
  model: ModelProtocol,
  structure_coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array | None = None,
  backbone_noise: float | None = None,
  ar_mask: jax.Array | None = None,
  structure_mapping: jax.Array | None = None,
  tie_group_map: jax.Array | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
  state_weights: jax.Array | None = None,
  bias: jax.Array | None = None,
  use_rolling_state: bool = False,
  ligand_coords: jax.Array | None = None,
  ligand_atom_types: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Score a sequence on a structure using the default scoring function.

  This is a convenience wrapper around `make_score_fn`.

  Args:
      prng_key: JAX random key.
      model: A Aminx Equinox model instance.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      sequence: Protein sequence to score.
      backbone_noise: Noise level for backbone coordinates.
      ar_mask: Autoregressive mask for scoring.
      structure_mapping: Mapping between structures.
      tie_group_map: Groups of tied positions.
      multi_state_strategy: How to combine multi-state logits.
      multi_state_temperature: Temperature for multi-state combination.
      state_weights: Weights for each state.
      bias: Sequence bias.
      use_rolling_state: Use rolling state scan vs vmap.
      ligand_coords: Ligand coordinates.
      ligand_atom_types: Ligand atom types.
      ligand_mask: Ligand atom mask.

  Returns:
      Tuple of (masked average score, logits, decoding order).

  """
  score_fn = make_score_fn(model)
  return cast(
    "tuple[jax.Array, jax.Array, jax.Array]",
    score_fn(
      prng_key=prng_key,
      sequence=sequence,
      structure_coordinates=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      backbone_noise=backbone_noise,
      ar_mask=ar_mask,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      multi_state_strategy=multi_state_strategy,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      bias=bias,
      use_rolling_state=use_rolling_state,
      ligand_coords=ligand_coords,
      ligand_atom_types=ligand_atom_types,
      ligand_mask=ligand_mask,
    ),
  )
