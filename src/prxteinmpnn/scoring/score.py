"""Score a given sequence on a structure using the ProteinMPNN model."""

from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model._shared import apply_multistate_to_all_logits
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from prxteinmpnn.bundles import LigandBundle, ProteinBundle
from prxteinmpnn.pipeline_registry import StageSet, make_geometric_mean_transform, resolve_hook
from prxteinmpnn.protocols import ModelProtocol, ScoreFn, StateVmapExactScoreFn
from prxteinmpnn.registry import (
  assert_known_multistate_mode,
  combine_strategy_to_index,
  multistate_mode_descriptor,
)
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  Logits,
  OneHotProteinSequence,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

SCORE_EPS = 1e-8


def score_sequence_with_encoding(
  model: ModelProtocol,
  sequence: ProteinSequence,
  encoding: tuple,
  tie_group_map: jax.Array | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
) -> tuple[Float, Logits, DecodingOrder]:
  """Score a sequence on a structure using pre-computed encodings."""
  _, _, decode_fn = make_encoding_sampling_split_fn(model)

  if sequence.ndim == 1:
    sequence = jax.nn.one_hot(sequence, num_classes=21)

  seq_len = sequence.shape[0]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  logits = decode_fn(encoding, sequence, ar_mask)
  if tie_group_map is not None:
    strategy_idx = jnp.asarray(
      combine_strategy_to_index(multi_state_strategy),
      dtype=jnp.int32,
    )
    logits = apply_multistate_to_all_logits(
      logits,
      tie_group_map,
      strategy_idx,
      multi_state_temperature,
    )

  log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
  score = -(sequence[..., :20] * log_probability).sum(-1)
  mask = encoding[3]  # mask is the 4th element in the encoding tuple
  masked_score_sum = (score * mask).sum(-1)
  mask_sum = mask.sum() + SCORE_EPS

  return masked_score_sum / mask_sum, logits, jnp.arange(seq_len)


def _make_score_fn_state_vmap_exact(
  model: ModelProtocol,
  *,
  inference: bool,
) -> StateVmapExactScoreFn:
  if inference and isinstance(model, eqx.Module):
    model = eqx.nn.inference_mode(model, value=True)

  is_lig = model.capabilities.is_ligand_model
  n_emb = int(model.w_s_embed.num_embeddings)

  def score_sequence_core_inner(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    *,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    state_flat_rows: jax.Array,
    n_flat_int: int,
    structure_mapping: jax.Array | None,
    tie_group_map: jax.Array | None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"],
    multi_state_temperature: Float,
    state_weights: jax.Array,
    y: jax.Array | None,
    y_t: jax.Array | None,
    y_m: jax.Array | None,
    ar_mask_stack: jax.Array | None,
    bias_flat: jax.Array | None,
    states_chunk_size: int = 0,
    stage_set: StageSet | None = None,
  ) -> tuple[Float, Logits, DecodingOrder]:

    strategy_idx = jnp.int32(combine_strategy_to_index(multi_state_strategy))
    oh = jax.nn.one_hot(sequence, n_emb) if sequence.ndim == 1 else sequence
    seq_stack = gather_flat_to_stack(oh, state_flat_rows)
    s_dim, p_dim = mask.shape[0], mask.shape[1]
    arm = (
      jnp.zeros((s_dim, p_dim, p_dim), dtype=jnp.int32)
      if ar_mask_stack is None
      else ar_mask_stack
    )

    scs_kw: dict[str, int] = {}
    if is_lig and states_chunk_size > 0:
      scs_kw["states_chunk_size"] = int(states_chunk_size)

    logit_transform_fn = stage_set.resolve_all()["logit_transform_fn"] if stage_set is not None else None

    if is_lig:
      logits = model.score_conditional(  # type: ignore[union-attr]
        prng_key,
        coords,
        mask,
        residue_index,
        chain_index,
        y,
        y_t,
        y_m,
        seq_stack,
        arm,
        state_flat_rows,
        n_flat_int,
        tie_group_map=tie_group_map,
        multi_state_strategy_idx=strategy_idx,
        state_weights=state_weights,
        state_mapping=structure_mapping,
        bias_flat=bias_flat,
        inference=True,
        logit_transform_fn=logit_transform_fn,
        **scs_kw,
      )
    else:
      logits = model.score_conditional(
        prng_key,
        coords,
        mask,
        residue_index,
        chain_index,
        seq_stack,
        arm,
        state_flat_rows,
        n_flat_int,
        tie_group_map=tie_group_map,
        multi_state_strategy_idx=strategy_idx,
        state_weights=state_weights,
        state_mapping=structure_mapping,
        bias_flat=bias_flat,
        inference=True,
        logit_transform_fn=logit_transform_fn,
      )

    if logit_transform_fn is not None:
      s_dim = mask.shape[0]
      logits_s = jnp.broadcast_to(logits[jnp.newaxis], (s_dim,) + logits.shape)
      logits = scatter_stack_to_flat(logits_s, state_flat_rows, n_flat_int)

    mask_flat = scatter_stack_to_flat(
      mask[..., jnp.newaxis],
      state_flat_rows,
      n_flat_int,
    )[..., 0]
    decoding_order = jnp.arange(n_flat_int, dtype=jnp.int32)
    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    score = -(oh[..., :20] * log_probability).sum(-1)
    masked_score_sum = (score * mask_flat).sum(-1)
    mask_sum = mask_flat.sum() + SCORE_EPS
    return masked_score_sum / mask_sum, logits, decoding_order

  if is_lig:
    score_sequence_core = score_sequence_core_inner
  else:

    @partial(jax.jit, static_argnames=("multi_state_strategy", "n_flat_int", "stage_set", "multi_state_temperature"))
    def score_sequence_core(
      prng_key: PRNGKeyArray,
      sequence: ProteinSequence | OneHotProteinSequence,
      *,
      coords: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      state_flat_rows: jax.Array,
      n_flat_int: int,
      structure_mapping: jax.Array | None,
      tie_group_map: jax.Array | None,
      multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"],
      multi_state_temperature: Float,
      state_weights: jax.Array,
      y: jax.Array | None,
      y_t: jax.Array | None,
      y_m: jax.Array | None,
      ar_mask_stack: jax.Array | None,
      bias_flat: jax.Array | None,
      states_chunk_size: int = 0,
      stage_set: StageSet | None = None,
    ) -> tuple[Float, Logits, DecodingOrder]:
      del states_chunk_size
      return score_sequence_core_inner(
        prng_key,
        sequence,
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        state_flat_rows=state_flat_rows,
        n_flat_int=n_flat_int,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
        multi_state_temperature=multi_state_temperature,
        state_weights=state_weights,
        y=y,
        y_t=y_t,
        y_m=y_m,
        ar_mask_stack=ar_mask_stack,
        bias_flat=bias_flat,
        states_chunk_size=0,
        stage_set=stage_set,
      )

  def score_sequence(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
    *,
    coords: jax.Array | None = None,
    mask: jax.Array | None = None,
    residue_index: jax.Array | None = None,
    chain_index: jax.Array | None = None,
    state_flat_rows: jax.Array | None = None,
    n_flat: int | None = None,
    state_weights: jax.Array | None = None,
    y: jax.Array | None = None,
    y_t: jax.Array | None = None,
    y_m: jax.Array | None = None,
    ar_mask_stack: jax.Array | None = None,
    bias_flat: jax.Array | None = None,
    states_chunk_size: int = 0,
    multistate_stack: ProteinBundle | None = None,
    ligand_stack: LigandBundle | None = None,
    stage_set: StageSet | None = None,
    **kwargs: object,
  ) -> tuple[Float, Logits, DecodingOrder]:
    del kwargs, structure_coordinates, mask, residue_index, chain_index, backbone_noise, ar_mask
    if multistate_stack is not None:
      coords = multistate_stack.coords
      mask = multistate_stack.mask
      residue_index = multistate_stack.residue_index
      chain_index = multistate_stack.chain_index
      state_flat_rows = multistate_stack.state_flat_rows
      n_flat = int(multistate_stack.n_flat)
    if ligand_stack is not None:
      if y is not None or y_t is not None or y_m is not None:
        msg = "state_vmap_exact scoring: pass either ligand_stack= or y_stack= / y_t_stack= / y_m_stack=, not both"
        raise ValueError(msg)
      y = ligand_stack.y
      y_t = ligand_stack.y_t
      y_m = ligand_stack.y_m
    if coords is None or mask is None or residue_index is None or chain_index is None:
      msg = "state_vmap_exact scoring requires stack tensors or multistate_stack="
      raise ValueError(msg)
    if state_flat_rows is None or n_flat is None or state_weights is None:
      msg = "state_vmap_exact scoring requires state_flat_rows=, n_flat=, state_weights="
      raise ValueError(msg)
    if is_lig and (y is None or y_t is None or y_m is None):
      msg = "PrxteinLigandMPNN state_vmap_exact scoring requires y_stack, y_t_stack, y_m_stack kwargs or ligand_stack="
      raise ValueError(msg)
    return score_sequence_core(
      prng_key,
      sequence,
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      state_flat_rows=state_flat_rows,
      n_flat_int=n_flat,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      multi_state_strategy=multi_state_strategy,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      y=y,
      y_t=y_t,
      y_m=y_m,
      ar_mask_stack=ar_mask_stack,
      bias_flat=bias_flat,
      states_chunk_size=states_chunk_size,
      stage_set=stage_set,
    )

  return cast("StateVmapExactScoreFn", score_sequence)


def make_score_fn(
  model: ModelProtocol,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  inference: bool = True,  # noqa: FBT001, FBT002
  multistate_mode: Literal["flat", "state_vmap_exact"] = "flat",
) -> ScoreFn | StateVmapExactScoreFn:
  """Create a function to score a sequence on a structure using PrxteinMPNN.

  Args:
    model: Protein or Ligand Equinox checkpoint.
    decoding_order_fn: Decoding order (ignored for ``state_vmap_exact``).
    _num_encoder_layers: Deprecated; ignored.
    _num_decoder_layers: Deprecated; ignored.
    inference: Use ``eqx.nn.inference_mode`` when True.
    multistate_mode: ``flat`` (default) full-graph conditional forward, or ``state_vmap_exact``
      stacked encode + scatter. The latter requires keyword stack arguments on the returned
      function (``coords_stack=``, ``mask_stack=``, …).

  Returns:
    JIT scoring function.

  """
  del _num_encoder_layers, _num_decoder_layers

  assert_known_multistate_mode(multistate_mode)
  ms_route = multistate_mode_descriptor(multistate_mode)

  if ms_route.uses_stacked_exact_score_factory:
    return cast(
      "StateVmapExactScoreFn",
      _make_score_fn_state_vmap_exact(model, inference=inference),
    )

  if inference and isinstance(model, eqx.Module):
    model = eqx.nn.inference_mode(model, value=True)

  try:
    n_aa = int(model.w_s_embed.num_embeddings)
  except AttributeError:
    n_aa = 21
  supports_multi_state_temperature = model.capabilities.accepts_multi_state_temperature

  @partial(jax.jit, static_argnames=("multi_state_strategy",))
  def score_sequence(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
    **kwargs: object,
  ) -> tuple[Float, Logits, DecodingOrder]:
    """Score a sequence on a structure using the ProteinMPNN model.

    Args:
      prng_key: JAX random key.
      sequence: Protein sequence (integer indices or one-hot).
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Optional noise for backbone coordinates.
      ar_mask: Optional custom autoregressive mask.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                   When provided (multi-state mode), prevents cross-structure
                   neighbors to avoid information leakage between conformational states.
      tie_group_map: Optional (N,) array mapping each position to tied groups.
                    When provided, logits are combined within groups before scoring.
      multi_state_strategy: Strategy for combining logits across tied positions.
      multi_state_temperature: Temperature for geometric_mean strategy.


    Returns:
      Tuple of (average score, logits, decoding order).

    Example:
      >>> score, logits, order = score_sequence(
      ...     key, seq, coords, mask, res_idx, chain_idx
      ... )

    """
    del kwargs
    decoding_order, prng_key = decoding_order_fn(prng_key, sequence.shape[0], None, None)
    autoregressive_mask = (
      cast("Callable", generate_ar_mask)(decoding_order) if ar_mask is None else ar_mask
    )

    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, num_classes=n_aa)

    # Run model in conditional mode (scoring a given sequence)
    if supports_multi_state_temperature:
      _, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach="conditional",
        prng_key=prng_key,
        ar_mask=autoregressive_mask,
        one_hot_sequence=sequence,
        temperature=0.0,  # Not used in conditional mode
        bias=None,  # No bias in scoring
        backbone_noise=backbone_noise,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
        multi_state_temperature=multi_state_temperature,
      )
    else:
      _, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach="conditional",
        prng_key=prng_key,
        ar_mask=autoregressive_mask,
        one_hot_sequence=sequence,
        temperature=0.0,  # Not used in conditional mode
        bias=None,  # No bias in scoring
        backbone_noise=backbone_noise,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
      )

    # Compute score from logits
    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    score = -(sequence[..., :20] * log_probability).sum(-1)
    masked_score_sum = (score * mask).sum(-1)
    mask_sum = mask.sum() + SCORE_EPS

    return masked_score_sum / mask_sum, logits, decoding_order

  return cast("ScoreFn", score_sequence)


make_score_sequence = make_score_fn
