from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray  # noqa: PLC2701

from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.host._sampling_helper import (
  RANK_WITH_TEMPERATURE,
  _DEFAULT_DECODING_ORDER_FN,
  _noop_sampling_structure_batch_io,
)
from prxteinmpnn.host.averaging import get_averaged_encodings, make_encoding_sampling_split_fn
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.types import ProteinSequence

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinMPNN


def _internal_sample_averaged(
  spec: SamplingSpecification,
  encoded_feat: tuple,
  keys_arr: PRNGKeyArray,
  sample_fn_with_params: Callable,
  tie_group_map: jnp.ndarray | None,
  num_groups: int | None,
) -> ProteinSequence:
  """Sample mapping over keys for averaged features."""
  decoding_order_keys = jax.random.split(jax.random.key(spec.random_seed + 1), spec.num_samples)

  temperature_array = jnp.asarray(spec.temperature, dtype=jnp.float32)

  def sample_single_sequence(
    key: PRNGKeyArray,
    decoding_order_key: PRNGKeyArray,
    encoded_feat: tuple,
    temperature: float,
  ) -> ProteinSequence:
    """Sample one sequence from averaged features."""
    seq_len = encoded_feat[0].shape[0]
    decoding_order, _ = _DEFAULT_DECODING_ORDER_FN(
      decoding_order_key,
      seq_len,
      tie_group_map,
      num_groups,
    )
    return sample_fn_with_params(key, encoded_feat, decoding_order, temperature=temperature)

  def sample_for_key(k: PRNGKeyArray, dok: PRNGKeyArray) -> ProteinSequence:
    return jax.vmap(
      lambda t: sample_single_sequence(k, dok, encoded_feat, t),
    )(temperature_array)

  vmap_sample_fn = jax.vmap(
    sample_for_key,
    in_axes=(0, 0),
    out_axes=0,
  )
  return vmap_sample_fn(keys_arr, decoding_order_keys)


def _compute_logits_averaged(
  spec: SamplingSpecification,
  averaged_encodings: tuple,
  sampled_sequences: ProteinSequence,
  decode_fn_wrapped: Callable,
) -> jax.Array:
  """Compute logits for the sampled sequences."""
  seq_len = sampled_sequences.shape[-1]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  if spec.average_encoding_mode == "inputs_and_noise":

    def get_logits_local_both(seq: ProteinSequence) -> jax.Array:
      return jax.vmap(lambda s: decode_fn_wrapped(averaged_encodings, s, ar_mask))(seq)

    vmap_logits = jax.vmap(get_logits_local_both)
    logits = vmap_logits(sampled_sequences[0])
    logits = jnp.expand_dims(logits, axis=0)
  else:

    def get_logits_local(seq: ProteinSequence, enc: tuple) -> jax.Array:
      # seq has shape (samples, temps, length) or (samples, length)
      if seq.ndim == 3:
        return jax.vmap(jax.vmap(lambda s: decode_fn_wrapped(enc, s, ar_mask)))(seq)
      return jax.vmap(lambda s: decode_fn_wrapped(enc, s, ar_mask))(seq)

    struct_axis = 0

    vmap_logits = jax.vmap(
      jax.vmap(get_logits_local, in_axes=(0, None)),
      in_axes=(0, (0, 0, struct_axis, struct_axis, struct_axis)),
    )
    logits = vmap_logits(sampled_sequences, averaged_encodings)

  return logits


def _sample_batch_averaged(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  model: "PrxteinMPNN",
  sample_fn: Callable,  # noqa: ARG001
  decode_fn: Callable,  # noqa: ARG001
  batch_idx: int,
  structure_batch_count: int,
  *,
  keys: PRNGKeyArray,
  tie_group_map: jnp.ndarray | None,
  num_groups: int | None,
  create_decode_wrapper: Callable,
) -> tuple[ProteinSequence, jax.Array]:
  """Orchestrate sampling in averaged mode."""
  if batch_idx > 0:
    msg = "Averaged encoding mode does not support multiple batches yet."
    raise NotImplementedError(msg)
  if structure_batch_count > 1:
    msg = "Averaged encoding mode does not support multiple structure batches yet."
    raise NotImplementedError(msg)

  structure_mapping = (
    jnp.asarray(spec.structure_mapping, dtype=jnp.int32)
    if spec.structure_mapping is not None
    else batched_ensemble.mapping
  )

  averaged_encodings = get_averaged_encodings(
    batched_ensemble,
    model,
    spec.backbone_noise,
    spec.noise_batch_size,
    spec.random_seed,
    spec.average_encoding_mode,
    structure_mapping=structure_mapping,
  )

  # Create a new sample_fn with the wrapper
  _, sample_fn_wrapped, decode_fn_wrapped = make_encoding_sampling_split_fn(
    model,
    decode_fn_wrapper=create_decode_wrapper,
  )

  sample_fn_with_params = partial(
    sample_fn_wrapped,
    bias=jnp.asarray(spec.bias, dtype=jnp.float32) if spec.bias is not None else None,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
    multi_state_strategy=spec.multi_state_strategy,
    multi_state_temperature=spec.multi_state_temperature,
  )

  if spec.average_encoding_mode == "inputs_and_noise":
    sampled_sequences = _internal_sample_averaged(
      spec,
      averaged_encodings,
      keys,
      sample_fn_with_params,
      tie_group_map,
      num_groups,
    )
    sampled_sequences = jnp.expand_dims(sampled_sequences, axis=0)
  else:
    struct_axis = 0

    def _call_internal(enc: tuple) -> ProteinSequence:
      return _internal_sample_averaged(
        spec,
        enc,
        keys,
        sample_fn_with_params,
        tie_group_map,
        num_groups,
      )

    vmap_sample_structures = jax.vmap(
      _call_internal,
      in_axes=((0, 0, struct_axis, struct_axis, struct_axis),),
    )
    sampled_sequences = vmap_sample_structures(
      averaged_encodings,
    )

  logits = _compute_logits_averaged(spec, averaged_encodings, sampled_sequences, decode_fn_wrapped)

  if sampled_sequences.ndim == 4:
    # (structures, samples, temps, length) -> (structures, samples, 1, temps, length)
    sampled_sequences = sampled_sequences[:, :, None, :]
    logits = logits[:, :, None, :, :]
  else:
    # (structures, samples, length) -> (structures, samples, 1, 1, length)
    sampled_sequences = sampled_sequences[:, :, None, None, :]
    logits = logits[:, :, None, None, :, :]

  pseudo_perplexity = None
  if spec.compute_pseudo_perplexity:
    one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))
    mask = batched_ensemble.mask
    if mask is None:
      mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    # nll shape: (structures, samples, 1, temps)
    # sum_mask shape: (structures,)
    pseudo_perplexity = jnp.exp(nll / jnp.sum(mask, axis=-1)[:, None, None, None])
    batch_idx_j = jnp.asarray(batch_idx, dtype=jnp.int32)
    batch_cnt_j = jnp.asarray(structure_batch_count, dtype=jnp.int32)
    jax.experimental.io_callback(
      _noop_sampling_structure_batch_io,
      None,
      jax.lax.stop_gradient(batch_idx_j),
      jax.lax.stop_gradient(batch_cnt_j),
      ordered=False,
    )
    return sampled_sequences, logits, pseudo_perplexity

  batch_idx_j = jnp.asarray(batch_idx, dtype=jnp.int32)
  batch_cnt_j = jnp.asarray(structure_batch_count, dtype=jnp.int32)
  jax.experimental.io_callback(
    _noop_sampling_structure_batch_io,
    None,
    jax.lax.stop_gradient(batch_idx_j),
    jax.lax.stop_gradient(batch_cnt_j),
    ordered=False,
  )
  return sampled_sequences, logits, None

