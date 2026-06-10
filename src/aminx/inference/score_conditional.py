"""Conditional scoring kernel using InferenceBundle."""

import jax
from jaxtyping import PRNGKeyArray

from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.encode import make_encode_fn
from aminx.tiling.iterator import VmapIterator
from aminx.types.arrays import Logits
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import StageSet


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
  *,
  atom_37: jax.Array | None = None,
  atom_37_mask: jax.Array | None = None,
  chain_mask: jax.Array | None = None,
) -> Logits:
  """Compute teacher-forced conditional logits."""
  import equinox as eqx
  import jax

  k_enc, k_dec = jax.random.split(prng_key)

  # Inject side-chain context into bundle if provided
  if atom_37 is not None or atom_37_mask is not None:
    bundle = eqx.tree_at(
      lambda b: (b.geometry.atom_37, b.geometry.atom_37_mask),
      bundle,
      (atom_37, atom_37_mask),
      is_leaf=lambda x: x is None,
    )

  # Encode using vmap strategy (parallel over S states)
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Use ConditionalDecode mode class for conditional decoding
  # (hardcoded Vmap iterator; STE uses this internally)
  decode_fn = ConditionalDecode(model=model, state_iterator=VmapIterator())
  return decode_fn(
    key=k_dec,
    enc=enc,
    bundle=bundle,
    config=config,
    stage_set=stage_set,
  )
