"""Conditional scoring kernel using InferenceBundle."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.encodings import EncoderOutput


from prxteinmpnn.types.stages import StageSet

# TODO: by default, 1 - jnp.eye should be the autoregressive mask

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Compute teacher-forced conditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)

    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand
    wave = bundle.wave

    S = geo.n_states

    # Encode using make_encode_fn, which controls scan vs vmap over S states
    encode_fn = make_encode_fn(model, use_rolling_state=config.use_rolling_state)
    enc = encode_fn(bundle, k_enc, config)

    def decode_one(nb: jax.Array, eb: jax.Array, nei: jax.Array, mk: jax.Array, arm: jax.Array, oh: jax.Array):
        if stage_set.decode_step is not None:
            return stage_set.decode_step(nb, eb, nei, mk, arm, oh, key=k_dec, inference=config.inference)
        return model.decoder.call_conditional(
            nb, eb, nei, mk, arm, oh, model.w_s_embed.weight, inference=config.inference, key=k_dec
        )

    # For conditional scoring, wave provides the AR masks, or we assume a full mask?
    seq_oh_stack = jnp.broadcast_to(cond.sequence_oh[None, ...], (S, *cond.sequence_oh.shape))

    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, geo.mask, cond.ar_mask, seq_oh_stack
    )

    logits_stack = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

    return stage_set.logit_transform(logits_stack, bias=cond.bias)
