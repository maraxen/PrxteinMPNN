"""Reverse-mode Jacobian implementation for efficient mutation scoring.

While forward-mode Jacobian (CatJac) computes sensitivity of *all* outputs
to perturbations, the reverse-mode gradient is much more efficient when we
only care about the gradient of a single scalar (e.g., the pooled log-likelihood
score). This allows computing the sensitivity of the score to all possible mutations
in a single backward pass.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from prxteinmpnn.utils.types import (
    ProteinSequence,
    AlphaCarbonMask,
)
from prxteinmpnn.sampling.conditional_logits import (
    make_encoding_conditional_logits_split_fn,
)


def make_reverse_jacobian_score_fn(model):
    """Create a function to compute reverse-mode gradients of the score.

    The returned function `grad_fn` takes the pre-computed `encoding` and
    a one-hot `sequence` array, and returns a gradient of identical shape
    (L, 21), representing `∂score / ∂one_hot[i, a]`.

    For a given mutation from WT amino acid `w` to mutant `m` at position `i`,
    the first-order approximation of the change in score is given by:
        Δscore ≈ grad[i, m] - grad[i, w]
    """
    encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

    @jax.jit
    def compute_grad_score(
        encoding: tuple,
        one_hot_sequence: Float[Array, "L 21"],
        ar_mask: Float[Array, "L L"],
    ) -> Float[Array, "L 21"]:
        """Compute ∂score/∂one_hot for a given sequence encoding."""
        # Extract the mask from the encoding tuple (4th element)
        mask = encoding[3]

        def score_fn(oh_2d):
            logits = decode_fn(encoding, oh_2d, ar_mask=ar_mask)
            # Only typical 20 amino acids contribute to the actual score probability
            log_p = jax.nn.log_softmax(logits, axis=-1)[..., :20]
            per_res_nll = -(oh_2d[..., :20] * log_p).sum(-1)
            # Mask and average
            return (per_res_nll * mask).sum() / (mask.sum() + 1e-8)

        # Single reverse-mode backward pass
        return jax.grad(score_fn)(one_hot_sequence)

    return encode_fn, compute_grad_score
