"""Debug autoregressive logits to understand why sampling produces mostly 13s."""

import jax
import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import random_decoding_order


def test_logits_distribution():
  """Debug the logits produced during autoregressive sampling."""
  # Load
  protein_tuple = next(parse_input("tests/data/1ubq.pdb"))
  protein = Protein.from_tuple(protein_tuple)
  model = load_model()

  # Setup
  key = jax.random.key(42)
  decoding_order, key = random_decoding_order(key, len(protein.aatype))
  ar_mask = generate_ar_mask(decoding_order, None, None, None)

  print("=" * 80)
  print("UNCONDITIONAL LOGITS (baseline)")
  print("=" * 80)

  # Get unconditional logits
  _, logits_uncond = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="unconditional",
  )

  # Analyze first 5 positions
  for pos in range(5):
    logits_pos = logits_uncond[pos, :20]
    probs = jax.nn.softmax(logits_pos)
    top_aa = jnp.argmax(probs)
    top_prob = probs[top_aa]
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

    print(f"\nPosition {pos}:")
    print(f"  Top AA: {top_aa}, prob: {top_prob:.3f}")
    print(f"  Entropy: {entropy:.3f}")
    print(f"  Top 5 AAs: {jnp.argsort(probs)[-5:][::-1]}")
    print(f"  Top 5 probs: {jnp.sort(probs)[-5:][::-1]}")

  print("\n" + "=" * 80)
  print("AUTOREGRESSIVE LOGITS")
  print("=" * 80)

  # Get autoregressive logits
  _, logits_ar = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="autoregressive",
    prng_key=key,
    ar_mask=ar_mask,
    temperature=jnp.array(0.1, dtype=jnp.float32),
  )

  # Analyze same positions
  for pos in range(5):
    logits_pos = logits_ar[pos, :20]
    probs = jax.nn.softmax(logits_pos)
    top_aa = jnp.argmax(probs)
    top_prob = probs[top_aa]
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

    print(f"\nPosition {pos}:")
    print(f"  Top AA: {top_aa}, prob: {top_prob:.3f}")
    print(f"  Entropy: {entropy:.3f}")
    print(f"  Top 5 AAs: {jnp.argsort(probs)[-5:][::-1]}")
    print(f"  Top 5 probs: {jnp.sort(probs)[-5:][::-1]}")


if __name__ == "__main__":
  test_logits_distribution()
