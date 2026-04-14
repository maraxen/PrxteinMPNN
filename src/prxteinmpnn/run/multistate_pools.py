"""Generation of diverse design pools for multistate structures."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.io.designs import DesignArrayRecordWriter, stream_design_to_host
from prxteinmpnn.run.prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.sampling.sample import make_sample_sequences

if TYPE_CHECKING:
  from prxteinmpnn.utils.data_structures import Protein

logger = logging.getLogger(__name__)

def prepare_sidechain_context(
  protein: Protein,
  max_context_atoms: int = 32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Extract sidechain heavy atoms and format for LigandMPNN."""
  L = protein.coordinates.shape[1]
  M = max_context_atoms

  if hasattr(protein, "xyz_37") and protein.xyz_37 is not None:
    sc_coords = protein.xyz_37[:, 5:, :]
    sc_mask = protein.xyz_37_m[:, 5:]

    Y = jnp.zeros((L, M, 3))
    Y_t = jnp.zeros((L, M), dtype=jnp.int32)
    Y_m = jnp.zeros((L, M))

    limit = min(sc_coords.shape[1], M)
    Y = Y.at[:, :limit, :].set(sc_coords[:, :limit, :])
    Y_m = Y_m.at[:, :limit].set(sc_mask[:, :limit])
    Y_t = Y_t.at[:, :limit].set(jnp.where(sc_mask[:, :limit], 6, 0))
    return Y, Y_t, Y_m

  return jnp.zeros((L, M, 3)), jnp.zeros((L, M), dtype=jnp.int32), jnp.zeros((L, M))

def merge_contexts(
  Y1: jnp.ndarray, Yt1: jnp.ndarray, Ym1: jnp.ndarray,
  Y2: jnp.ndarray, Yt2: jnp.ndarray, Ym2: jnp.ndarray,
  max_total: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Concatenate two sets of context atoms."""
  Y = jnp.concatenate([Y1, Y2], axis=1)
  Y_t = jnp.concatenate([Yt1, Yt2], axis=1)
  Y_m = jnp.concatenate([Ym1, Ym2], axis=1)
  return Y[:, :max_total, :], Y_t[:, :max_total], Y_m[:, :max_total]

class DesignPoolRunner:
  """Runner for generating diverse design pools across multiple states and contexts.
  
  Uses massive parallelization (vmap + lax.map) to saturate GPUs.
  """

  def __init__(self, spec: SamplingSpecification, output_path: str):
    self.spec = spec
    self.protein_iterator, self.model = prep_protein_stream_and_model(spec)
    self.writer = DesignArrayRecordWriter(output_path)
    self.sampler_fn = make_sample_sequences(
      model=self.model,
      sampling_strategy=spec.sampling_strategy,
    )

  def run_all_pools(self):
    """Run generation for all 4 context pools and all weighting/combination strategies."""
    for batched_ensemble in self.protein_iterator:
      # Context 1: Backbone Only
      self.generate_pool(batched_ensemble, "BackboneOnly")

      # Context 2: Backbone + Ligand
      if hasattr(batched_ensemble, "Y") and batched_ensemble.Y is not None:
        self.generate_pool(batched_ensemble, "BackboneLigand")

      # Context 3: Backbone + Sidechains
      self.generate_pool(batched_ensemble, "BackboneSidechain")

      # Context 4: Full Context
      if hasattr(batched_ensemble, "Y") and batched_ensemble.Y is not None:
        self.generate_pool(batched_ensemble, "FullContext")

  def generate_pool(self, protein: Protein, pool_type: str):
    """Generate designs for a specific context pool."""
    L = protein.coordinates.shape[1]

    if pool_type == "BackboneOnly":
      Y, Y_t, Y_m = jnp.zeros((L, 1, 3)), jnp.zeros((L, 1), dtype=jnp.int32), jnp.zeros((L, 1))
    elif pool_type == "BackboneLigand":
      Y, Y_t, Y_m = protein.Y, protein.Y_t, protein.Y_m
    elif pool_type == "BackboneSidechain":
      Y, Y_t, Y_m = prepare_sidechain_context(protein)
    elif pool_type == "FullContext":
      Y_sc, Yt_sc, Ym_sc = prepare_sidechain_context(protein)
      Y, Y_t, Y_m = merge_contexts(protein.Y, protein.Y_t, protein.Y_m, Y_sc, Yt_sc, Ym_sc)

    # Weighting strategies
    num_states = protein.coordinates.shape[0]
    weight_strategies = {
      "Uniform": jnp.ones(num_states) / num_states,
      "Manual": jnp.ones(num_states) / num_states,
    }

    # Combination strategies
    comb_strategies = ["arithmetic_mean", "geometric_mean", "product"]

    for w_name, weights in weight_strategies.items():
      for comb_alg in comb_strategies:
        self._sample_massively(
          protein, Y, Y_t, Y_m, weights, comb_alg, pool_type, w_name,
        )

  def _sample_massively(
    self, protein: Protein, Y, Y_t, Y_m, weights, comb_alg, pool_type, weight_strategy,
  ):
    """Execute massively parallel sampling using vmap + lax.map + io_callback."""
    num_samples = self.spec.num_samples
    batch_size = self.spec.samples_batch_size  # Usually 16-64 to saturate GPU
    num_batches = (num_samples + batch_size - 1) // batch_size

    key = jax.random.key(self.spec.random_seed)

    # Define metadata schema for host callback
    metadata = {
      "pool_type": pool_type,
      "state_mapping": protein.mapping.tolist() if protein.mapping is not None else [],
      "weight_strategy": weight_strategy,
      "combination_algorithm": comb_alg,
      "structure_ids": getattr(protein, "structure_ids", []),
    }

    # Helper for host callback
    def host_callback(sequences, logits, scores, state_weights):
      # sequences: [B, L]
      # logits: [B, L, 21]
      for i in range(sequences.shape[0]):
        payload = {
          "sequence": sequences[i],
          "logits": logits[i],
          "scores": scores[i],
          "state_weights": state_weights[i],
          "metadata": metadata,
        }
        stream_design_to_host(self.writer, **payload)

    # Vmapped sampler for GPU saturation
    vmapped_sampler = jax.vmap(
      partial(
        self.sampler_fn,
        structure_coordinates=protein.coordinates,
        mask=protein.mask,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        Y=Y, Y_t=Y_t, Y_m=Y_m,
        state_weights=weights,
        state_mapping=protein.mapping,
        multi_state_strategy=comb_alg,
        temperature=self.spec.temperature[0] if isinstance(self.spec.temperature, (list, jnp.ndarray)) else self.spec.temperature,
      ),
      in_axes=(0,),
    )

    @jax.jit
    def run_batch(batch_key):
      keys = jax.random.split(batch_key, batch_size)
      seqs, logits, _ = vmapped_sampler(keys)

      # Dummy scores for now
      scores = jnp.zeros((batch_size, 1))
      weights_broadcast = jnp.broadcast_to(weights, (batch_size, weights.shape[0]))

      # Use io_callback to stream out of the JIT loop
      jax.experimental.io_callback(
        host_callback,
        None, # Returns None
        seqs, logits, scores, weights_broadcast,
      )

    # Loop over batches using lax.map for memory efficiency
    batch_keys = jax.random.split(key, num_batches)
    jax.lax.map(run_batch, batch_keys)

  def close(self):
    self.writer.close()
