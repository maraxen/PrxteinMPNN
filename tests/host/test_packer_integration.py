"""Unit and integration tests for Packer integration components.

Verifies:
1. PackerBundle multi-state scaling and structure.
2. InferencePlan decode/sample execution with active packer.
3. Safety validation checks (raising ValueError when packer is active but bundle lacks packer bundle).
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from prxteinmpnn.host.plan import InferencePlan, InferenceComponents, make_inference_plan
from prxteinmpnn.types.bundles import (
    InferenceBundle,
    GeometryBundle,
    ConditioningBundle,
    LigandBundle,
    WaveScheduleBundle,
    PackerBundle,
    PackerResult
)
from prxteinmpnn.inference.sample_autoregressive import SampleResult
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.encodings import EncoderOutput

# Dummy/Mock Packer implementation for testing
class DummyPacker(eqx.Module):
    num_mix: int = eqx.field(static=True, default=3)
    hidden_dim: int = eqx.field(static=True, default=128)

    def __call__(self, key, bundle, config):
        # bundle is a single PackerBundle (no state dimension S)
        L = bundle.sequence.shape[-1]
        return PackerResult(
            mean=jnp.ones((L, 4, self.num_mix)) * 0.5,
            concentration=jnp.ones((L, 4, self.num_mix)) * 1.5,
            mix_logits=jnp.zeros((L, 4, self.num_mix))
        )


def test_packer_bundle_multi_state_scaling():
    """Unit test: Verify PackerBundle multi-state scaling works properly."""
    S, L, M = 3, 10, 5
    
    # Create batched fields representing S states
    sequence = jnp.zeros((L,), dtype=jnp.int32)  # Initially unbatched (L,)
    backbone_coords = jnp.zeros((S, L, 14, 3))
    backbone_mask = jnp.zeros((S, L, 14))
    ligand_coords = jnp.zeros((S, L, M, 3))
    ligand_mask = jnp.zeros((S, L, M))
    ligand_atom_types = jnp.zeros((S, L, M))
    mask = jnp.zeros((S, L))
    residue_index = jnp.zeros((S, L), dtype=jnp.int32)
    chain_labels = jnp.zeros((S, L), dtype=jnp.int32)
    
    # Multi-state PackerBundle
    packer_bundle = PackerBundle(
        sequence=sequence,
        backbone_coords=backbone_coords,
        backbone_mask=backbone_mask,
        ligand_coords=ligand_coords,
        ligand_mask=ligand_mask,
        ligand_atom_types=ligand_atom_types,
        mask=mask,
        residue_index=residue_index,
        chain_labels=chain_labels,
        backbone_noise=jnp.array(0.0)
    )
    
    assert packer_bundle.sequence.shape == (L,)
    assert packer_bundle.backbone_coords.shape == (S, L, 14, 3)
    assert packer_bundle.ligand_coords.shape == (S, L, M, 3)
    assert packer_bundle.mask.shape == (S, L)


def test_inference_plan_decode_with_packer_active():
    """Integration test: Verify InferencePlan.decode executes Packer with proper sequence broadcast and vmap."""
    S, L, M = 2, 8, 4
    
    # 1. Setup mock/dummy components
    packer_model = DummyPacker()
    
    class DummyModel(eqx.Module):
        pass
        
    class DummySpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    # Create plan
    plan = make_inference_plan(DummyModel(), DummySpec(), packer=packer_model)
    assert plan.packer is packer_model
    
    # Mock decode_fn to return a SampleResult
    def mock_decode_fn(key, enc, bundle, config, stage_set):
        return SampleResult(
            sequence=jnp.arange(L, dtype=jnp.int32),
            logits=jnp.zeros((L, 21))
        )
    # Since InferencePlan is a standard dataclass, assign decode_fn directly
    plan.decode_fn = mock_decode_fn
    
    # 2. Setup inputs
    enc = EncoderOutput(
        node_features=jnp.zeros((S, L, 16)),
        edge_features=jnp.zeros((S, L, 5, 16)),
        neighbor_indices=jnp.zeros((S, L, 5), dtype=jnp.int32),
        mask=jnp.ones((S, L))
    )
    
    # Setup dummy geometry/conditioning/ligand/wave using the exact class signatures
    geom = GeometryBundle(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.zeros((S, L), dtype=jnp.int32),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
        n_states=S,
        n_canonical=20,
        n_flat=L
    )
    cond = ConditioningBundle(
        fixed_mask=jnp.zeros(L),
        fixed_tokens=jnp.zeros(L, dtype=jnp.int32),
        bias=jnp.zeros((L, 21)),
        tie_group_map=jnp.zeros((S, L), dtype=jnp.int32),
        state_weights=jnp.ones(S),
        sequence_oh=jnp.zeros((L, 21)),
        ar_mask=jnp.ones((S, L, L))
    )
    ligand = LigandBundle(
        ligand_coords=jnp.zeros((S, L, M, 3)),
        ligand_mask=jnp.zeros((S, L, M)),
        ligand_atom_types=jnp.zeros((S, L, M), dtype=jnp.int32)
    )
    wave = WaveScheduleBundle.empty(L)
    
    # Multi-state PackerBundle
    packer_bundle = PackerBundle(
        sequence=jnp.zeros((L,), dtype=jnp.int32),
        backbone_coords=jnp.zeros((S, L, 14, 3)),
        backbone_mask=jnp.zeros((S, L, 14)),
        ligand_coords=jnp.zeros((S, L, M, 3)),
        ligand_mask=jnp.zeros((S, L, M)),
        ligand_atom_types=jnp.zeros((S, L, M)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.zeros((S, L), dtype=jnp.int32),
        chain_labels=jnp.zeros((S, L), dtype=jnp.int32),
        backbone_noise=jnp.array(0.0)
    )
    
    bundle = InferenceBundle(
        geometry=geom,
        conditioning=cond,
        ligand=ligand,
        wave=wave,
        backbone_noise=jnp.array(0.0),
        packer=packer_bundle
    )
    
    config = InferenceConfig()
    key = jax.random.PRNGKey(42)
    
    # 3. Execute
    result = plan.decode(enc, bundle, key, config)
    
    # 4. Assertions
    assert isinstance(result, SampleResult)
    assert result.packer_result is not None
    assert isinstance(result.packer_result, PackerResult)
    
    # Verify shape of packer_result (stacked over S conformational states)
    # mean shape: (S, L, 4, num_mix)
    assert result.packer_result.mean.shape == (S, L, 4, 3)
    assert result.packer_result.concentration.shape == (S, L, 4, 3)
    assert result.packer_result.mix_logits.shape == (S, L, 4, 3)
    
    # Verify values inside stack
    assert jnp.all(result.packer_result.mean == 0.5)
    assert jnp.all(result.packer_result.concentration == 1.5)


def test_inference_plan_decode_safety_validation():
    """Safety validation test: Verify ValueError is raised when plan.packer is active but bundle.packer is None."""
    packer_model = DummyPacker()
    
    class DummyModel(eqx.Module):
        pass
        
    class DummySpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    plan = make_inference_plan(DummyModel(), DummySpec(), packer=packer_model)
    
    # Create bundle without packer
    geom = GeometryBundle(
        coords=jnp.zeros((1, 5, 4, 3)),
        mask=jnp.ones((1, 5)),
        residue_index=jnp.zeros((1, 5), dtype=jnp.int32),
        chain_index=jnp.zeros((1, 5), dtype=jnp.int32),
        n_states=1,
        n_canonical=20,
        n_flat=5
    )
    cond = ConditioningBundle(
        fixed_mask=jnp.zeros(5),
        fixed_tokens=jnp.zeros(5, dtype=jnp.int32),
        bias=jnp.zeros((5, 21)),
        tie_group_map=jnp.zeros((1, 5), dtype=jnp.int32),
        state_weights=jnp.ones(1),
        sequence_oh=jnp.zeros((5, 21)),
        ar_mask=jnp.ones((1, 5, 5))
    )
    ligand = LigandBundle(
        ligand_coords=jnp.zeros((1, 5, 2, 3)),
        ligand_mask=jnp.zeros((1, 5, 2)),
        ligand_atom_types=jnp.zeros((1, 5, 2), dtype=jnp.int32)
    )
    wave = WaveScheduleBundle.empty(5)
    
    # bundle.packer is None by default
    bundle = InferenceBundle(
        geometry=geom,
        conditioning=cond,
        ligand=ligand,
        wave=wave,
        backbone_noise=jnp.array(0.0),
        packer=None
    )
    
    enc = EncoderOutput(
        node_features=jnp.zeros((1, 5, 16)),
        edge_features=jnp.zeros((1, 5, 3, 16)),
        neighbor_indices=jnp.zeros((1, 5, 3), dtype=jnp.int32),
        mask=jnp.ones((1, 5))
    )
    
    config = InferenceConfig()
    key = jax.random.PRNGKey(42)
    
    with pytest.raises(ValueError, match="Packer model is configured on the InferencePlan, but the InferenceBundle contains no packer bundle data"):
        plan.decode(enc, bundle, key, config)
