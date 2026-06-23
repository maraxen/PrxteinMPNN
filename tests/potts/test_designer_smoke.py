"""Smoke test for MpnnPottsDesigner — collected-only gate."""
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from jaxtyping import PRNGKeyArray


@pytest.mark.slow
@pytest.mark.potts
class TestMpnnPottsDesignerSmoke:
    """Collected under 'potts and slow'; full run requires real Aminx + PottsModel weights."""

    def test_designer_attributes(self):
        """Verify MpnnPottsDesigner and DesignResult are importable with expected attrs."""
        from aminx.potts.designer import DesignResult, MpnnPottsDesigner

        assert hasattr(MpnnPottsDesigner, "run_design")
        assert hasattr(DesignResult, "__dataclass_fields__")

    def test_designer_run_with_real_mpnn_weights(self, rng_key: PRNGKeyArray):
        """Test MpnnPottsDesigner.run_design with real MPNN weights and synthetic Potts model.

        Loads:
        - Real Aminx MPNN weights from src/aminx/model_params/proteinmpnn_v_48_020.eqx.zst
        - Synthetic PottsModel (minimal, for testing orchestration only)

        Creates minimal test inputs (5-residue chain) and verifies:
        - MPNN model loads without error
        - MpnnPottsDesigner constructs successfully
        - run_design returns expected output shapes and types
        - DesignResult fields are populated correctly

        NOTE: Actual Potts model weights are not loaded (incompatible with current
        skeleton); instead we use a synthetic PottsModel to verify orchestration.
        """
        from aminx.io.weights import load_weights
        from aminx.model.mpnn import Aminx
        from aminx.potts.designer import DesignResult, MpnnPottsDesigner
        from aminx.potts.model import PottsModel, PottsParams

        # Resolve MPNN weights path
        repo_root = Path(__file__).parent.parent.parent
        mpnn_weights_path = repo_root / "src" / "aminx" / "model_params" / "proteinmpnn_v_48_020.eqx.zst"

        # Skip if MPNN weights don't exist (acceptable for CI without weights)
        if not mpnn_weights_path.exists():
            pytest.skip(f"MPNN model weights not found at {mpnn_weights_path}")

        # Split keys
        k_mpnn, k_potts, k_run = jax.random.split(rng_key, 3)

        # Create Aminx skeleton with standard ProteinMPNN architecture
        # num_positional_embeddings: the proteinmpnn_v_48_020 model was trained with
        # 32 positional embeddings (encoded as (16, 66) in weight matrix).
        aminx_skeleton = Aminx(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            num_positional_embeddings=32,
            key=k_mpnn,
        )

        # Load real MPNN weights
        aminx_model = load_weights(local_path=str(mpnn_weights_path), skeleton=aminx_skeleton)
        assert isinstance(aminx_model, Aminx), f"Expected Aminx, got {type(aminx_model)}"

        # Create synthetic PottsModel (for testing, not loading real weights)
        potts_model = PottsModel(
            hidden_dim=128,
            num_aa=21,
            k_neighbors=16,
            edge_features_dim=128,
            trw_iters=15,
            key=k_potts,
            training=False,
        )

        # Construct MpnnPottsDesigner with real MPNN + synthetic Potts
        designer = MpnnPottsDesigner(mpnn=aminx_model, potts=potts_model)
        assert designer.mpnn is aminx_model
        assert designer.potts is potts_model

        # Create minimal test inputs: 5-residue chain
        n_residues = 5
        coords = jnp.ones((n_residues, 37, 3), dtype=jnp.float32)
        mask = jnp.ones((n_residues,), dtype=jnp.float32)
        residue_index = jnp.arange(n_residues, dtype=jnp.int32)
        chain_index = jnp.zeros((n_residues,), dtype=jnp.int32)

        # Mock the expensive potts.infer_params to avoid JAX computation
        # This allows us to test the orchestration without full inference cost
        def mock_infer_params(*args, **kwargs) -> PottsParams:
            return PottsParams(
                marginals=jnp.ones((n_residues, 21)) / 21,
                h=jnp.zeros((n_residues, 21)),
                J=jnp.zeros((n_residues, n_residues, 21, 21)),
                rho=jnp.zeros((n_residues, n_residues)),
                W=jnp.ones((n_residues, n_residues)),
            )

        # Run smoke design with mocked Potts inference
        with patch.object(
            PottsModel,
            "infer_params",
            side_effect=mock_infer_params,
        ):
            result = designer.run_design(
                key=k_run,
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                n_gibbs_sweeps=1,
            )

        # Verify output shape and type
        assert isinstance(result, DesignResult)
        assert result.mpnn_logits.shape == (n_residues, 21)
        assert result.potts_marginals.shape == (n_residues, 21)
        assert result.gibbs_seq.shape == (n_residues,)
        assert isinstance(result.sequence, str)
        assert len(result.sequence) == n_residues

        # Verify values are finite and in expected ranges
        assert jnp.all(jnp.isfinite(result.mpnn_logits))
        assert jnp.all(jnp.isfinite(result.potts_marginals))
        assert jnp.all((result.gibbs_seq >= 0) & (result.gibbs_seq < 21))

        # Verify sequence is valid amino acids
        from aminx.utils.aa_convert import MPNN_ALPHABET
        for aa_char in result.sequence:
            assert aa_char in MPNN_ALPHABET
