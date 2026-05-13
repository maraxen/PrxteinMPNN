import inspect
import jax
from prxteinmpnn.model.capabilities import ModelCapabilities
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.protocols import ModelProtocol

KEY = jax.random.PRNGKey(42)

def test_protein_mpnn_satisfies_protocol():
    m = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=KEY)
    assert isinstance(m, ModelProtocol)
    assert isinstance(m.capabilities, ModelCapabilities)
    for method in ["score_unconditional_from_payload",
                   "score_conditional_from_payload",
                   "sample_autoregressive_state_vmap_exact_from_payload"]:
        assert hasattr(m, method)

def test_ligand_mpnn_keyword_only_ligand():
    for method in ["score_unconditional_from_payload",
                   "score_conditional_from_payload",
                   "sample_autoregressive_state_vmap_exact_from_payload"]:
        sig = inspect.signature(getattr(PrxteinLigandMPNN, method))
        assert "ligand" in sig.parameters
        assert sig.parameters["ligand"].kind == inspect.Parameter.KEYWORD_ONLY
