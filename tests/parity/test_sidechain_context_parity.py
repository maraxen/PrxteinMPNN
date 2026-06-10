"""Side-chain-context logits parity: aminx score path vs LigandMPNN reference.

Comparand (matches the proven ``test_decoder_conditional_scoring_parity`` pattern):
the reference ``ProteinMPNN.score(feature_dict, use_sequence=True)["log_probs"]``
derives its decoding order internally from ``randn``; the aminx score path consumes
an ``ar_mask`` encoding the SAME order. Inputs come from a real structure
(``tests/data/1ubq.pdb``) parsed to atom37 so the side-chain slots (5:) carry real
atoms. The aminx ``fixed_mask`` (1=fixed) maps to the reference ``chain_mask``
(1=designable) via ``fixed_mask = 1 - chain_mask``; side-chain context is gated to
fixed residues in both frameworks.

Results (ligandmpnn_v_32_010_25, 1ubq):
- side chains OFF (all-designable):  Pearson ~0.9998   -> conventions/harness correct
- side chains OFF (partial-fixed):   Pearson ~0.9998   -> fixed_mask mapping correct
- side chains ON  (partial-fixed):   Pearson >=0.95    -> parity (was ~0.85; fixed)

The side-chain-ON divergence (previously ~0.85) was a bug in
``ProteinFeaturesLigand._make_angle_features``: the residue-frame projection used the
einsum subscript ``"lqp, lym -> lyp"``, which (because ``q`` and ``m`` each appear in
only one operand) independently summed both axes -- an outer product of column-sums
rather than the frame projection ``e_p . diff``. Corrected to ``"lqp, lyq -> lyp"``.
The corruption only surfaced with side chains ON because the dummy ligand is fully
masked in the OFF baseline, so the corrupted node features never contributed there.
"""

from __future__ import annotations

import io
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zstandard as zstd
from scipy.stats import pearsonr

from aminx import parse_structure
from aminx.inference import score_conditional
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.model.ligand_mpnn import PrxteinLigandMPNN
from tests.parity.reference_utils import require_heavy_parity_prereqs

torch = pytest.importorskip("torch")

# --- topology of ligandmpnn_v_32_010_25 ---
_K_NEIGHBORS = 32
_NUM_POSITIONAL = 32
_NODE_FEATURES = 128
_EDGE_FEATURES = 128
_HIDDEN_FEATURES = 128
_NUM_ENCODER_LAYERS = 3
_NUM_DECODER_LAYERS = 3
_ATOM_CONTEXT_NUM = 16

_CHECKPOINT_ID = "ligandmpnn_v_32_010_25"
_REFERENCE_PT = f"model_params/{_CHECKPOINT_ID}.pt"
_CONVERTED_EQX = f"src/aminx/model_params/{_CHECKPOINT_ID}.eqx.zst"
_PDB = "tests/data/1ubq.pdb"

_PARITY_THRESHOLD = 0.95  # the no-side-chain path converts cleanly (~0.9998)

reference_root, repo_root = require_heavy_parity_prereqs(
    python_modules=["torch"],
    reference_rel_paths=[_REFERENCE_PT],
    converted_rel_paths=[_CONVERTED_EQX],
)

import model_utils  # noqa: E402  (sys.path prepared by require_heavy_parity_prereqs)


def _load_reference_model(*, use_side_chain_context: bool) -> Any:
    model = model_utils.ProteinMPNN(
        num_letters=21,
        node_features=_NODE_FEATURES,
        edge_features=_EDGE_FEATURES,
        hidden_dim=_HIDDEN_FEATURES,
        num_encoder_layers=_NUM_ENCODER_LAYERS,
        num_decoder_layers=_NUM_DECODER_LAYERS,
        k_neighbors=_K_NEIGHBORS,
        atom_context_num=_ATOM_CONTEXT_NUM,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=use_side_chain_context,
    )
    ckpt = torch.load(str(reference_root / _REFERENCE_PT), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_aminx_model(*, use_side_chain_context: bool) -> PrxteinLigandMPNN:
    skeleton = PrxteinLigandMPNN(
        node_features=_NODE_FEATURES,
        edge_features=_EDGE_FEATURES,
        hidden_features=_HIDDEN_FEATURES,
        num_encoder_layers=_NUM_ENCODER_LAYERS,
        num_decoder_layers=_NUM_DECODER_LAYERS,
        k_neighbors=_K_NEIGHBORS,
        num_positional_embeddings=_NUM_POSITIONAL,
        ligand_mpnn_use_side_chain_context=use_side_chain_context,
        key=jax.random.PRNGKey(0),
    )
    with open(repo_root / _CONVERTED_EQX, "rb") as f:
        stream = io.BytesIO(zstd.ZstdDecompressor().decompress(f.read()))
    return eqx.tree_deserialise_leaves(stream, skeleton)


def _structure() -> dict[str, np.ndarray]:
    p = parse_structure(_PDB)
    xyz37 = np.asarray(p.coordinates, np.float32)  # (L, 37, 3)
    return {
        "xyz37": xyz37,
        "xyz37_m": np.asarray(p.atom_mask, np.float32),  # (L, 37)
        "aa": np.asarray(p.aatype, np.int64),  # (L,)
        "Xbb": xyz37[:, [0, 1, 2, 4], :],  # N,CA,C,O (atom37: O at slot 4)
        "ridx": np.asarray(p.residue_index, np.int64),
        "cidx": np.asarray(p.chain_index, np.int64),
    }


def _ar_mask(chain_mask: np.ndarray, randn: np.ndarray) -> np.ndarray:
    """Backward AR mask for the decoding order the reference derives from randn."""
    order = np.argsort((chain_mask + 1e-4) * np.abs(randn))
    pos = {tok: i for i, tok in enumerate(order)}
    L = chain_mask.shape[0]
    ar = np.zeros((L, L), np.int32)
    for i in range(L):
        for j in range(L):
            if pos[j] < pos[i]:
                ar[i, j] = 1
    return ar


def _logits(*, use_sc: bool, chain_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference_log_probs, aminx_log_probs), both (L, 21)."""
    s = _structure()
    L = s["aa"].shape[0]
    randn = np.random.default_rng(0).standard_normal((L,)).astype(np.float32)
    actx = _ATOM_CONTEXT_NUM

    fd = {
        "X": torch.from_numpy(s["Xbb"][None].copy()),
        "S": torch.from_numpy(s["aa"][None].copy()),
        "mask": torch.ones((1, L), dtype=torch.float32),
        "chain_mask": torch.from_numpy(chain_mask[None].astype(np.float32)),
        "R_idx": torch.from_numpy(s["ridx"][None].copy()),
        "chain_labels": torch.from_numpy(s["cidx"][None].copy()),
        "randn": torch.from_numpy(randn[None].copy()),
        "Y": torch.zeros((1, L, actx, 3), dtype=torch.float32),
        "Y_t": torch.zeros((1, L, actx), dtype=torch.int32),
        "Y_m": torch.zeros((1, L, actx), dtype=torch.int32),
        "xyz_37": torch.from_numpy(s["xyz37"][None].copy()),
        "xyz_37_m": torch.from_numpy(s["xyz37_m"][None].copy()),
        "batch_size": 1,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
    }
    with torch.no_grad():
        ref_lp = _load_reference_model(use_side_chain_context=use_sc).score(
            fd, use_sequence=True
        )["log_probs"].numpy()[0]

    kw = dict(
        coords=jnp.asarray(s["Xbb"]),
        mask=jnp.ones((L,), jnp.float32),
        residue_index=jnp.asarray(s["ridx"], jnp.int32),
        chain_index=jnp.asarray(s["cidx"], jnp.int32),
        sequence=jax.nn.one_hot(jnp.asarray(s["aa"]), 21),
        ar_mask=jnp.asarray(_ar_mask(chain_mask, randn)),
        ligand_coords=jnp.zeros((L, actx, 3)),
        ligand_atom_types=jnp.zeros((L, actx), jnp.int32),
        ligand_mask=jnp.zeros((L, actx)),
        fixed_mask=jnp.asarray(1.0 - chain_mask),  # aminx 1=fixed == ref (1-chain_mask)
        mode="score_conditional",
    )
    if use_sc:
        kw["xyz_37"] = jnp.asarray(s["xyz37"])
        kw["xyz_37_m"] = jnp.asarray(s["xyz37_m"])
    bundle, config = build_inference_bundle(**kw)
    jax_logits = score_conditional.kernel(
        _load_aminx_model(use_side_chain_context=use_sc),
        jax.random.PRNGKey(0),
        bundle,
        config,
        make_stage_set(),
    )
    jax_lp = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))
    return ref_lp, jax_lp


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pearsonr(a.ravel(), b.ravel())[0])


def _all_designable(L: int) -> np.ndarray:
    return np.ones((L,), np.float32)


def _partial_fixed(L: int) -> np.ndarray:
    cm = np.ones((L,), np.float32)
    cm[::2] = 0.0  # every other residue fixed
    return cm


@pytest.mark.requires_weights
@pytest.mark.parity_heavy
def test_baseline_no_sidechain_parity() -> None:
    """No side chains, all-designable: validates comparand + conventions (~0.9998)."""
    L = _structure()["aa"].shape[0]
    ref, jax_ = _logits(use_sc=False, chain_mask=_all_designable(L))
    assert _corr(ref, jax_) >= _PARITY_THRESHOLD


@pytest.mark.requires_weights
@pytest.mark.parity_heavy
def test_control_partial_fixed_no_sidechain_parity() -> None:
    """No side chains, partial-fixed: fixed_mask mapping does not perturb parity."""
    L = _structure()["aa"].shape[0]
    ref, jax_ = _logits(use_sc=False, chain_mask=_partial_fixed(L))
    assert _corr(ref, jax_) >= _PARITY_THRESHOLD


@pytest.mark.requires_weights
@pytest.mark.parity_heavy
def test_sidechain_context_logits_parity() -> None:
    """Side chains ON, partial-fixed: aminx score-path logits vs reference.

    Regression guard for the ``_make_angle_features`` einsum fix (see module
    docstring): the side-chain-context logits must now match the reference to the
    same ~0.95+ tolerance as the no-side-chain baseline.
    """
    L = _structure()["aa"].shape[0]
    ref, jax_ = _logits(use_sc=True, chain_mask=_partial_fixed(L))
    assert _corr(ref, jax_) >= _PARITY_THRESHOLD


@pytest.mark.requires_weights
@pytest.mark.parity_heavy
def test_sidechain_on_differs_from_off_when_residues_fixed() -> None:
    """Wiring: side-chain context must change aminx logits when residues are fixed."""
    L = _structure()["aa"].shape[0]
    _, jax_on = _logits(use_sc=True, chain_mask=_partial_fixed(L))
    _, jax_off = _logits(use_sc=False, chain_mask=_partial_fixed(L))
    assert float(np.abs(jax_on - jax_off).max()) > 0.01


@pytest.mark.requires_weights
@pytest.mark.parity_heavy
def test_sidechain_noop_when_all_designable() -> None:
    """Wiring: context is gated to fixed residues, so all-designable is a no-op."""
    L = _structure()["aa"].shape[0]
    _, jax_on = _logits(use_sc=True, chain_mask=_all_designable(L))
    _, jax_off = _logits(use_sc=False, chain_mask=_all_designable(L))
    assert float(np.abs(jax_on - jax_off).max()) < 1e-5
