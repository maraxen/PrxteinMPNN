
import pytest
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
from prxteinmpnn.model.mpnn import PrxteinMPNN
import jax

def test_training_spec_defaults():
    spec = TrainingSpecification(inputs="dummy")
    assert spec.training_mode == "autoregressive"
    assert spec.diffusion_schedule_type == "cosine"

def test_training_spec_diffusion():
    spec = TrainingSpecification(
        inputs="dummy",
        training_mode="diffusion",
        diffusion_num_steps=500
    )
    assert spec.training_mode == "diffusion"
    assert spec.diffusion_num_steps == 500

def test_load_model_autoregressive():
    model = load_model(model_version="v_48_020", training_mode="autoregressive")
    assert isinstance(model, PrxteinMPNN)
    assert not isinstance(model, DiffusionPrxteinMPNN)

def test_load_model_diffusion():
    model = load_model(model_version="v_48_020", training_mode="diffusion")
    assert isinstance(model, DiffusionPrxteinMPNN)
