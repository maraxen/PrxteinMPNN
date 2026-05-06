"""Registry wiring for :data:`~prxteinmpnn.registry.SAMPLERS` (Phase 5 prep)."""

from __future__ import annotations

import pytest

from prxteinmpnn.registry import SAMPLERS
from prxteinmpnn.run.sampling_driver import SamplingDriver
from prxteinmpnn.run.specs import SamplingSpecification


def test_make_sample_sequences_registered() -> None:
  import importlib

  importlib.import_module("prxteinmpnn.sampling.sample")
  from prxteinmpnn.sampling.sample import make_sample_sequences

  assert SAMPLERS.get("make_sample_sequences") is make_sample_sequences


def test_sampling_driver_unknown_key_raises() -> None:
  spec = SamplingSpecification(inputs=["/tmp/x.pdb"], num_samples=1)
  driver = SamplingDriver(spec, sampler_factory_key="does_not_exist")
  with pytest.raises(KeyError, match="Unknown sampler factory"):
    driver.build_sampler_fn(object())


def test_sampling_driver_sampler_factory_keys_includes_default() -> None:
  spec = SamplingSpecification(inputs=["/tmp/x.pdb"], num_samples=1)
  driver = SamplingDriver(spec)
  keys = driver.sampler_factory_keys()
  assert "make_sample_sequences" in keys
