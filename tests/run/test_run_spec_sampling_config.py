"""Tests for SamplingConfig and build_run_spec sampling wiring (RS-6a)."""

import pytest
from aminx.run.specs import SamplingSpecification
from aminx.run.spec import build_run_spec


class TestSamplingConfigBasics:
  """Verify SamplingConfig fields round-trip from SamplingSpecification."""

  def test_sampling_config_num_samples(self):
    """num_samples field wires correctly."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=4)
    rs = build_run_spec(spec)
    assert rs.sampling.num_samples == 4

  def test_sampling_config_temperature(self):
    """temperature field converts to tuple."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, temperature=[0.1, 0.5]
    )
    rs = build_run_spec(spec)
    assert tuple(rs.sampling.temperature) == (0.1, 0.5)

  def test_sampling_config_backbone_noise(self):
    """backbone_noise field converts to tuple."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, backbone_noise=[0.0, 0.1]
    )
    rs = build_run_spec(spec)
    assert tuple(rs.sampling.backbone_noise) == (0.0, 0.1)

  def test_sampling_config_return_logits(self):
    """return_logits boolean field wires correctly."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1, return_logits=True)
    rs = build_run_spec(spec)
    assert rs.sampling.return_logits is True

  def test_sampling_config_random_seed_default(self):
    """random_seed defaults to 42 (RunSpecification base default)."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1)
    rs = build_run_spec(spec)
    assert rs.sampling.random_seed == 42

  def test_sampling_config_random_seed_set(self):
    """random_seed can be set."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1, random_seed=99)
    rs = build_run_spec(spec)
    assert rs.sampling.random_seed == 99

  def test_sampling_config_compute_pseudo_perplexity_default(self):
    """compute_pseudo_perplexity defaults to False."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1)
    rs = build_run_spec(spec)
    assert rs.sampling.compute_pseudo_perplexity is False

  def test_sampling_config_compute_pseudo_perplexity_set(self):
    """compute_pseudo_perplexity can be set to True."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, compute_pseudo_perplexity=True
    )
    rs = build_run_spec(spec)
    assert rs.sampling.compute_pseudo_perplexity is True


class TestIOConfigPaths:
  """Verify IOConfig output_h5_path and cache_path wiring."""

  def test_io_output_h5_path_set(self):
    """output_h5_path wires from spec to IOConfig."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, output_h5_path='/tmp/out.h5'
    )
    rs = build_run_spec(spec)
    assert rs.io.output_h5_path is not None
    assert str(rs.io.output_h5_path) == '/tmp/out.h5'

  def test_io_output_h5_path_unset(self):
    """output_h5_path is None when unset on spec."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1)
    rs = build_run_spec(spec)
    assert rs.io.output_h5_path is None

  def test_io_cache_path_set(self):
    """cache_path wires from spec to IOConfig."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, cache_path='/tmp/cache'
    )
    rs = build_run_spec(spec)
    assert rs.io.cache_path is not None
    assert str(rs.io.cache_path) == '/tmp/cache'

  def test_io_cache_path_unset(self):
    """cache_path is None when unset on spec."""
    spec = SamplingSpecification(inputs=['f.pdb'], num_samples=1)
    rs = build_run_spec(spec)
    assert rs.io.cache_path is None

  def test_io_cache_path_round_trip(self):
    """cache_path round-trips correctly."""
    spec = SamplingSpecification(
      inputs=['f.pdb'], num_samples=1, cache_path='/some/path'
    )
    rs = build_run_spec(spec)
    rs2 = build_run_spec(spec)
    assert rs.io.cache_path == rs2.io.cache_path
