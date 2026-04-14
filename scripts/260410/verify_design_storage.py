"""Tests for ArrayRecord design storage."""

import os
import jax.numpy as jnp
from prxteinmpnn.io.designs import DesignArrayRecordWriter
from array_record.python import array_record_module
import msgpack
import msgpack_numpy as m

# Ensure msgpack-numpy is patched
m.patch()

def test_design_writer_reader(tmp_path):
  output_file = str(tmp_path / "test_designs.arrayrecord")
  writer = DesignArrayRecordWriter(output_file)
  
  payload = {
    "sequence": jnp.array([1, 2, 3], dtype=jnp.int8),
    "logits": jnp.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=jnp.float16),
    "scores": jnp.array([0.5], dtype=jnp.float32),
    "state_weights": jnp.array([1.0], dtype=jnp.float32),
    "metadata": {
      "pool_type": "BackboneOnly",
      "state_mapping": [0, 0, 0],
      "weight_strategy": "Uniform",
      "combination_algorithm": "arithmetic_mean",
      "structure_ids": ["struct1"],
      "parent_structure_idx": 0,
    }
  }
  
  writer.write(payload)
  writer.close()
  
  # Read back using raw ArrayRecordReader
  reader = array_record_module.ArrayRecordReader(output_file)
  records = list(reader.read_all())
  assert len(records) == 1
  
  # Deserialize
  read_payload = msgpack.unpackb(records[0], object_hook=m.decode)
  
  # Check fields
  assert jnp.array_equal(read_payload["sequence"], payload["sequence"])
  assert jnp.allclose(read_payload["logits"], payload["logits"])
  assert read_payload["metadata"]["pool_type"] == "BackboneOnly"
  assert read_payload["metadata"]["combination_algorithm"] == "arithmetic_mean"
