"""Pipeline Protocol implementations for prxteinmpnn.

Each Pipeline is a host-only frozen dataclass implementing the Pipeline protocol:
    pipeline(module, key, inputs, *, fns: PipelineFns) -> OutputT

Available pipelines:
    UnconditionalPipeline  — unconditional sequence scoring
    ConditionalPipeline    — conditional (teacher-forced) sequence scoring
    AutoregressivePipeline — temperature-sampled autoregressive sequence design
    STEPipeline            — straight-through estimator for differentiable design
"""
