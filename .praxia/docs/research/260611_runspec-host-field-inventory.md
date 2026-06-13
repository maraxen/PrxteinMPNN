# RS-1 Host Field Inventory (AC-RS-1a)

**task_id:** 260611_runspec-xtrax-unified  
**backlog:** #1620 (RS-1)  
**date:** 2026-06-11  
**scope:** Execution reads in `host/{plan,prep,runner,streaming,kernel_dispatch}.py` only (research artifact; no product code changes).

## Summary

- **Row count:** 136 execution-site reads across 5 host modules (kernel_dispatch.py=16, plan.py=14, prep.py=36, runner.py=50, streaming.py=20).
- **Unique read patterns:** 69
- **Existing `run_spec` read sites:** 2 codebase-wide — (1) `streaming.py:310` `spec.run_spec.multistate.n_states` (in-scope host hot path); (2) `training/trainer.py:53` `spec.run_spec.precision.compute` (training path, out of host scope).
- **Flag — `use_unified_driver` default mismatch (RS-5):** Historical audits recorded `getattr(spec, "use_unified_driver", False)` at `kernel_dispatch.py:171` vs `SamplingSpecification.use_unified_driver=True` (`specs.py:311`). **Current source uses default `True`** (aligned with spec); RS-5 AC is satisfied — retain row to guard regression.

## Partition key

| Target class | T4.1 fate |
|---|---|
| `run_spec.io/resource/batching/precision` | Generic → xtrax (RS-6 migration reads) |
| `run_spec.multistate/ligand/tied/grid/averaging` | Protein sub-config — **stay in aminx**, never xtrax MOVE |
| `run_spec.planner` (RS-2) | aminx PlannerTopology builder |
| `protein-only façade` | Dataclass fields with no RunSpec leaf or build-time-only projection |

## Inventory table

| file:line | read pattern | current source | target run_spec path | migration sprint | notes |
|---|---|---|---|---|---|
| kernel_dispatch.py:109 | `spec.backbone_noise` | RunSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| kernel_dispatch.py:110 | `spec.temperature` | SamplingSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| kernel_dispatch.py:125 | `spec.tie_group_map` | RunSpecification | run_spec.tied.tie_group_map | RS-6 | Protein sub-config (traced) |
| kernel_dispatch.py:127 | `spec.tie_group_map` | RunSpecification | run_spec.tied.tie_group_map | RS-6 | Protein sub-config (traced) |
| kernel_dispatch.py:128 | `spec.tie_group_map` | RunSpecification | run_spec.tied.tie_group_map | RS-6 | Protein sub-config (traced) |
| kernel_dispatch.py:132 | `spec.structure_mapping` | RunSpecification | run_spec.tied.structure_mapping | RS-6 | Protein sub-config (traced) |
| kernel_dispatch.py:133 | `spec.structure_mapping` | RunSpecification | run_spec.tied.structure_mapping | RS-6 | Protein sub-config (traced) |
| kernel_dispatch.py:158 | `spec.state_weights` | SamplingSpecification | run_spec.averaging.state_weights | RS-6 | Protein sub-config |
| kernel_dispatch.py:171 | `getattr(spec, "use_unified_driver"` | SamplingSpecification | run_spec.planner.unified_driver | RS-5 | **RS-5 flag**: AC required getattr default True (was False in audits); current `kernel_dispatch.py:171` uses True, matching `specs.py:311` |
| kernel_dispatch.py:199 | `spec.bias` | SamplingSpecification | protein-only façade | — | Never xtrax |
| kernel_dispatch.py:270 | `spec.bias` | SamplingSpecification | protein-only façade | — | Never xtrax |
| kernel_dispatch.py:291 | `spec.backbone_noise` | RunSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| kernel_dispatch.py:356 | `spec.bias` | SamplingSpecification | protein-only façade | — | Never xtrax |
| kernel_dispatch.py:423 | `spec.bias` | SamplingSpecification | protein-only façade | — | Never xtrax |
| kernel_dispatch.py:444 | `spec.backbone_noise` | RunSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| kernel_dispatch.py:516 | `spec.compute_pseudo_perplexity` | SamplingSpecification | protein-only façade | — | Never xtrax |
| plan.py:87 | `getattr(spec, "batch_size"` | RunSpecification | run_spec.batching.batch_size | RS-6 | Generic → xtrax T4.1 |
| plan.py:89 | `getattr(spec, "samples_batch_size"` | SamplingSpecification | run_spec.batching.samples_batch_size | RS-6 | Also run_spec.resource.sample_batch_size |
| plan.py:92 | `getattr(spec, "temperature"` | SamplingSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| plan.py:94 | `getattr(spec, "backbone_noise"` | RunSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| plan.py:194 | `spec.num_samples` | SamplingSpecification | protein-only façade | — | Never xtrax |
| plan.py:227 | `hasattr(spec, "samples_chunk_size"` | SamplingSpecification | run_spec.batching.samples_chunk_size | RS-6 |  |
| plan.py:227 | `spec.samples_chunk_size` | SamplingSpecification | run_spec.batching.samples_chunk_size | RS-6 |  |
| plan.py:228 | `spec.samples_chunk_size` | SamplingSpecification | run_spec.batching.samples_chunk_size | RS-6 |  |
| plan.py:640 | `getattr(spec, "use_rolling_state"` | SamplingSpecification | run_spec.planner.use_rolling_state | RS-2 | PlannerTopology (new) |
| plan.py:643 | `getattr(spec, "multi_state_strategy"` | Sampling/Scoring | run_spec.multistate.combine_strategy | RS-6 | Protein sub-config (aminx-only) |
| plan.py:644 | `getattr(spec, "multi_state_temperature"` | RunSpecification | run_spec.tied.multi_state_temperature | RS-6 | Protein sub-config |
| plan.py:645 | `getattr(spec, "state_weights"` | SamplingSpecification | run_spec.averaging.state_weights | RS-6 | Protein sub-config |
| plan.py:650 | `getattr(spec, "average_node_features"` | Sampling/Scoring | run_spec.averaging.average_node_features | RS-6 | Protein sub-config |
| plan.py:663 | `getattr(spec, "sampling_strategy"` | SamplingSpecification | protein-only façade | — | Never xtrax |
| prep.py:37 | `spec.checkpoint_registry_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:39 | `spec.checkpoint_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:42 | `spec.checkpoint_registry_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:48 | `spec.model_family` | RunSpecification | run_spec.ligand.model_family | RS-6 | Protein sub-config |
| prep.py:49 | `spec.checkpoint_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:55 | `spec.model_family` | RunSpecification | run_spec.ligand.model_family | RS-6 | Protein sub-config |
| prep.py:56 | `spec.checkpoint_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:96 | `spec.chain_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:97 | `spec.model` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:98 | `spec.altloc` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:99 | `spec.topology` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:103 | `spec.inputs` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:104 | `spec.batch_size` | RunSpecification | run_spec.batching.batch_size | RS-6 | Generic → xtrax T4.1 |
| prep.py:105 | `spec.foldcomp_database` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:107 | `spec.pass_mode` | RunSpecification | run_spec.tied.pass_mode | RS-6 | Protein sub-config |
| prep.py:108 | `spec.use_preprocessed` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:109 | `spec.preprocessed_index_path` | RunSpecification | protein-only façade | — | Build-time → run_spec.io.manifest_path |
| prep.py:110 | `spec.split` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:111 | `spec.use_electrostatics` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:112 | `spec.estat_noise` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:113 | `spec.estat_noise_mode` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:114 | `spec.use_vdw` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:115 | `spec.vdw_noise` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:116 | `spec.vdw_noise_mode` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:117 | `spec.max_length` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:118 | `spec.truncation_strategy` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:124 | `spec.model_local_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:125 | `spec.model_local_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:129 | `spec.cache_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:130 | `spec.cache_path` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:132 | `spec.checkpoint_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:134 | `spec.checkpoint_id` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:135 | `spec.model_weights` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:136 | `spec.model_version` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:141 | `spec.model_version` | RunSpecification | protein-only façade | — | Never xtrax |
| prep.py:142 | `spec.model_weights` | RunSpecification | protein-only façade | — | Never xtrax |
| runner.py:164 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| runner.py:170 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:202 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:225 | `spec.grid_mode` | SamplingSpecification | run_spec.grid.grid_mode | RS-6 | Protein sub-config |
| runner.py:232 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:312 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| runner.py:326 | `spec.sequences_to_score` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:331 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:333 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:341 | `spec.random_seed` | RunSpecification | protein-only façade | — | Not on RunSpec today |
| runner.py:352 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:353 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:363 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:364 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:400 | `spec.multi_state_strategy` | Sampling/Scoring | run_spec.multistate.combine_strategy | RS-6 | Protein sub-config (aminx-only) |
| runner.py:404 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:406 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:410 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:412 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:416 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:418 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:438 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| runner.py:442 | `spec.return_decoding_orders` | ScoringSpecification | protein-only façade | — | Never xtrax |
| runner.py:492 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| runner.py:518 | `spec.inspection_features` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:527 | `spec.random_seed` | RunSpecification | protein-only façade | — | Not on RunSpec today |
| runner.py:545 | `spec.inspection_features` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:631 | `spec.distance_matrix` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:632 | `spec.distance_matrix_method` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:634 | `spec.distance_matrix_method` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:636 | `spec.distance_matrix_method` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:646 | `spec.distance_matrix_method` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:649 | `spec.cross_input_similarity` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:668 | `spec.distance_matrix` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:671 | `spec.cross_input_similarity` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:681 | `spec.similarity_metric` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:683 | `spec.similarity_metric` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:685 | `spec.similarity_metric` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:688 | `spec.similarity_metric` | InspectionSpecification | protein-only façade | RS-8 | Inspection export gap |
| runner.py:714 | `spec.combine` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:718 | `spec.jacobian_mode` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:718 | `spec.compute_apc` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:736 | `spec.compute_apc` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:741 | `spec.random_seed` | RunSpecification | protein-only façade | — | Not on RunSpec today |
| runner.py:742 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| runner.py:770 | `spec.jacobian_mode` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:796 | `spec.apc_residue_batch_size` | JacobianSpecification | run_spec.batching.apc_residue_batch_size | RS-6 | Generic → xtrax T4.1 |
| runner.py:817 | `spec.jacobian_mode` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:824 | `spec.jacobian_mode` | JacobianSpecification | protein-only façade | — | Never xtrax |
| runner.py:840 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:60 | `spec.use_arrayrecord` | SamplingSpecification | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:60 | `spec.campaign_mode` | SamplingSpecification | run_spec.grid.campaign_mode | RS-6 | Protein sub-config |
| streaming.py:65 | `spec.use_arrayrecord` | SamplingSpecification | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:65 | `spec.campaign_mode` | SamplingSpecification | run_spec.grid.campaign_mode | RS-6 | Protein sub-config |
| streaming.py:86 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:87 | `spec.grid_mode` | SamplingSpecification | run_spec.grid.grid_mode | RS-6 | Protein sub-config |
| streaming.py:88 | `spec.model_family` | RunSpecification | run_spec.ligand.model_family | RS-6 | Protein sub-config |
| streaming.py:89 | `spec.ligand_conditioning` | SamplingSpecification | run_spec.ligand.ligand_conditioning | RS-6 | Protein sub-config |
| streaming.py:90 | `spec.sidechain_conditioning` | SamplingSpecification | run_spec.ligand.sidechain_conditioning | RS-6 | Protein sub-config |
| streaming.py:117 | `spec.campaign_mode` | SamplingSpecification | run_spec.grid.campaign_mode | RS-6 | Protein sub-config |
| streaming.py:141 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| streaming.py:166 | `spec.backbone_noise` | RunSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| streaming.py:167 | `spec.temperature` | SamplingSpecification | protein-only façade | RS-6 | Axis values; cardinality → run_spec.planner (RS-2) |
| streaming.py:215 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| streaming.py:247 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:248 | `spec.grid_mode` | SamplingSpecification | run_spec.grid.grid_mode | RS-6 | Protein sub-config |
| streaming.py:293 | `spec.output_h5_path` | task subclasses | protein-only façade | RS-6 | Build-time → run_spec.io.sink_kind |
| streaming.py:310 | `spec.run_spec.multistate.n_states` | RunSpecification | run_spec.multistate.n_states | RS-6 | **EXISTING** run_spec read (1 of 2 codebase sites; 2nd: `training/trainer.py:53` `run_spec.precision.compute`) |
| streaming.py:352 | `spec.return_logits` | Sampling/Scoring | protein-only façade | RS-6 | Never xtrax |
| streaming.py:398 | `spec.grid_mode` | SamplingSpecification | run_spec.grid.grid_mode | RS-6 | Protein sub-config |

## Protein-only fields — must never move to xtrax

These fields have **no generic xtrax destination** (T4.1 MOVE boundary). They remain on aminx dataclass façades or aminx-only RunSpec sub-configs:

### Permanent dataclass façade (no xtrax, many never on RunSpec)

- `altloc`
- `backbone_noise`
- `bias`
- `cache_path`
- `chain_id`
- `checkpoint_id`
- `checkpoint_registry_path`
- `combine`
- `compute_apc`
- `compute_pseudo_perplexity`
- `cross_input_similarity`
- `distance_matrix`
- `distance_matrix_method`
- `estat_noise`
- `estat_noise_mode`
- `foldcomp_database`
- `inputs`
- `inspection_features`
- `jacobian_mode`
- `max_length`
- `model`
- `model_local_path`
- `model_version`
- `model_weights`
- `num_samples`
- `output_h5_path`
- `preprocessed_index_path`
- `random_seed`
- `return_decoding_orders`
- `return_logits`
- `sampling_strategy`
- `sequences_to_score`
- `similarity_metric`
- `split`
- `temperature`
- `topology`
- `truncation_strategy`
- `use_arrayrecord`
- `use_electrostatics`
- `use_preprocessed`
- `use_vdw`
- `vdw_noise`
- `vdw_noise_mode`

### aminx RunSpec sub-configs (stay in aminx after T4.1; RS-6 retargets reads, not MOVE)

- `run_spec.averaging.average_node_features`
- `run_spec.averaging.state_weights`
- `run_spec.batching.apc_residue_batch_size`
- `run_spec.batching.batch_size`
- `run_spec.batching.samples_batch_size`
- `run_spec.batching.samples_chunk_size`
- `run_spec.grid.campaign_mode`
- `run_spec.grid.grid_mode`
- `run_spec.ligand.ligand_conditioning`
- `run_spec.ligand.model_family`
- `run_spec.ligand.sidechain_conditioning`
- `run_spec.multistate.combine_strategy`
- `run_spec.multistate.n_states`
- `run_spec.planner.unified_driver`
- `run_spec.planner.use_rolling_state`
- `run_spec.tied.multi_state_temperature`
- `run_spec.tied.pass_mode`
- `run_spec.tied.structure_mapping`
- `run_spec.tied.tie_group_map`
