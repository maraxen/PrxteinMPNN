// Drive the SHIPPING ph_design.mjs host-loop methods under Node with onnxruntime-web (wasm, 1 thread), ONE (cell, case) per process.
//
// No grading here: read one job, run PhDesigner.design from PDB text with the recorded draws as the replay, write the designs (with
// their recorded pick events). The tracked Python gate (scripts/browser_validation/protonpotts_ph_methods_node_gate.py) does every comparison.
//
// Usage: node run_ph_methods_node.mjs --ort-dir <dir with node_modules/onnxruntime-web> --scoring-models <dir>
//          --ph-models <dir> --job <job.json> --out <result.json>
// job.json: {"pdb": "<PDB text>", "labels": null, "bucket": <int>, "config": {...camelCase PhDesigner config...},
//            "replay": [{"uniforms": [...], "randints": [...], "perms": [[...], ...]}, ...]  (one entry per sample)}
// result.json: {"ok": bool, "error": str|null, "num_threads": int, "designs": [{sequence, sample, method, label, designable, pins,
//               final_potts_energy, selective_energy, selective_energies, n_draws, events: [{uniform, choice}]}] | null}

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { PhDesigner } from "./ph_design.mjs";

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) throw new Error(`missing --${name}`);
  return process.argv[i + 1];
}

const ortPkgDir = join(arg("ort-dir"), "node_modules", "onnxruntime-web");
const ort = await import(pathToFileURL(join(ortPkgDir, "dist", "ort.wasm.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(join(ortPkgDir, "dist") + "/").href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const scoringDir = arg("scoring-models");
const phDir = arg("ph-models");
const scoring = JSON.parse(readFileSync(join(scoringDir, "MANIFEST.json"), "utf8"));
const ph = JSON.parse(readFileSync(join(phDir, "MANIFEST.json"), "utf8"));
const job = JSON.parse(readFileSync(arg("job"), "utf8"));

const bytes = (dir, file) => new Uint8Array(readFileSync(join(dir, file)));
const record = { ok: false, error: null, num_threads: ort.env.wasm.numThreads, designs: null };
try {
  const scoringEntry = scoring.buckets.find((b) => b.bucket === job.bucket);
  const phEntry = ph.buckets.find((b) => b.bucket === job.bucket);
  if (!scoringEntry || !phEntry) throw new Error(`no manifest entry for bucket ${job.bucket}`);
  const graphs = {};
  for (const [name, meta] of Object.entries(phEntry.graphs)) graphs[name] = bytes(phDir, meta.file);
  const designer = await PhDesigner.create(ort, ph, scoringEntry, job.bucket, {
    table: bytes(scoringDir, scoringEntry.graphs.table.file),
    energy: bytes(scoringDir, scoringEntry.graphs.energy.file),
    graphs,
  });
  // The replay: one entry per sample, each {uniforms, randints, perms} exactly as the Python record wrote it (float64 values kept).
  const designs = await designer.design(job.pdb, job.labels, job.config, job.replay);
  record.designs = designs.map((d) => ({
    sequence: Array.from(d.sequence),
    sample: d.sample,
    method: d.method,
    label: d.label,
    designable: d.designable,
    pins: d.pins.map((p) => ({ position: p.position, protIdx: p.protIdx, depIdxs: p.depIdxs, resId: p.resId, type: p.protonationType })),
    final_potts_energy: d.finalPottsEnergy,
    selective_energy: d.selectiveEnergy,
    selective_energies: d.selectiveEnergies,
    n_draws: d.nDraws,
    events: (d.events || []).map((e) => ({ uniform: e.uniform, choice: e.choice })),
  }));
  record.ok = true;
} catch (err) {
  record.error = String(err && err.stack ? err.stack : err);
}
writeFileSync(arg("out"), JSON.stringify(record, null, 2));
