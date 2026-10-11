// In-page driver for the X11c gate: the pH host-loop methods (PhDesigner.design with config.method in converged_mcmc,
// converged_mcmc_combined, two_phase, gibbs) with recorded draws as the replay, in a real browser.
//
// Loaded by ph_methods_index.html (copied to the site root as index.html). Same protocol as ph_page.mjs / run_p07.mjs: set
// `window.__PARITY_DONE__` when finished, the payload on `window.__PARITY_RESULT__`, any failure on `window.__PARITY_ERROR__`.
//
// `cells.json` (written by the tracked Python gate) supplies cells[{id, bucket, pdb, labels, config, replay}]: the same job shape
// run_ph_methods_node.mjs reads, so each returned design has exactly that runner's per-design shape and X11b's comparator consumes it
// unchanged. ph_design.mjs is imported as-is. This grades NOTHING; every comparison and threshold lives in the tracked Python gate.

import * as ort from "./ort/ort.wasm.min.mjs";

import { PhDesigner } from "./ph_design.mjs";

const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.arrayBuffer();
}

// The per-design record, field for field as run_ph_methods_node.mjs writes it.
function record(d) {
  return {
    sequence: d.sequence === null ? null : Array.from(d.sequence),
    sample: d.sample,
    replay_exhausted: d.replayExhausted === true,
    argmins: (d.argmins || []).slice(),
    method: d.method,
    label: d.label,
    designable: d.designable,
    pins: d.pins.map((p) => ({ position: p.position, protIdx: p.protIdx, depIdxs: p.depIdxs, resId: p.resId, type: p.protonationType })),
    final_potts_energy: d.finalPottsEnergy ?? null,
    selective_energy: d.selectiveEnergy ?? null,
    selective_energies: d.selectiveEnergies ?? null,
    n_draws: d.nDraws,
    events: (d.events || []).map((e) => ({ uniform: e.uniform, choice: e.choice, probs: Array.from(e.probs) })),
  };
}

async function main() {
  // ABSOLUTE, not "./ort/": ORT resolves wasmPaths relative to its own module URL (see split_page.mjs).
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = numThreads;
  ort.env.logLevel = "error";

  const cells = await (await fetch("./cells.json")).json();
  const scoring = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/scoring/MANIFEST.json")));
  const ph = JSON.parse(new TextDecoder().decode(await fetchBytes("./models/ph/MANIFEST.json")));
  const designers = {};
  for (const bucket of [...new Set(cells.cells.map((c) => c.bucket))]) {
    const scoringEntry = scoring.buckets.find((b) => b.bucket === bucket);
    const phEntry = ph.buckets.find((b) => b.bucket === bucket);
    if (!scoringEntry || !phEntry) throw new Error(`no manifest entry for bucket ${bucket}`);
    const graphs = {};
    for (const [name, meta] of Object.entries(phEntry.graphs)) graphs[name] = new Uint8Array(await fetchBytes(`./models/ph/${meta.file}`));
    designers[bucket] = await PhDesigner.create(ort, ph, scoringEntry, bucket, {
      table: new Uint8Array(await fetchBytes(`./models/scoring/${scoringEntry.graphs.table.file}`)),
      energy: new Uint8Array(await fetchBytes(`./models/scoring/${scoringEntry.graphs.energy.file}`)),
      graphs,
    });
  }
  const numThreadsAfterInit = ort.env.wasm.numThreads;

  const results = [];
  for (const cell of cells.cells) {
    const rec = { id: cell.id, ok: false, error: null, designs: null, wall_ms: null };
    try {
      const pdbText = new TextDecoder().decode(await fetchBytes(`./${cell.pdb}`));
      const t0 = performance.now();
      // design(pdbText, labels, config, uniforms, replay): the 4th argument is block_descent's uniforms (null); the replay is the 5th.
      const designs = await designers[cell.bucket].design(pdbText, cell.labels, cell.config, null, cell.replay);
      rec.wall_ms = performance.now() - t0;
      rec.designs = designs.map(record);
      rec.ok = true;
    } catch (err) {
      rec.error = String((err && err.stack) || err);
    }
    results.push(rec);
  }

  return {
    n_cells: results.length,
    num_threads_requested: numThreads,
    num_threads_after_init: numThreadsAfterInit,
    num_threads_effective: ort.env.wasm.numThreads,
    cross_origin_isolated: Boolean(window.crossOriginIsolated),
    ort_version: String((ort.env && ort.env.versions && ort.env.versions.common) || ""),
    user_agent: navigator.userAgent,
    cells: results,
  };
}

main()
  .then((res) => {
    window.__PARITY_RESULT__ = res;
    window.__PARITY_DONE__ = true;
  })
  .catch((err) => {
    window.__PARITY_ERROR__ = String((err && err.stack) || err);
    window.__PARITY_DONE__ = true;
  });
