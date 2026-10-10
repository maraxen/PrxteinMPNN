// Module Web Worker: ProtonPottsMPNN scoring, pH design and sequence sampling, off the main thread.
//
// Protocol (app.mjs is the sender):
//   in:  {id, kind: "score" | "design" | "sample", modelBase, numThreads, pdbText, params}
//   out: {id, ok: true, result} | {id, ok: false, error}
//
// Each method calls the same browser module, with the same arguments and the same output fields, as the in-page
// validation harnesses: protonpotts_page.mjs (score), ph_page.mjs (design), sampler_page.mjs (sample). Nothing in
// this file does model arithmetic.
//
// Model layout under modelBase (graph file names are relative to the manifest that lists them, as in the harnesses):
//   scoring/MANIFEST.json   table + energy graphs; used by "score", and by "design" as its scorer
//   ph/MANIFEST.json        pH graphs; used by "design" together with the scoring manifest
//   decoder/MANIFEST.json   encode + decode graphs; used by "sample"
//
// Integrity: every .onnx is sha256-checked against the "sha256" its manifest entry carries, before it becomes a
// session. A missing digest or a mismatch is an error naming the file. The manifests are not hashed here.
//
// Threads: onnxruntime-web fixes its wasm thread count at the FIRST session creation in this worker; later values are
// ignored, and without cross-origin isolation it falls back to 1. Every reply therefore carries `env` with the effective
// count ORT reports, so the caller sees what actually ran. Sessions are cached per (kind, modelBase, bucket, numThreads).
//
// Uniforms: design() takes injected uniforms (the replay contract). When params.uniforms is absent, this worker
// draws them with seededStreams (mulberry32 + Box-Muller, decoder_sampler.mjs), seeded by params.seed + sample index.
// Browser-generated streams are NOT the validated stream; the bit-exact validation passes uniforms explicitly.
// The same caveat applies to "sample" when no uniforms are given: decoder_sampler.mjs seeds its own streams.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { BUCKETS, VOCABULARY, buildProtonPottsInputs, pickBucket } from "../../browser/protonpotts-scorer/protonpotts_inputs.mjs"; // BUILD_REWRITE_IMPORT
import { createProtonPottsScorer } from "../../browser/protonpotts-scorer/protonpotts_scorer.mjs"; // BUILD_REWRITE_IMPORT
import { DEFAULT_CONFIG, PhDesigner } from "../../browser/protonpotts-scorer/ph_design.mjs"; // BUILD_REWRITE_IMPORT
import { DEFAULT_TEMPERATURE, ProtonPottsSampler, seededStreams } from "../../browser/protonpotts-scorer/decoder_sampler.mjs"; // BUILD_REWRITE_IMPORT

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = "error";

// key -> Promise<session object>. A failed build is dropped so a retry starts clean.
const cache = new Map();

function cached(key, build) {
  if (!cache.has(key)) {
    const pending = build();
    cache.set(key, pending);
    pending.catch(() => {
      if (cache.get(key) === pending) cache.delete(key);
    });
  }
  return cache.get(key);
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.json();
}

async function fetchVerifiedGraph(url, meta) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (!meta || typeof meta.sha256 !== "string") {
    throw new Error(`${url}: the manifest gives no sha256 for this file; refusing to load it`);
  }
  const got = await sha256Hex(bytes);
  if (got !== meta.sha256) {
    throw new Error(`${url}: sha256 ${got} does not match the manifest (${meta.sha256})`);
  }
  return bytes;
}

function findBucket(manifest, bucket, family) {
  const entry = (manifest.buckets || []).find((b) => b.bucket === bucket);
  if (!entry) throw new Error(`${family} manifest has no bucket ${bucket}`);
  return entry;
}

/** name -> verified graph bytes, for the graphs `names` of one bucket entry living under `${modelBase}/${dir}/`. */
async function loadGraphs(modelBase, dir, entry, names, family) {
  const metas = names.map((name) => {
    const meta = entry.graphs && entry.graphs[name];
    if (!meta || !meta.file) throw new Error(`${family} manifest bucket ${entry.bucket} lacks graph ${name}`);
    return meta;
  });
  const bytes = await Promise.all(
    metas.map((meta) => fetchVerifiedGraph(`${modelBase}/${dir}/${meta.file}`, meta)),
  );
  return Object.fromEntries(names.map((name, i) => [name, bytes[i]]));
}

function getScorer(modelBase, bucket, numThreads) {
  return cached(`score|${modelBase}|${bucket}|${numThreads}`, async () => {
    ort.env.wasm.numThreads = numThreads;
    const manifest = await fetchJson(`${modelBase}/scoring/MANIFEST.json`);
    const entry = findBucket(manifest, bucket, "scoring");
    const models = await loadGraphs(modelBase, "scoring", entry, ["table", "energy"], "scoring");
    return createProtonPottsScorer(ort, manifest, bucket, models);
  });
}

function getDesigner(modelBase, bucket, numThreads) {
  return cached(`design|${modelBase}|${bucket}|${numThreads}`, async () => {
    ort.env.wasm.numThreads = numThreads;
    // Both manifests, exactly as ph_page.mjs reads them: the scorer's entry supplies the table/energy graphs.
    const scoringManifest = await fetchJson(`${modelBase}/scoring/MANIFEST.json`);
    const phManifest = await fetchJson(`${modelBase}/ph/MANIFEST.json`);
    const scoringEntry = findBucket(scoringManifest, bucket, "scoring");
    const phEntry = findBucket(phManifest, bucket, "ph");
    const scoringGraphs = await loadGraphs(modelBase, "scoring", scoringEntry, ["table", "energy"], "scoring");
    const phGraphs = await loadGraphs(modelBase, "ph", phEntry, Object.keys(phEntry.graphs), "ph");
    return PhDesigner.create(ort, phManifest, scoringEntry, bucket, {
      table: scoringGraphs.table,
      energy: scoringGraphs.energy,
      graphs: phGraphs,
    });
  });
}

function getSampler(modelBase, bucket, numThreads) {
  return cached(`sample|${modelBase}|${bucket}|${numThreads}`, async () => {
    ort.env.wasm.numThreads = numThreads;
    const manifest = await fetchJson(`${modelBase}/decoder/MANIFEST.json`);
    const entry = findBucket(manifest, bucket, "decoder");
    const models = await loadGraphs(modelBase, "decoder", entry, ["encode", "decode"], "decoder");
    return ProtonPottsSampler.create(ort, manifest, bucket, models);
  });
}

/** Bucket for a structure: the smallest one that holds its kept residues. */
function probeBucket(pdbText, labels, requested = null) {
  // Build at the largest bucket first: the probe only needs lTotal, and the labels check runs against the loaded
  // residues, which does not depend on the bucket.
  const probe = buildProtonPottsInputs(pdbText, Math.max(...BUCKETS), labels);
  if (requested !== null && requested !== undefined) {
    // An explicit bucket (as the harness cells carry) must exist and hold the structure.
    if (!BUCKETS.includes(requested)) throw new Error(`bucket ${requested} is not one of ${BUCKETS.join(", ")}`);
    if (probe.lTotal > requested) throw new Error(`structure has ${probe.lTotal} kept residues; bucket ${requested} is too small`);
    return { bucket: requested, lTotal: probe.lTotal };
  }
  const bucket = pickBucket(probe.lTotal);
  if (bucket === null) throw new Error(`structure has ${probe.lTotal} kept residues; the largest bucket is ${Math.max(...BUCKETS)}`);
  return { bucket, lTotal: probe.lTotal };
}

function normalizeLabels(labels) {
  if (labels === null || labels === undefined) return null;
  if (!Array.isArray(labels)) throw new Error("labels must be an array (one entry per loaded residue, '' for none) or null");
  return labels.map((label) => label ?? "");
}

async function runScore({ modelBase, numThreads, pdbText, params }) {
  const labels = normalizeLabels(params.labels);
  const { bucket } = probeBucket(pdbText, labels, params.bucket);
  const scorer = await getScorer(modelBase, bucket, numThreads);
  const variants = params.variants || [];
  const variantNames = params.variantNames || variants.map((_, i) => `variant-${i + 1}`);
  if (variantNames.length !== variants.length) throw new Error("variantNames must match variants one to one");
  // Same call as protonpotts_page.mjs: score(pdbText, labels, variants). energies[0] is the native row.
  const out = await scorer.score(pdbText, labels, variants);
  return {
    bucket,
    l_total: out.lTotal,
    tokens: out.tokens,
    names: ["native", ...variantNames],
    energies: Array.from(out.energies),
  };
}

async function runDesign({ modelBase, numThreads, pdbText, params }) {
  const labels = normalizeLabels(params.labels);
  const { bucket, lTotal } = probeBucket(pdbText, labels, params.bucket);
  const config = params.config || {};
  const merged = { ...DEFAULT_CONFIG, ...config };
  const designer = await getDesigner(modelBase, bucket, numThreads);

  let uniforms = null;
  if (params.uniforms) {
    uniforms = params.uniforms.map((u) => Float32Array.from(u));
  } else if (merged.temperature > 0) {
    // One stream of blockMaxRounds * maxMutations per sample: the length the validated gates (X7b/X7c) inject.
    const seed = params.seed ?? 0;
    const length = merged.blockMaxRounds * merged.maxMutations;
    uniforms = Array.from({ length: merged.samplesPerSite }, (_, s) => seededStreams(seed + s, length).uniforms);
  }
  // Same call as ph_page.mjs: design(pdbText, labels, cell.config, uniforms). The config is passed unmerged; the
  // designer merges it over DEFAULT_CONFIG itself.
  const designs = await designer.design(pdbText, labels, config, uniforms);
  return {
    bucket,
    l_total: lTotal,
    designs: designs.map((d) => ({
      sequence: Array.from(d.sequence),
      symbols: Array.from(d.sequence, (t) => VOCABULARY[t]).join(""),
      sample: d.sample,
      method: d.method,
      label: d.label,
      designable: d.designable,
      pins: d.pins.map((p) => ({
        position: p.position,
        protIdx: p.protIdx,
        depIdxs: p.depIdxs,
        resId: p.resId,
        type: p.protonationType,
      })),
      final_potts_energy: d.finalPottsEnergy,
      selective_energy: d.selectiveEnergy,
      selective_energies: d.selectiveEnergies,
      n_draws: d.nDraws,
    })),
  };
}

async function runSample({ modelBase, numThreads, pdbText, params }) {
  const labels = normalizeLabels(params.labels);
  const { bucket } = probeBucket(pdbText, labels, params.bucket);
  const sampler = await getSampler(modelBase, bucket, numThreads);
  const seed = params.seed ?? 0;
  const temperature = params.temperature ?? DEFAULT_TEMPERATURE;
  // Same options as sampler_page.mjs builds from a cell: per-position temperature, designed mask, bias, and the
  // injected uniforms + noise (both or neither) are passed through with the same typed-array conversions.
  const options = { seed, temperature: Array.isArray(temperature) ? Float32Array.from(temperature) : temperature };
  if (params.designed) options.designed = params.designed.map((x) => x === 1 || x === true);
  if (params.bias) options.bias = Float32Array.from(params.bias);
  if (params.uniforms || params.noise) {
    options.uniforms = Float32Array.from(params.uniforms || []);
    options.noise = Float32Array.from(params.noise || []);
  }
  const out = await sampler.sample(pdbText, labels, options);
  return {
    bucket,
    l_total: out.lTotal,
    seed,
    temperature,
    sequence: Array.from(out.sequence),
    symbols: out.symbols,
    decoding_order: Array.from(out.decodingOrder),
    log_probs: Array.from(out.logProbs),
  };
}

const HANDLERS = new Map([
  ["score", runScore],
  ["design", runDesign],
  ["sample", runSample],
]);

function threadCount(value) {
  return Number.isInteger(value) && value >= 1 ? value : 1;
}

self.onmessage = async (event) => {
  const message = event.data || {};
  const { id } = message;
  try {
    const handler = HANDLERS.get(message.kind);
    if (!handler) throw new Error(`unknown kind ${JSON.stringify(message.kind)}`);
    if (typeof message.modelBase !== "string" || !message.modelBase) throw new Error("modelBase is required");
    if (typeof message.pdbText !== "string" || !message.pdbText) throw new Error("pdbText is required");
    const result = await handler({
      modelBase: message.modelBase,
      numThreads: threadCount(message.numThreads),
      pdbText: message.pdbText,
      params: message.params || {},
    });
    // What ORT reports AFTER the session exists: onnxruntime-web fixes its thread count at the first wasm init, so a
    // later request for a different count, or a fallback to 1 without cross-origin isolation, shows up here as data.
    self.postMessage({
      id, ok: true, result,
      env: {
        num_threads_effective: ort.env.wasm.numThreads,
        cross_origin_isolated: Boolean(self.crossOriginIsolated),
        ort_version: (ort.env.versions && ort.env.versions.web) || null,
      },
    });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String((e && e.stack) || e) });
  }
};
