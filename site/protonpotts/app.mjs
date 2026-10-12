// ProtonPottsMPNN in the browser: scoring, pH design and sequence sampling. Plain ES module, no bundler.
//
// This file owns the DOM and the worker. Every method call goes through runJob(), and window.__PP_RUN__ is that same
// function, so the buttons and the headless validation hook share one code path. The worker's protocol is documented
// at the top of worker.mjs.
import { MODEL_BASE } from "./config.js";

const el = (id) => document.getElementById(id);
const pdbFileInput = el("pdb-file");
const pdbTextInput = el("pdb-text");
const labelsInput = el("labels");
const threadsNote = el("threads-note");
const statusLine = el("status-line");
const runButtons = { score: el("score-run"), design: el("design-run"), sample: el("sample-run") };
const outputs = { score: el("score-out"), design: el("design-out"), sample: el("sample-out") };
const downloads = { score: el("score-download"), design: el("design-download"), sample: el("sample-download") };
const LABELS = { score: "Scoring", design: "pH design", sample: "Sampling" };

// Method-specific inputs. Keys match DEFAULT_CONFIG in ph_design.mjs.
const CV_METHODS = ["converged_mcmc", "converged_mcmc_combined", "two_phase"];
const METHOD_NOTES = {
  block_descent: "block_descent: validated in Chromium and as deployed (X7c, X10). Browser-drawn uniforms are not the validated stream.",
  converged_mcmc: "converged_mcmc: validated under Node against aminx on replayed draws (X11b, in progress). Browser runs use seeded draws, not the validated stream.",
  converged_mcmc_combined: "converged_mcmc_combined: validated under Node against aminx on replayed draws (X11b, in progress). Browser runs use seeded draws, not the validated stream.",
  two_phase: "two_phase: validated under Node against aminx on replayed draws (X11b, in progress). Browser runs use seeded draws, not the validated stream.",
  gibbs: "gibbs: validated under Node against aminx on replayed draws (X11b, in progress). Browser runs use seeded draws, not the validated stream.",
};
const designMethod = el("design-method");
const designMethodNote = el("design-method-note");

function refreshDesignMethod() {
  const method = designMethod.value;
  const show = (id, on) => {
    el(id).style.display = on ? "" : "none";
  };
  show("design-cv-patience-wrap", CV_METHODS.includes(method));
  show("design-cv-max-wrap", CV_METHODS.includes(method));
  show("design-combined-wrap", method === "converged_mcmc_combined");
  show("design-two-phase-wrap", method === "two_phase");
  designMethodNote.textContent = METHOD_NOTES[method] || "";
}
designMethod.addEventListener("change", refreshDesignMethod);
refreshDesignMethod();

const state = {
  fileText: null, // text of the uploaded file; takes precedence over the pasted text while set
  results: { score: null, design: null, sample: null },
};

// ---------------------------------------------------------------------
// Threads and status
// ---------------------------------------------------------------------

// ORT-web only runs multithreaded under cross-origin isolation; without it the worker gets one thread.
function computeNumThreads() {
  return self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
}

function refreshThreadsNote() {
  threadsNote.textContent = `threads: ${computeNumThreads()} (cross-origin isolated: ${Boolean(self.crossOriginIsolated)})`;
}

function setStatus(text, kind) {
  statusLine.textContent = text;
  statusLine.classList.remove("status-error", "status-ok");
  if (kind === "error") statusLine.classList.add("status-error");
  if (kind === "ok") statusLine.classList.add("status-ok");
}

function setBusy(busy) {
  for (const button of Object.values(runButtons)) button.disabled = busy;
}

// ---------------------------------------------------------------------
// Worker and the shared run path
// ---------------------------------------------------------------------

const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
const pending = new Map(); // id -> resolve
let jobCounter = 0;

worker.onmessage = (event) => {
  const reply = event.data;
  const resolve = pending.get(reply.id);
  if (!resolve) return;
  pending.delete(reply.id);
  resolve(reply);
};

worker.onerror = (event) => {
  const message = `Worker error: ${event.message || "unknown"}`;
  for (const [id, resolve] of pending) resolve({ id, ok: false, error: message });
  pending.clear();
  setStatus(message, "error");
  setBusy(false);
};

/**
 * Post one job to the worker. Resolves with the worker's reply object ({id, ok, result} or {id, ok, error}); it never
 * rejects, so a failed method is data for the caller to inspect.
 */
function runJob(msg) {
  const id = msg.id ?? `job-${++jobCounter}-${Date.now()}`;
  const outbound = {
    ...msg,
    id,
    modelBase: msg.modelBase ?? MODEL_BASE,
    numThreads: msg.numThreads ?? computeNumThreads(),
  };
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker.postMessage(outbound);
  });
}

// Headless validation hook and the one code path the buttons use.
window.__PP_RUN__ = (msg) => runJob(msg);
window.__PP_READY__ = true;

// ---------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------

function currentPdbText() {
  if (state.fileText !== null) return state.fileText;
  return pdbTextInput.value.trim() ? pdbTextInput.value : "";
}

pdbFileInput.addEventListener("change", async () => {
  const file = pdbFileInput.files[0];
  if (!file) {
    state.fileText = null;
    return;
  }
  try {
    state.fileText = await file.text();
    pdbTextInput.value = "";
    setStatus(`Loaded ${file.name} (${state.fileText.length} characters).`, "ok");
  } catch (error) {
    state.fileText = null;
    setStatus(`Could not read ${file.name}: ${error.message}`, "error");
  }
});

pdbTextInput.addEventListener("input", () => {
  state.fileText = null;
  pdbFileInput.value = "";
});

/** Labels: comma-separated, one per loaded residue, empty field = no label. Blank input = unlabelled (null). */
function parseLabels(text) {
  const trimmed = text.trim();
  if (!trimmed) return null;
  return trimmed.split(",").map((label) => label.trim());
}

/**
 * Score variants, one per line: `name: tokens` or just `tokens`. Tokens are separated by commas or whitespace; a body
 * with none of those is read one character per residue. Lines starting with # are comments. Each token is a one-letter
 * code or a v6 protonation token (HIS-P, HIS-S, HIS-A, ASP-P, ASP-D, ASP-A, GLU-P, GLU-D, GLU-A). Row length must equal
 * the number of kept residues; the worker checks it.
 */
function parseVariants(text) {
  const names = [];
  const rows = [];
  const lines = text
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  lines.forEach((line, i) => {
    let name = `variant-${i + 1}`;
    let body = line;
    const colon = line.indexOf(":");
    if (colon >= 0) {
      name = line.slice(0, colon).trim() || name;
      body = line.slice(colon + 1).trim();
    }
    if (!body) throw new Error(`variant "${name}" has no sequence`);
    const tokens = /[\s,]/u.test(body) ? body.split(/[\s,]+/u).filter(Boolean) : Array.from(body);
    names.push(name);
    rows.push(tokens);
  });
  return { names, rows };
}

function numberField(input, what) {
  const value = Number(input.value);
  if (input.value.trim() === "" || !Number.isFinite(value)) throw new Error(`${what} must be a number`);
  return value;
}

function collectParams(kind) {
  const labels = parseLabels(labelsInput.value);
  if (kind === "score") {
    const { names, rows } = parseVariants(el("score-variants").value);
    return { labels, variants: rows, variantNames: names };
  }
  if (kind === "design") {
    const binderChain = el("design-binder").value.trim();
    if (!binderChain) throw new Error("Binder chain is required: the chain whose residues are designed, e.g. A.");
    const temperature = numberField(el("design-temperature"), "Temperature");
    if (temperature < 0) throw new Error("Temperature must be zero or positive.");
    const method = el("design-method").value;
    // Defaults mirror DEFAULT_CONFIG in ph_design.mjs; the worker merges whatever is sent over it. Only the fields the
    // chosen method reads are sent, so the config never carries a knob that method ignores.
    const config = {
      binderChain,
      method,
      temperature,
      samplesPerSite: Math.trunc(numberField(el("design-samples"), "Samples per site")),
      maxMutations: Math.trunc(numberField(el("design-max-mutations"), "Max mutations")),
    };
    if (CV_METHODS.includes(method)) {
      config.cvPatience = Math.trunc(numberField(el("design-cv-patience"), "cvPatience"));
      config.cvMax = Math.trunc(numberField(el("design-cv-max"), "cvMax"));
    }
    if (method === "converged_mcmc_combined") {
      config.combinedLambda = numberField(el("design-combined-lambda"), "combinedLambda");
    }
    if (method === "two_phase") {
      config.twoPhaseFrac = numberField(el("design-two-phase-frac"), "twoPhaseFrac");
    }
    return {
      labels,
      seed: Math.trunc(numberField(el("design-seed"), "Seed")),
      config,
    };
  }
  if (kind === "sample") {
    const temperature = numberField(el("sample-temperature"), "Temperature");
    if (temperature <= 0) throw new Error("Temperature must be positive.");
    return {
      labels,
      seed: Math.trunc(numberField(el("sample-seed"), "Seed")),
      temperature,
    };
  }
  throw new Error(`unknown kind ${kind}`);
}

// ---------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------

function fillTable(container, headers, rows) {
  container.textContent = "";
  const table = document.createElement("table");
  table.className = "results-table";
  const headRow = table.createTHead().insertRow();
  for (const header of headers) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = header;
    headRow.appendChild(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const cell of row) tr.insertCell().textContent = String(cell);
  }
  container.appendChild(table);
}

function renderScore(result) {
  const rows = result.names.map((name, i) => [name, result.energies[i].toFixed(4)]);
  fillTable(outputs.score, ["Variant", "Potts energy"], rows);
  const note = document.createElement("p");
  note.className = "result-note";
  note.textContent = `Bucket ${result.bucket}, ${result.l_total} kept residues. Row 0 is the native sequence.`;
  outputs.score.appendChild(note);
}

function renderDesign(result) {
  if (result.designs.length === 0) {
    outputs.design.textContent = "No designable pH positions were found; nothing to design.";
    return;
  }
  const rows = result.designs.map((d, i) => [
    i + 1,
    d.sample,
    d.final_potts_energy.toFixed(4),
    d.selective_energy.toFixed(4),
    d.n_draws,
    d.symbols,
  ]);
  fillTable(outputs.design, ["#", "Sample", "Final energy", "Selective energy", "Draws", "Sequence"], rows);
  const note = document.createElement("p");
  note.className = "result-note";
  note.textContent = `Bucket ${result.bucket}, ${result.l_total} kept residues. Sorted by final energy.`;
  outputs.design.appendChild(note);
}

function renderSample(result) {
  const { log_probs: logProbs, sequence, l_total: length } = result;
  let total = 0;
  const N_TOKENS = 30; // the v6 vocabulary size, the row stride of log_probs
  sequence.forEach((token, i) => {
    total += logProbs[i * N_TOKENS + token];
  });
  const pre = document.createElement("pre");
  pre.className = "sequence-block";
  pre.textContent = result.symbols;
  outputs.sample.textContent = "";
  outputs.sample.appendChild(pre);
  const note = document.createElement("p");
  note.className = "result-note";
  note.textContent = `Bucket ${result.bucket}, ${length} residues, seed ${result.seed}, `
    + `temperature ${Number(result.temperature).toPrecision(4)}, mean log-prob of chosen tokens ${(total / length).toFixed(4)}.`;
  outputs.sample.appendChild(note);
}

function showError(kind, message) {
  outputs[kind].textContent = message;
  outputs[kind].classList.add("output-error");
}

function handleReply(kind, reply) {
  outputs[kind].classList.remove("output-error");
  if (!reply.ok) {
    showError(kind, reply.error);
    setStatus(`${LABELS[kind]} failed; the error is shown in its results section.`, "error");
    downloads[kind].disabled = true;
    state.results[kind] = null;
    return;
  }
  state.results[kind] = reply.result;
  if (kind === "score") renderScore(reply.result);
  if (kind === "design") renderDesign(reply.result);
  if (kind === "sample") renderSample(reply.result);
  downloads[kind].disabled = false;
  setStatus(`${LABELS[kind]} done.`, "ok");
}

function downloadJson(kind) {
  const result = state.results[kind];
  if (!result) return;
  const blob = new Blob([JSON.stringify({ kind, result }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `protonpotts_${kind}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// ---------------------------------------------------------------------
// Buttons: the same runJob() the hook uses
// ---------------------------------------------------------------------

async function runAction(kind) {
  const pdbText = currentPdbText();
  if (!pdbText) {
    setStatus("Load a PDB file or paste its text first.", "error");
    return;
  }
  let params;
  try {
    params = collectParams(kind);
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  setBusy(true);
  setStatus(`${LABELS[kind]} running (threads: ${computeNumThreads()})...`, "");
  const reply = await runJob({ kind, pdbText, params });
  setBusy(false);
  handleReply(kind, reply);
}

runButtons.score.addEventListener("click", () => runAction("score"));
runButtons.design.addEventListener("click", () => runAction("design"));
runButtons.sample.addEventListener("click", () => runAction("sample"));
downloads.score.addEventListener("click", () => downloadJson("score"));
downloads.design.addEventListener("click", () => downloadJson("design"));
downloads.sample.addEventListener("click", () => downloadJson("sample"));

refreshThreadsNote();
setStatus("Load a structure to begin.", "");
