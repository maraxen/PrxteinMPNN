// ProtonPotts pH design in the browser: block descent with the per-block graphs (ADR decision 11).
//
// JS port of src/aminx/families/protonpotts_mpnn/ph_design.py (design_structure) for method "block_descent" on the Potts
// backend. The graphs do the arithmetic; this module owns the plan, the sweep, the convergence test, the uniforms and the
// bookkeeping, exactly as the host loop of tests/protonpotts/test_ph_export.py does:
//
//   table graph -> merged pair table -> field_at (chunks) -> plan (ph_plan.mjs) -> zblock per block -> pool
//   -> sweeps of visit calls until a sweep changes nothing -> energy graph and field_at for the score.
//
// Graph contracts: scripts/browser_validation/protonpotts_ph_export_gate.py (names and shapes in the manifest).

import { ProtonPottsInputError, VOCABULARY, buildProtonPottsInputs } from "./protonpotts_inputs.mjs";
import {
  DEFAULT_DEP_MAP,
  PhPlanError,
  blockTable,
  energyRank,
  knnRank,
  planFromCenterTypes,
  planFromExplicitCenters,
  validTokenMask,
} from "./ph_plan.mjs";
import { run } from "./protonpotts_scorer.mjs";

const SENTINEL = 1_000_000;
const THREE_OF = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};
const f32 = Math.fround;

export const SWEEP_ORDERS = Object.freeze(["position", "knn", "energy"]);

export const DEFAULT_CONFIG = Object.freeze({
  binderChain: null,
  centerTypes: ["HIS-P", "ASP-P", "GLU-P"],
  explicitCenters: [],
  depMap: DEFAULT_DEP_MAP,
  forbiddenTokens: ["HIS-A", "ASP-A", "GLU-A", "UNK"],
  temperature: 0.05,
  samplesPerSite: 2,
  combinedLambda: 0.3,
  blockSize: 3,
  neighbourK: 16,
  maxMutations: 20,
  infillScope: "neighbourhood",
  repetitiveWindowParents: ["ARG", "LYS", "HIS", "ASP", "GLU"],
  blockMaxRounds: 10,
  method: "block_descent",
  backend: "potts",
  selective: true,
  twoPhaseFrac: 0.5,
  cvPatience: 3,
  cvMax: 50,
  placementBy: "scan_potts",
  sweepOrder: "position",
});

/** Parent three-letter residue of a token index (X has none): HIS-P is HIS, K is LYS. */
function parentOf(index) {
  const symbol = VOCABULARY[index];
  if (symbol === "X") return null;
  return symbol.length > 1 ? symbol.slice(0, 3) : THREE_OF[symbol];
}

/** (V,) float32 mask, 1.0 where a token's parent residue is in `parents` (port of ph_descent.rep_class_mask). */
export function repClassMask(parents) {
  const wanted = new Set(parents);
  return Float32Array.from(VOCABULARY, (_s, i) => {
    const name = parentOf(i);
    return name !== null && wanted.has(name) ? 1 : 0;
  });
}

export class PhDesigner {
  /**
   * @param {object} ort onnxruntime-web
   * @param {object} manifest the PH manifest (family "protonpottsmpnn_ph"), already restricted to this bucket's entry
   * @param {object} scoringEntry the scoring manifest's entry for the same bucket (table graph meta)
   * @param {number} bucket
   * @param {{table: Uint8Array, graphs: Object<string, Uint8Array>}} models  graph bytes keyed by manifest graph name
   */
  constructor(ort, manifest, scoringEntry, bucket, sessions) {
    this.ort = ort;
    this.manifest = manifest;
    this.scoringEntry = scoringEntry;
    this.bucket = bucket;
    this.sessions = sessions; // { table, energy, [graphName]: InferenceSession }
    this.entry = manifest.buckets.find((b) => b.bucket === bucket);
    if (!this.entry) throw new Error(`PH manifest has no bucket ${bucket}`);
  }

  static async create(ort, manifest, scoringEntry, bucket, models) {
    const sessions = { table: await ort.InferenceSession.create(models.table), energy: await ort.InferenceSession.create(models.energy) };
    const entry = manifest.buckets.find((b) => b.bucket === bucket);
    for (const name of Object.keys(entry.graphs)) sessions[name] = await ort.InferenceSession.create(models.graphs[name]);
    return new PhDesigner(ort, manifest, scoringEntry, bucket, sessions);
  }

  async _run(graphName, named) {
    return run(this.ort, this.sessions[graphName], this.entry.graphs[graphName], named);
  }

  /** Conditional energies of the whole chain: chunked field_at calls (never the scatter-add full-chain function). */
  async _field(table, eIdx, seq, length) {
    const { chunk } = this.manifest;
    const vocab = VOCABULARY.length;
    const field = new Float32Array(this.bucket * vocab);
    for (let start = 0; start < length; start += chunk) {
      const positions = new Int32Array(chunk);
      for (let k = 0; k < chunk; k += 1) positions[k] = Math.min(start + k, length - 1);
      const [rows] = await this._run(`field_L${this.bucket}`, { table, e_idx: eIdx, seq: { data: seq }, positions: { data: positions } });
      for (let k = 0; k < Math.min(chunk, length - start); k += 1) {
        field.set(rows.data.subarray(k * vocab, (k + 1) * vocab), (start + k) * vocab);
      }
    }
    return field;
  }

  /**
   * Design one structure. `labels` as buildProtonPottsInputs. `uniforms` is an array of Float32Array, one per sample, needed when
   * config.temperature > 0 (the replay contract of design_structure); at temperature <= 0 none are used.
   * Returns designs sorted by final Potts energy (stable), like design_structure.
   */
  async design(pdbText, labels, config = {}, uniforms = null, replay = null) {
    // Methods other than block_descent route to the host-loop port below (_designMethods); block_descent is untouched.
    if ((config.method ?? DEFAULT_CONFIG.method) !== "block_descent") {
      return this._designMethods(pdbText, labels, { ...DEFAULT_CONFIG, ...config }, replay);
    }
    const cfg = { ...DEFAULT_CONFIG, ...config };
    const built = buildProtonPottsInputs(pdbText, this.bucket, labels);
    const L = built.lTotal;
    const B = this.bucket;
    const vocab = VOCABULARY.length;
    if (cfg.blockSize !== this.manifest.block_size) throw new PhPlanError(`block_size ${cfg.blockSize}: this export has ${this.manifest.block_size}`);
    if (!cfg.binderChain) throw new PhPlanError("binderChain is required for pH design");
    if (!SWEEP_ORDERS.includes(cfg.sweepOrder)) {
      throw new PhPlanError(`unknown sweepOrder ${JSON.stringify(cfg.sweepOrder)}; use one of ${SWEEP_ORDERS.join(", ")}`);
    }
    const sampling = cfg.temperature > 0;
    if (sampling && (!uniforms || uniforms.length !== cfg.samplesPerSite)) {
      throw new PhPlanError(`temperature > 0 needs one uniform array per sample (${cfg.samplesPerSite})`);
    }

    const [table, eIdxT] = await run(this.ort, this.sessions.table, this.scoringEntry.graphs.table, built.inputs);
    const eIdx = Int32Array.from(eIdxT.data);
    const seq = new Int32Array(B);
    seq.set(built.tokens);
    const binder = new Uint8Array(B);
    const resId = new Int32Array(B);
    built.kept.forEach(([chain, number], i) => {
      binder[i] = chain === cfg.binderChain ? 1 : 0;
      resId[i] = number;
    });
    if (!binder.some((x) => x)) throw new ProtonPottsInputError(`binder chain ${cfg.binderChain} has no designable position`);

    const slots = eIdx.length / B;
    const planOptions = { infillScope: cfg.infillScope, neighbourK: cfg.neighbourK, maxMutations: cfg.maxMutations };
    let plan;
    if (cfg.explicitCenters.length) {
      plan = planFromExplicitCenters(eIdx, B, slots, binder, resId, cfg.explicitCenters, cfg.depMap, planOptions);
    } else {
      if (!cfg.centerTypes.length) throw new PhPlanError("block_descent needs centerTypes or explicitCenters");
      const field = await this._field(table, { data: eIdx }, seq, L);
      plan = planFromCenterTypes(field, vocab, eIdx, B, slots, binder, resId, cfg.centerTypes, cfg.depMap, planOptions);
    }
    if (!plan || plan.designable.length === 0) return [];

    const P = plan.pins.length;
    const depth = this.manifest.depth;
    const pinPositions = Int32Array.from(plan.pins.map((p) => p.position));
    const pinProt = Int32Array.from(plan.pins.map((p) => p.protIdx));
    const pinDep = new Int32Array(P * depth);
    const pinDepValid = new Uint8Array(P * depth);
    plan.pins.forEach((pin, i) => {
      pin.depIdxs.forEach((d, j) => {
        pinDep[i * depth + j] = d;
        pinDepValid[i * depth + j] = 1;
      });
    });
    const validTokens = validTokenMask(cfg.forbiddenTokens);
    const repMask = repClassMask(cfg.repetitiveWindowParents);
    const residueNumber = Int32Array.from({ length: B }, (_, i) => (binder[i] ? resId[i] : -SENTINEL - 100 * i));
    const pinned = (s) => {
      const out = Int32Array.from(s);
      pinPositions.forEach((p, i) => {
        out[p] = pinProt[i];
      });
      return out;
    };
    // Visit order (aminx ph_design._visit_order): position keeps plan.designable; knn and energy reorder it once, on the pinned seed,
    // before the z-scales and the block table are built. plan.designable itself (the reported `designable`) is never reordered.
    let visit = plan.designable;
    if (cfg.sweepOrder === "knn") {
      visit = knnRank(eIdx, slots, plan.designable, plan.pins);
    } else if (cfg.sweepOrder === "energy") {
      if (table.dims && (table.dims.length !== 4 || table.dims[0] !== B || table.dims[1] !== slots || table.dims[2] !== vocab || table.dims[3] !== vocab)) {
        throw new PhPlanError(`merged table dims [${table.dims}] are not [${B}, ${slots}, ${vocab}, ${vocab}]`);
      }
      visit = energyRank(table.data, eIdx, slots, vocab, plan.pins, pinned(seq), plan.designable);
    }
    const { blocks, valid } = blockTable(eIdx, slots, visit, cfg.blockSize);
    const nBlocks = visit.length;
    const tableIn = { data: table.data };
    const common = {
      table: tableIn,
      e_idx: { data: eIdx },
      pin_positions: { data: pinPositions },
      pin_prot: { data: pinProt },
      pin_dep: { data: pinDep },
      pin_dep_valid: { data: pinDepValid },
      valid_tokens: { data: validTokens },
    };
    const blockAt = (n) => ({
      block: { data: blocks.slice(n * cfg.blockSize, (n + 1) * cfg.blockSize) },
      block_valid: { data: valid.slice(n * cfg.blockSize, (n + 1) * cfg.blockSize) },
    });

    // z-scales once from the pinned seed sequence (the same for every sample), then the weights
    const seed = pinned(seq);
    const varH = new Float32Array(B);
    const varS = new Float32Array(B);
    const hasH = new Int32Array(B);
    const hasS = new Int32Array(B);
    for (let n = 0; n < nBlocks; n += 1) {
      const [vh, hh, vs, hs] = await this._run(`zblock_L${B}_P${P}`, { ...common, seq: { data: seed }, ...blockAt(n) });
      varH[n] = vh.data[0];
      hasH[n] = hh.data[0];
      varS[n] = vs.data[0];
      hasS[n] = hs.data[0];
    }
    const lam = cfg.combinedLambda;
    const [zscales, wh, wsel] = await this._run(`pool_L${B}`, {
      var_h: { data: varH }, has_h: { data: hasH }, var_s: { data: varS }, has_s: { data: hasS },
      w_h: { data: Float32Array.of(f32(1 - lam)) }, w_s: { data: Float32Array.of(f32(lam)) },
    });

    const designs = [];
    for (let sample = 0; sample < cfg.samplesPerSite; sample += 1) {
      const u = sampling ? uniforms[sample] : null;
      const work = pinned(seq);
      let draws = 0;
      let rounds = 0;
      let changed = true;
      while (changed && rounds < cfg.blockMaxRounds) {
        changed = false;
        for (let n = 0; n < nBlocks; n += 1) {
          const uniform = sampling ? u[Math.min(draws, u.length - 1)] : 0;
          const [digits] = await this._run(`visit_L${B}_P${P}`, {
            ...common,
            seq: { data: work },
            ...blockAt(n),
            rep_mask: { data: repMask },
            residue_number: { data: residueNumber },
            wh: { data: wh.data },
            wsel: { data: wsel.data },
            uniform: { data: Float32Array.of(uniform) },
            temperature: { data: Float32Array.of(cfg.temperature) },
          });
          if (sampling) draws += 1;
          for (let c = 0; c < cfg.blockSize; c += 1) {
            if (!valid[n * cfg.blockSize + c]) continue;
            const pos = blocks[n * cfg.blockSize + c];
            if (digits.data[c] !== work[pos]) changed = true;
            work[pos] = digits.data[c];
          }
        }
        rounds += 1;
      }
      designs.push(await this._score(table, eIdx, work, plan, sample, draws, L, visit));
    }
    // stable sort by final energy (ties keep plan-then-sample order)
    return designs.map((d, i) => [d, i]).sort((a, b) => a[0].finalPottsEnergy - b[0].finalPottsEnergy || a[1] - b[1]).map(([d]) => d);
  }

  async _score(table, eIdx, work, plan, sample, draws, length, visit = null) {
    const rows = this.scoringEntry.graphs.energy.input_shapes[3][0];
    const seqs = new Int32Array(rows * this.bucket);
    seqs.set(work.subarray(0, this.bucket), 0);
    const [energies] = await run(this.ort, this.sessions.energy, this.scoringEntry.graphs.energy, {
      table: { data: table.data },
      e_idx: { data: eIdx },
      pad_valid: { data: Uint8Array.from({ length: this.bucket }, (_, i) => (i < length ? 1 : 0)) },
      sequences: { data: seqs },
    });
    const gaps = [];
    if (plan.pins.length) {
      const { chunk } = this.manifest;
      const positions = new Int32Array(chunk);
      plan.pins.forEach((p, i) => {
        positions[i] = p.position;
      });
      const [fieldRows] = await this._run(`field_L${this.bucket}`, { table: { data: table.data }, e_idx: { data: eIdx }, seq: { data: work }, positions: { data: positions } });
      const vocab = VOCABULARY.length;
      plan.pins.forEach((p, i) => {
        // numpy: float32 row values, mean in float32, difference in float32, then float()
        let sum = 0;
        for (const d of p.depIdxs) sum = f32(sum + fieldRows.data[i * vocab + d]);
        const mean = f32(sum / p.depIdxs.length);
        gaps.push(f32(fieldRows.data[i * vocab + p.protIdx] - mean));
      });
    }
    return {
      sequence: Int32Array.from(work.subarray(0, length)),
      method: "block_descent",
      sample,
      pins: plan.pins,
      designable: plan.designable,
      label: plan.label,
      finalPottsEnergy: energies.data[0],
      selectiveEnergy: gaps.reduce((a, b) => a + b, 0),
      selectiveEnergies: gaps,
      nDraws: draws,
      ...(visit ? { visitOrder: Array.from(visit) } : {}),
    };
  }

  /** Conditional Potts rows (Python MethodContext.rows): field_L graph, positions padded to the graph's own length by repeating the last. */
  async _rows(table, eIdx, seq, positions) {
    const name = `field_L${this.bucket}`;
    const meta = this.entry.graphs[name];
    const nPos = meta.input_shapes[meta.input_names.indexOf("positions")][0];
    if (positions.length > nPos) throw new PhPlanError(`${positions.length} rows requested, the ${name} graph takes ${nPos}`);
    const padded = new Int32Array(nPos);
    for (let k = 0; k < nPos; k += 1) padded[k] = positions[Math.min(k, positions.length - 1)];
    const [rows] = await this._run(name, { table, e_idx: eIdx, seq: { data: seq }, positions: { data: padded } });
    const vocab = VOCABULARY.length;
    return positions.map((_, k) => Float32Array.from(rows.data.subarray(k * vocab, (k + 1) * vocab)));
  }

  /**
   * pH host-loop methods (converged_mcmc, converged_mcmc_combined, two_phase, gibbs) on the Potts backend: the port of
   * ph_methods.design_methods. Returns designs sorted by final Potts energy (stable), as design_methods does.
   * `replay`: one {uniforms, randints, perms} per sample (gibbs uses replay[0]); null draws a seeded stream instead (NOT the validated one).
   */
  async _designMethods(pdbText, labels, cfg, replay) {
    const method = cfg.method;
    if (method === "autoregressive") {
      throw new PhPlanError("method 'autoregressive' needs the decoder field (backend 'mpnn'); the browser pH host loop does not port it");
    }
    if (!NEW_METHODS.includes(method)) {
      throw new PhPlanError(`unknown pH method ${JSON.stringify(method)}; the browser port runs block_descent and ${NEW_METHODS.join(", ")}`);
    }
    if (cfg.backend !== "potts") throw new PhPlanError(`method ${method} is graded on backend 'potts' only (got ${JSON.stringify(cfg.backend)})`);
    if (cfg.placementBy !== "scan_potts") throw new PhPlanError(`placementBy ${JSON.stringify(cfg.placementBy)} is not ported; use 'scan_potts'`);
    if (method !== "gibbs" && !cfg.selective) {
      throw new PhPlanError(`method ${method} with selective=false needs the non-selective planner, which ph_plan.mjs does not port`);
    }
    if (method === "two_phase" && !(cfg.twoPhaseFrac > 0 && cfg.twoPhaseFrac <= 1)) {
      throw new PhPlanError(`two_phase_frac must be in (0, 1], got ${cfg.twoPhaseFrac}`);
    }
    if (!cfg.binderChain) throw new PhPlanError("binderChain is required for pH design");
    const n = cfg.samplesPerSite;
    if (replay !== null && replay.length !== n) throw new PhPlanError(`replay has ${replay.length} entries, expected samplesPerSite=${n}`);

    const built = buildProtonPottsInputs(pdbText, this.bucket, labels);
    const L = built.lTotal;
    const B = this.bucket;
    const [tableT, eIdxT] = await run(this.ort, this.sessions.table, this.scoringEntry.graphs.table, built.inputs);
    const eIdx = Int32Array.from(eIdxT.data);
    const table = { data: tableT.data };
    const eIdxIn = { data: eIdx };
    const seq = new Int32Array(B);
    seq.set(built.tokens);
    const binder = new Uint8Array(B);
    const resId = new Int32Array(B);
    built.kept.forEach(([chain, number], i) => {
      binder[i] = chain === cfg.binderChain ? 1 : 0;
      resId[i] = number;
    });
    if (!binder.some((x) => x)) throw new ProtonPottsInputError(`binder chain ${cfg.binderChain} has no designable position`);

    const ctx = {
      valid: validTokenMask(cfg.forbiddenTokens),
      cdfOrder: upstreamCdfOrder(),
      rows: (s, positions) => this._rows(table, eIdxIn, s, positions),
    };
    ctx.fieldAt = async (s, i) => (await ctx.rows(s, [i]))[0];
    const drawsFor = (sample) => {
      if (replay !== null) {
        const r = replay[sample];
        return new Draws({ uniforms: r.uniforms ?? [], randints: r.randints ?? [], perms: r.perms ?? [], cdfOrder: ctx.cdfOrder });
      }
      return new Draws({ seed: (cfg.seed ?? 0) + sample, cdfOrder: ctx.cdfOrder });
    };

    const designs = [];
    const exhausted = [];
    if (method === "gibbs") {
      const free = [];
      for (let i = 0; i < B; i += 1) if (binder[i]) free.push(i);
      const draws = drawsFor(0); // one Draws shared by every row, as Python's draws_for(0)
      const outs = [];
      const rowEvents = [];
      for (let s = 0; s < n; s += 1) {
        const first = draws.events.length;
        try {
          outs.push(await gibbsOptimize(ctx, cfg, seq, free, draws));
        } catch (err) {
          if (!(err instanceof DrawsError)) throw err;
          outs.push(null); // replay exhausted in this row: recorded as a divergence, not a crash
        }
        rowEvents.push(draws.events.slice(first)); // this row's draws only, so a mismatch is located within its own row
      }
      for (let s = 0; s < n; s += 1) {
        if (outs[s] === null) {
          exhausted.push({ sample: s, method, replayExhausted: true, argmins: [], events: rowEvents[s], nDraws: draws.events.length, sequence: null, pins: [], designable: free, label: "" });
          continue;
        }
        // nDraws is the whole call's draw count (Python finish(..., len(draws.events)) after every row), events are the row's own.
        designs.push(await this._finishMethod(table, eIdxIn, outs[s], [], free, "", rowEvents[s], s, method, L, draws.events.length));
      }
    } else {
      const slots = eIdx.length / B;
      const planOptions = { infillScope: cfg.infillScope, neighbourK: cfg.neighbourK, maxMutations: cfg.maxMutations };
      let plan;
      if (cfg.explicitCenters.length) {
        plan = planFromExplicitCenters(eIdx, B, slots, binder, resId, cfg.explicitCenters, cfg.depMap, planOptions);
      } else {
        if (!cfg.centerTypes.length) throw new PhPlanError(`${method} needs centerTypes or explicitCenters`);
        const field = await this._field(table, eIdxIn, seq, L);
        plan = planFromCenterTypes(field, VOCABULARY.length, eIdx, B, slots, binder, resId, cfg.centerTypes, cfg.depMap, planOptions);
      }
      if (!plan || plan.designable.length === 0) return [];
      for (let sample = 0; sample < n; sample += 1) {
        const draws = drawsFor(sample);
        let out;
        try {
          out = method === "two_phase"
            ? await twoPhase(ctx, cfg, plan.pins, plan.designable, seq, draws)
            : await converge(ctx, cfg, plan.pins, plan.designable, seq, draws, { method });
        } catch (err) {
          if (!(err instanceof DrawsError)) throw err;
          // replay exhausted: no sequence, but the argmin trace and draws so far are kept for the gate's divergence check
          exhausted.push({ sample, method, replayExhausted: true, argmins: draws.argmins.slice(), events: draws.events.slice(), nDraws: draws.events.length, sequence: null, pins: plan.pins, designable: plan.designable, label: plan.label });
          continue;
        }
        designs.push(await this._finishMethod(table, eIdxIn, out, plan.pins, plan.designable, plan.label, draws.events, sample, method, L, undefined, draws.argmins.slice()));
      }
    }
    // stable sort by final energy, as design_methods (designs.sort(key=final_potts_energy)); replay-exhausted designs follow, in sample order
    const sorted = designs.map((d, i) => [d, i]).sort((a, b) => a[0].finalPottsEnergy - b[0].finalPottsEnergy || a[1] - b[1]).map(([d]) => d);
    return [...sorted, ...exhausted];
  }

  /** Final energy (energy graph, as _score) and centre gaps (Python-float _gap over field_L rows) of one method design. */
  async _finishMethod(table, eIdxIn, seqFull, pins, designable, label, events, sample, method, length, nDraws = events.length, argmins = []) {
    const rows = this.scoringEntry.graphs.energy.input_shapes[3][0];
    const seqs = new Int32Array(rows * this.bucket);
    seqs.set(seqFull.subarray(0, this.bucket), 0);
    const [energies] = await run(this.ort, this.sessions.energy, this.scoringEntry.graphs.energy, {
      table,
      e_idx: eIdxIn,
      pad_valid: { data: Uint8Array.from({ length: this.bucket }, (_, i) => (i < length ? 1 : 0)) },
      sequences: { data: seqs },
    });
    const gaps = pins.length
      ? (await this._rows(table, eIdxIn, seqFull, pins.map((p) => p.position))).map((row, k) => gapPy(row, pins[k]))
      : [];
    return {
      sequence: Int32Array.from(seqFull.subarray(0, length)),
      method,
      sample,
      pins,
      designable,
      label,
      finalPottsEnergy: energies.data[0],
      selectiveEnergy: gaps.reduce((a, b) => a + b, 0),
      selectiveEnergies: gaps,
      nDraws,
      events: events.slice(),
      argmins,
      replayExhausted: false,
    };
  }
}

// ---- pH host-loop methods on the Potts backend: port of src/aminx/families/protonpotts_mpnn/ph_methods.py -------------------------------
//
// Arithmetic follows numpy/JAX: float32 wherever the Python arrays are float32 (conditional rows, scores, softmax, the gibbs cumulative),
// JS doubles where Python uses float() (_gap, disruption, the inverse-CDF cumulative of Draws.pickProbs, float64 in numpy).
// Fresh draws (replay null) use a seeded mulberry32 stream, NOT numpy's PCG64 stream: they are reproducible here but are not the validated
// stream. The validated stream is the recorded replay.

export const NEW_METHODS = ["converged_mcmc", "converged_mcmc_combined", "two_phase", "gibbs"];
const MIN_TEMPERATURE = 1e-3; // ph_methods.py:64 (UP:1927; potts_gibbs_optimize's caller UP:1217)
const MAX_GIBBS_SWEEPS = 1000; // ph_methods.py:66 (UP:1217 max_iters)
// vocab.py:70-91 FOUNDRY_PREFIX_ORDER (atomworks STANDARD_AA + UNK); the nine v6 tokens follow at upstream indices 21..29.
const FOUNDRY_PREFIX_ORDER = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"];
const ONE_OF_THREE = new Map(Object.entries(THREE_OF).map(([one, three]) => [three, one]));

export class DrawsError extends Error {
  constructor(message) {
    super(message);
    this.name = "DrawsError";
  }
}

/** Python round() on a float: round half to even (round(0.5) = 0, round(1.5) = 2, round(2.5) = 2). */
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** aminx index of the j-th upstream token: _cdf_order() = argsort(token_permutation()) = upstream_to_aminx_index (vocab.py:140, ph_sample.py:75). */
export function upstreamCdfOrder() {
  const index = new Map(VOCABULARY.map((symbol, i) => [symbol, i]));
  const V = VOCABULARY.length;
  const out = new Int32Array(V);
  FOUNDRY_PREFIX_ORDER.forEach((name, j) => {
    out[j] = name === "UNK" ? index.get("X") : index.get(ONE_OF_THREE.get(name));
  });
  for (let j = FOUNDRY_PREFIX_ORDER.length; j < V; j += 1) out[j] = index.get(VOCABULARY[j]); // v6 tokens: upstream j = aminx j
  return out;
}

/** mulberry32: a small seeded stream for fresh draws. Not numpy's PCG64 (see the section header). */
function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The randomness of one optimiser call (Python Draws): recorded uniforms, randints and permutations consumed in order, or a seeded
 * fresh stream when a kind has no record. `events` collects {uniform, choice} for every multinomial pick (gibbs: every position).
 */
export class Draws {
  constructor({ seed = null, uniforms = null, randints = null, perms = null, cdfOrder = null } = {}) {
    if (seed === null && uniforms === null && randints === null && perms === null) {
      throw new DrawsError("Draws needs a seed (fresh) or recorded draws (uniforms, randints or perms)");
    }
    this.rng = seed === null ? null : seededRandom(seed);
    this.sources = { uniform: uniforms, randint: randints, perm: perms };
    this.used = { uniform: 0, randint: 0, perm: 0 };
    this.cdfOrder = cdfOrder; // aminx index of the j-th upstream token; null = identity order
    this.events = [];
    this.argmins = []; // X11b amendment 3: the chosen token of every T <= 0 pick (not a draw; events and nDraws are unchanged)
  }

  _next(kind, fresh) {
    const source = this.sources[kind];
    if (source === null) {
      if (this.rng === null) throw new DrawsError(`no recorded ${kind} and no seed`);
      return fresh();
    }
    const index = this.used[kind];
    if (index >= source.length) throw new DrawsError(`replay ran out of ${kind} draws after ${index}`);
    this.used[kind] = index + 1;
    return source[index];
  }

  /** The next n uniforms (gibbs sweeps). Shares its counter with the per-pick uniforms, as Python's take_uniforms does. */
  takeUniforms(n) {
    const source = this.sources.uniform;
    if (source === null) {
      if (this.rng === null) throw new DrawsError("no recorded uniform and no seed");
      return Array.from({ length: n }, () => this.rng());
    }
    const start = this.used.uniform;
    const stop = start + n;
    if (stop > source.length) throw new DrawsError(`replay ran out of uniform draws after ${start}`);
    this.used.uniform = stop;
    return source.slice(start, stop).map(Number);
  }

  /** Index into a list of `high` neighbours; a recorded value must lie in [0, high). */
  randint(high) {
    const value = this._next("randint", () => Math.floor(this.rng() * high));
    if (!Number.isInteger(value) || value < 0 || value >= high) {
      throw new DrawsError(`recorded randint ${value} outside [0, ${high})`);
    }
    return value;
  }

  /** A permutation of range(size); a recorded one must have length size (Python validates the length only). */
  permutation(size) {
    const perm = this._next("perm", () => {
      const a = Array.from({ length: size }, (_, i) => i);
      for (let i = size - 1; i > 0; i -= 1) {
        const j = Math.floor(this.rng() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    });
    if (perm.length !== size) throw new DrawsError(`recorded permutation has length ${perm.length}, expected ${size}`);
    return Array.from(perm, Number);
  }

  /**
   * One multinomial draw from probs (aminx token order): cumulative sum in float64 over probs taken in cdfOrder, position = first index
   * whose cumulative mass reaches u * total (np.searchsorted side='left'), clamped to the last; choice = cdfOrder[position].
   */
  pickProbs(probs) {
    const u = this._next("uniform", () => this.rng());
    const n = probs.length;
    const order = this.cdfOrder ?? Array.from({ length: n }, (_, i) => i);
    const cumulative = new Float64Array(n);
    let acc = 0;
    for (let j = 0; j < n; j += 1) {
      acc += probs[order[j]];
      cumulative[j] = acc;
    }
    const target = u * cumulative[n - 1];
    let position = 0;
    while (position < n && cumulative[position] < target) position += 1;
    position = Math.min(position, n - 1);
    const choice = order[position];
    this.events.push({ uniform: u, choice, probs: Array.from(probs) });
    return choice;
  }
}

/**
 * softmax(-score / T) in float32, as numpy computes it on a float32 score: z = -score / f32(T), z -= max(z), p = exp(z), p /= sum(p).
 * +Infinity scores get probability 0. The sum is a sequential float32 sum; numpy's pairwise sum can differ in the last ulp (the gate's
 * fragile-pick margin covers that), and Math.exp is rounded to float32 from a double exp (numpy's float32 exp may differ by an ulp).
 */
export function softmaxProbs(score, temperature) {
  const t = f32(temperature);
  const n = score.length;
  const z = new Float32Array(n);
  let m = -Infinity;
  for (let i = 0; i < n; i += 1) {
    z[i] = f32(-score[i] / t);
    if (z[i] > m) m = z[i];
  }
  if (m === -Infinity) throw new DrawsError("softmax of a score with no finite entry");
  const p = new Float32Array(n);
  let s = 0;
  for (let i = 0; i < n; i += 1) {
    p[i] = f32(Math.exp(f32(z[i] - m)));
    s = f32(s + p[i]);
  }
  for (let i = 0; i < n; i += 1) p[i] = f32(p[i] / s);
  return p;
}

/** Upstream _pick (UP:1774): the first argmin at T <= 0 (no draw), else one multinomial draw from softmax(-score / T). */
export function pick(score, temperature, draws) {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < score.length; i += 1) if (score[i] < score[best]) best = i;
    if (draws) draws.argmins.push(best);
    return best;
  }
  return draws.pickProbs(softmaxProbs(score, temperature));
}

/** _gap (ph_methods.py:320): e_P - mean_d e_D of one centre, in Python floats as upstream accumulates it. */
function gapPy(row, pin) {
  const eP = row[pin.protIdx];
  let ed = 0;
  for (const d of pin.depIdxs) ed += row[d];
  return eP - ed / pin.depIdxs.length;
}

/** Score row with invalid tokens at +Infinity (np.where(ctx.valid, score, inf)). */
function maskInf(valid, score) {
  const out = new Float32Array(score.length);
  for (let k = 0; k < score.length; k += 1) out[k] = valid[k] ? score[k] : Infinity;
  return out;
}

/** _score_at (ph_methods.py:426-446): the stability row, minus the mean deprotonated-centre row when selective (or blended by lam). */
async function scoreAt(ctx, seq, i, pins, { selective, lam }) {
  for (const p of pins) seq[p.position] = p.protIdx;
  const eStab = Float32Array.from(await ctx.fieldAt(seq, i));
  const V = eStab.length;
  if (!selective && lam === null) return maskInf(ctx.valid, eStab);
  const ed = new Float32Array(V);
  for (const p of pins) {
    const c = new Float32Array(V);
    for (const d of p.depIdxs) {
      seq[p.position] = d;
      const row = await ctx.fieldAt(seq, i);
      for (let k = 0; k < V; k += 1) c[k] = f32(c[k] + row[k]);
    }
    seq[p.position] = p.protIdx;
    const nDeps = p.depIdxs.length;
    for (let k = 0; k < V; k += 1) ed[k] = f32(ed[k] + f32(c[k] / nDeps));
  }
  for (let k = 0; k < V; k += 1) ed[k] = f32(ed[k] / pins.length);
  const score = new Float32Array(V);
  if (lam !== null) {
    const a = f32(1 + lam);
    const b = f32(lam);
    for (let k = 0; k < V; k += 1) score[k] = f32(f32(a * eStab[k]) - f32(b * ed[k]));
  } else {
    for (let k = 0; k < V; k += 1) score[k] = f32(eStab[k] - ed[k]);
  }
  return maskInf(ctx.valid, score);
}

/** _converge (ph_methods.py:449-475): random single-site moves until cv_patience*N unchanged steps or cv_max*N steps. Returns the sequence. */
async function converge(ctx, cfg, pins, neigh, initial, draws, opts = {}) {
  const method = opts.method ?? cfg.method;
  const lam = method === "converged_mcmc_combined" ? cfg.combinedLambda : null;
  const selective = lam === null ? (opts.selective ?? cfg.selective) : false;
  const seq = Int32Array.from(opts.start ?? initial);
  for (const p of pins) seq[p.position] = p.protIdx;
  const list = Array.from(neigh, Number);
  const n = list.length;
  const patience = cfg.cvPatience * n;
  const cap = cfg.cvMax * n;
  let since = 0;
  let step = 0;
  while (n > 0 && since < patience && step < cap) {
    const i = list[draws.randint(n)];
    const token = pick(await scoreAt(ctx, seq, i, pins, { selective, lam }), cfg.temperature, draws);
    since = token !== seq[i] ? 0 : since + 1;
    seq[i] = token;
    step += 1;
  }
  return seq;
}

/** _two_phase (ph_methods.py:478-506): commit the least-disruptive selective picks, then converge the rest without selectivity. */
async function twoPhase(ctx, cfg, pins, neigh, initial, draws) {
  const seq = Int32Array.from(initial);
  for (const p of pins) seq[p.position] = p.protIdx;
  const list = Array.from(neigh, Number);
  const selPick = new Map();
  const disruption = new Map();
  for (const i of list) {
    const a = pick(await scoreAt(ctx, seq, i, pins, { selective: true, lam: null }), cfg.temperature, draws);
    const eStab = await ctx.fieldAt(seq, i);
    selPick.set(i, a);
    disruption.set(i, f32(eStab[a] - eStab[seq[i]]));
  }
  const ranked = [...list].sort((x, y) => disruption.get(x) - disruption.get(y)); // stable, like Python's sorted
  const k = Math.max(1, Math.min(list.length - 1, roundHalfEven(cfg.twoPhaseFrac * list.length)));
  const committed = ranked.slice(0, k);
  for (const i of committed) seq[i] = selPick.get(i);
  const committedSet = new Set(committed);
  const rest = list.filter((i) => !committedSet.has(i));
  return converge(ctx, cfg, pins, rest, initial, draws, { start: seq, method: "converged_mcmc", selective: false });
}

/**
 * potts_gibbs_optimize (convergence mode, ph_methods.py:512-571) for one row: sweeps over the free positions in a drawn order, each
 * resampled from softmax(-E/T) with E the conditional Potts row, written before the next is read. The inverse CDF is float32 (the
 * reference runs with x64 off). Stops after a sweep that changed nothing, or MAX_GIBBS_SWEEPS.
 */
async function gibbsOptimize(ctx, cfg, start, free, draws) {
  const temperature = Math.max(cfg.temperature, MIN_TEMPERATURE);
  const cdf = ctx.cdfOrder;
  const V = ctx.valid.length;
  const seq = Int32Array.from(start);
  for (let sweep = 0; sweep < MAX_GIBBS_SWEEPS; sweep += 1) {
    const positions = draws.permutation(free.length).map((k) => free[k]);
    const uniforms = draws.takeUniforms(positions.length);
    let changedAny = false;
    for (let k = 0; k < positions.length; k += 1) {
      const pos = positions[k];
      const row = (await ctx.rows(seq, [pos]))[0];
      const probs = softmaxProbs(maskInf(ctx.valid, row), temperature);
      const cumulative = new Float32Array(V);
      let acc = 0;
      for (let j = 0; j < V; j += 1) {
        acc = f32(acc + probs[cdf[j]]);
        cumulative[j] = acc;
      }
      const target = f32(f32(uniforms[k]) * cumulative[V - 1]);
      let index = 0;
      while (index < V && cumulative[index] < target) index += 1;
      index = Math.min(index, V - 1);
      const token = cdf[index];
      if (token !== seq[pos]) changedAny = true;
      seq[pos] = token;
      draws.events.push({ uniform: uniforms[k], choice: token, probs: Array.from(probs) });
    }
    if (!changedAny) break;
  }
  return seq;
}

