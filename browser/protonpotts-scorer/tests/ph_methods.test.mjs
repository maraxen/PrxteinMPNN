import assert from "node:assert/strict";
import { test } from "node:test";

import { Draws, DrawsError, pick, roundHalfEven, softmaxProbs, upstreamCdfOrder } from "../ph_design.mjs";

// ---- pure helpers: no model, no Python ----------------------------------------------------------------------------------------------

test("roundHalfEven: banker's rounding as Python round()", () => {
  assert.equal(roundHalfEven(0.5), 0);
  assert.equal(roundHalfEven(1.5), 2);
  assert.equal(roundHalfEven(2.5), 2);
  assert.equal(roundHalfEven(-0.5), 0);
  assert.equal(roundHalfEven(3.4), 3);
  assert.equal(roundHalfEven(3.6), 4);
});

test("upstreamCdfOrder: a permutation of the 30 aminx tokens; UNK maps to X, the v6 tokens are identity", () => {
  const order = upstreamCdfOrder();
  assert.deepEqual([...order].sort((a, b) => a - b), Array.from({ length: 30 }, (_, i) => i));
  assert.equal(order[0], 0); // ALA -> A
  assert.equal(order[20], 20); // UNK -> X
  assert.equal(order[21], 21); // HIS-P, upstream 21
});

test("Draws replays recorded uniforms, randints and permutations in order", () => {
  const draws = new Draws({ uniforms: [0.1, 0.2, 0.3], randints: [2, 0], perms: [[1, 0, 2]] });
  assert.deepEqual(draws.takeUniforms(2), [0.1, 0.2]);
  assert.equal(draws.randint(3), 2);
  assert.equal(draws.randint(3), 0);
  assert.deepEqual(draws.permutation(3), [1, 0, 2]);
  assert.throws(() => draws.takeUniforms(2), DrawsError); // one uniform left
});

test("Draws throws on exhaustion, on an out-of-range randint and on a permutation of the wrong length", () => {
  assert.throws(() => new Draws({ randints: [] }).randint(3), DrawsError);
  assert.throws(() => new Draws({ randints: [3] }).randint(3), DrawsError);
  assert.throws(() => new Draws({ randints: [-1] }).randint(3), DrawsError);
  assert.throws(() => new Draws({ perms: [[0, 1]] }).permutation(3), DrawsError);
  assert.throws(() => new Draws({}), DrawsError); // no seed and no record
});

test("pickProbs: inverse CDF in a non-identity cdfOrder, side='left' at an exact boundary", () => {
  // aminx-order probs; cdfOrder visits 2, 0, 3, 1 -> ordered masses 0.5, 0.125, 0.125, 0.25 -> cumulative 0.5, 0.625, 0.75, 1
  const probs = Float64Array.of(0.125, 0.25, 0.5, 0.125);
  const cdfOrder = Int32Array.of(2, 0, 3, 1);
  const draws = new Draws({ uniforms: [0.5, 0.6, 0.625, 0.99], cdfOrder });
  assert.equal(draws.pickProbs(probs), 2); // 0.5 == cumulative[0]: first index >= target
  assert.equal(draws.pickProbs(probs), 0); // 0.6 in (0.5, 0.625]
  assert.equal(draws.pickProbs(probs), 0); // 0.625 == cumulative[1]: side='left' keeps index 1 (token 0), not index 2
  assert.equal(draws.pickProbs(probs), 1); // 0.99 in (0.75, 1]
  assert.equal(draws.events.length, 4);
  // Events also carry the probabilities drawn from (X11b amendment 261010), in aminx token order.
  assert.deepEqual(draws.events[2], { uniform: 0.625, choice: 0, probs: [0.125, 0.25, 0.5, 0.125] });
});

test("pickProbs with the identity order when no cdfOrder is given", () => {
  const draws = new Draws({ uniforms: [0.0, 0.5, 0.999] });
  const quarter = Float64Array.of(0.25, 0.25, 0.25, 0.25);
  assert.equal(draws.pickProbs(quarter), 0);
  assert.equal(draws.pickProbs(quarter), 1);
  assert.equal(draws.pickProbs(quarter), 3);
});

test("softmaxProbs: +Infinity scores get probability 0 and the rest sums to 1", () => {
  const p = softmaxProbs(Float32Array.of(0, 1, Infinity), 1);
  assert.equal(p[2], 0);
  assert.ok(p[0] > p[1] && p[1] > 0);
  const sum = p.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6, `sum ${sum}`);
});

test("pick at T <= 0 returns the FIRST argmin on ties and draws nothing", () => {
  const draws = new Draws({ uniforms: [] });
  assert.equal(pick(Float32Array.of(3, 1, 1, 2), 0, draws), 1);
  assert.equal(pick(Float32Array.of(3, 1, 1, 2), -1, draws), 1);
  assert.equal(draws.events.length, 0);
});

test("pick at T > 0 with no uniforms left throws (negative control: the draw is really consumed)", () => {
  assert.throws(() => pick(Float32Array.of(0, 1), 0.05, new Draws({ uniforms: [] })), DrawsError);
});

test("pick at T <= 0 pushes the chosen token to draws.argmins and not to events (X11b amendment 3)", () => {
  const draws = new Draws({ uniforms: [] });
  assert.equal(pick(Float32Array.of(3, 1, 1, 2), 0, draws), 1);
  assert.equal(pick(Float32Array.of(0, 5), 0, draws), 0);
  assert.deepEqual(draws.argmins, [1, 0]);
  assert.equal(draws.events.length, 0); // nDraws unchanged: argmin picks are not draws
  assert.deepEqual(draws.takeUniforms(0), []);
});

test("pick at T > 0 leaves argmins untouched", () => {
  const draws = new Draws({ uniforms: [0.5] });
  pick(Float32Array.of(0, 1), 0.05, draws);
  assert.deepEqual(draws.argmins, []);
  assert.equal(draws.events.length, 1);
});
