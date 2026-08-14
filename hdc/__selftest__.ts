/**
 * Self-test for HDC core. Run with: bun run src/lib/hdc/__selftest__.ts
 * Mirrors the unit-test list from plan §24.
 */

import {
  HDC_DIM,
  generateHypervector,
  xorVec,
  bindMany,
  hammingDistance,
  toBipolar,
  fromBipolar,
  signPackInt16,
  popcountVec,
  hammingToScore,
} from "./hypervector";
import { stableHash53, prngFromId } from "./prng";

let pass = 0;
let fail = 0;

function check(name: string, cond: boolean, extra = "") {
  if (cond) {
    pass++;
    console.log(`  ✓ ${name}`);
  } else {
    fail++;
    console.error(`  ✗ ${name} ${extra}`);
  }
}

console.log("HDC self-test");
console.log("-------------");

// 1. Determinism: same id → same vector
const a1 = generateHypervector("entity::e-001");
const a2 = generateHypervector("entity::e-001");
check("determinism: same id → identical vector", hammingDistance(a1, a2) === 0);

// 2. Different ids → ~50% distance (uncorrelated)
const b1 = generateHypervector("entity::e-001");
const b2 = generateHypervector("entity::e-002");
const d12 = hammingDistance(b1, b2);
check(
  "different ids → ~50% Hamming distance",
  Math.abs(d12 - HDC_DIM / 2) < HDC_DIM * 0.02,
  `dist=${d12}`
);

// 3. Bit balance ~50/50
const pop = popcountVec(b1);
const ideal = HDC_DIM / 2;
check(
  "bit balance within 1%",
  Math.abs(pop - ideal) < HDC_DIM * 0.01,
  `pop=${pop} ideal=${ideal}`
);

// 4. XOR reversibility: a XOR b XOR b = a
const c = xorVec(b1, b2);
const d = xorVec(c, b2);
check("XOR reversibility", hammingDistance(d, b1) === 0);

// 5. XOR self → zero vector
const selfXor = xorVec(b1, b1);
let allZero = true;
for (let i = 0; i < selfXor.length; i++) {
  if (selfXor[i] !== 0) {
    allZero = false;
    break;
  }
}
check("XOR self → zero vector", allZero);

// 6. pack/unpack round-trip preserves bits
const bipolar = toBipolar(b1);
const repacked = fromBipolar(bipolar);
check("pack/unpack round-trip", hammingDistance(b1, repacked) === 0);

// 7. sign(0) deterministic policy: → +1 (bit set)
const zeroS = new Int16Array(HDC_DIM); // all zeros
const k = signPackInt16(zeroS);
const kpop = popcountVec(k);
check("sign(0) = +1 → all bits set", kpop === HDC_DIM);

// 8. bindMany with single arg returns copy
const single = bindMany([b1]);
check("bindMany single = identity", hammingDistance(single, b1) === 0);

// 9. bindMany associativity (XOR is associative)
const v1 = generateHypervector("v1");
const v2 = generateHypervector("v2");
const v3 = generateHypervector("v3");
const left = bindMany([bindMany([v1, v2]), v3]);
const right = bindMany([v1, bindMany([v2, v3])]);
check("bindMany associativity", hammingDistance(left, right) === 0);

// 10. PRNG determinism
const r1 = prngFromId("test");
const r2 = prngFromId("test");
const seq1 = [r1(), r1(), r1()];
const seq2 = [r2(), r2(), r2()];
check(
  "prng determinism",
  seq1.every((v, i) => v === seq2[i])
);

// 11. stableHash53 stability
const h1 = stableHash53("subject::e-42");
const h2 = stableHash53("subject::e-42");
check("stableHash53 stability", h1 === h2);

// 12. stableHash53 distribution (different inputs → different outputs)
const hashes = new Set<number>();
for (let i = 0; i < 1000; i++) {
  hashes.add(stableHash53(`id-${i}`));
}
check("stableHash53 uniqueness on 1000 inputs", hashes.size === 1000);

// 13. hammingToScore monotonicity
const s0 = hammingToScore(0);
const s25 = hammingToScore(HDC_DIM / 4);
const s50 = hammingToScore(HDC_DIM / 2);
check(
  "score monotonicity: 0 > D/4 > D/2",
  s0 > s25 && s25 > s50,
  `${s0} ${s25} ${s50}`
);
check("score at D/2 ≈ 0", Math.abs(s50 - 0) < 1e-9);
check("score at 0 = 1", s0 === 1);

// 14. Vector size is 8KB
check("vector size = 8192 bytes", b1.length === 8192);

console.log("-------------");
console.log(`Pass: ${pass}, Fail: ${fail}`);
if (fail > 0) {
  process.exit(1);
}
