/**
 * HDC hypervector core.
 *
 * Per plan section 7:
 *   D = 65536 bits
 *   storage: packed binary (1 bit per cell, 8192 bytes per vector)
 *   conceptual: bipolar -1/+1
 *   binding: XOR (multiplicative in bipolar space: +1*+1=+1, +1*-1=-1)
 *   distance: Hamming
 *
 * Physical layout: Uint8Array of length 8192 (8 KB), MSB-first within each byte.
 *
 * Critical invariants:
 *   - Generation is fully deterministic from a string id.
 *   - Bit balance is forced to ~50/50 by rejection on gross imbalance.
 *   - sign(0) is defined deterministically as +1.
 */

import { prngFromId } from "./prng";

export const HDC_DIM = 65536; // bits
export const HDC_BYTES = HDC_DIM >>> 3; // 8192 bytes

/** A packed binary hypervector. */
export type HyperVector = Uint8Array; // length === HDC_BYTES

/**
 * Population count via lookup table — 16-bit nibble LUT is fastest in JS
 * without SIMD. Used both for Hamming distance and bit-balance checks.
 */
const POPCNT_LUT = new Uint8Array(65536);
for (let i = 0; i < 65536; i++) {
  let n = i;
  let c = 0;
  while (n) {
    n &= n - 1;
    c++;
  }
  POPCNT_LUT[i] = c;
}

/**
 * Count set bits in a packed vector. Used for balance check + tests.
 */
export function popcountVec(v: Uint8Array): number {
  let total = 0;
  const view = new Uint16Array(v.buffer, v.byteOffset, v.byteLength >>> 1);
  for (let i = 0; i < view.length; i++) {
    total += POPCNT_LUT[view[i]];
  }
  return total;
}

/**
 * Generate a deterministic, balanced hypervector from a string id.
 *
 * Algorithm:
 *   1. Seed PRNG with id (128-bit entropy surface).
 *   2. Fill D bits uniformly at random.
 *   3. Check balance: |popcount - D/2| < D * 0.005 (~327 bits).
 *      If outside, retry with a salted id (max 8 attempts).
 *
 * Returns a packed Uint8Array of length HDC_BYTES.
 */
export function generateHypervector(id: string): HyperVector {
  for (let attempt = 0; attempt < 8; attempt++) {
    const seedId = attempt === 0 ? id : `${id}::balance-retry-${attempt}`;
    const rng = prngFromId(seedId);
    const v = new Uint8Array(HDC_BYTES);

    // Fill 16 bits at a time via Uint16 view for speed.
    const view = new Uint16Array(v.buffer);
    for (let i = 0; i < view.length; i++) {
      // Combine two rng() calls for a 16-bit value: top 8 bits + low 8 bits.
      const hi = (rng() * 256) | 0;
      const lo = (rng() * 256) | 0;
      view[i] = (hi << 8) | lo;
    }

    const pop = popcountVec(v);
    const ideal = HDC_DIM >>> 1;
    const tolerance = Math.floor(HDC_DIM * 0.005); // 0.5%
    if (Math.abs(pop - ideal) <= tolerance) {
      return v;
    }
    // else retry
  }
  // Fallback: even after retries, return last attempt. Balance will be close.
  // This branch is extraordinarily rare given the tolerance.
  const rng = prngFromId(`${id}::fallback`);
  const v = new Uint8Array(HDC_BYTES);
  const view = new Uint16Array(v.buffer);
  for (let i = 0; i < view.length; i++) {
    const hi = (rng() * 256) | 0;
    const lo = (rng() * 256) | 0;
    view[i] = (hi << 8) | lo;
  }
  return v;
}

/**
 * XOR two hypervectors (binding operation).
 * Result written into `out` (which may equal a or b for in-place).
 */
export function xorInto(a: HyperVector, b: HyperVector, out: HyperVector): void {
  for (let i = 0; i < HDC_BYTES; i++) {
    out[i] = a[i] ^ b[i];
  }
}

/** Convenience: return new vector = a XOR b. */
export function xorVec(a: HyperVector, b: HyperVector): HyperVector {
  const out = new Uint8Array(HDC_BYTES);
  xorInto(a, b, out);
  return out;
}

/**
 * XOR-bind a variadic list of vectors: v1 ⊗ v2 ⊗ ... ⊗ vn.
 * Identity element is all-ones (since x ⊗ 1 = x).
 * Returns a new vector.
 */
export function bindMany(vectors: HyperVector[]): HyperVector {
  if (vectors.length === 0) {
    const ones = new Uint8Array(HDC_BYTES);
    ones.fill(0xff);
    return ones;
  }
  const out = new Uint8Array(HDC_BYTES);
  out.set(vectors[0]);
  for (let i = 1; i < vectors.length; i++) {
    xorInto(out, vectors[i], out);
  }
  return out;
}

/**
 * Hamming distance between two packed hypervectors.
 * Implemented with a 16-bit popcount LUT for speed.
 */
export function hammingDistance(a: HyperVector, b: HyperVector): number {
  let dist = 0;
  const va = new Uint16Array(a.buffer, a.byteOffset, a.byteLength >>> 1);
  const vb = new Uint16Array(b.buffer, b.byteOffset, b.byteLength >>> 1);
  for (let i = 0; i < va.length; i++) {
    dist += POPCNT_LUT[va[i] ^ vb[i]];
  }
  return dist;
}

/**
 * Convert a packed binary vector to bipolar +1/-1 representation
 * (Int8Array of length D). Used for accumulation in S-blocks.
 *
 * Convention: bit set → +1, bit clear → -1.
 */
export function toBipolar(v: HyperVector): Int8Array {
  const out = new Int8Array(HDC_DIM);
  for (let i = 0; i < HDC_DIM; i++) {
    const byteIdx = i >>> 3;
    const bitIdx = 7 - (i & 7);
    out[i] = (v[byteIdx] >> bitIdx) & 1 ? 1 : -1;
  }
  return out;
}

/**
 * Pack a bipolar vector back to binary.
 * Per plan §7: deterministic sign(0) policy → 0 maps to +1.
 */
export function fromBipolar(b: Int8Array): HyperVector {
  const out = new Uint8Array(HDC_BYTES);
  for (let i = 0; i < HDC_DIM; i++) {
    // sign(0) = +1 → bit set
    if (b[i] >= 0) {
      const byteIdx = i >>> 3;
      const bitIdx = 7 - (i & 7);
      out[byteIdx] |= 1 << bitIdx;
    }
  }
  return out;
}

/**
 * Deterministic sign function with sign(0) = +1 policy (plan §7).
 * Returns +1 for x >= 0, -1 for x < 0.
 */
export function sign(x: number): 1 | -1 {
  return x >= 0 ? 1 : -1;
}

/**
 * Apply sign() element-wise to an Int16Array accumulator (S-block)
 * and produce a packed binary K-vector.
 */
export function signPackInt16(s: Int16Array): HyperVector {
  const out = new Uint8Array(HDC_BYTES);
  for (let i = 0; i < HDC_DIM; i++) {
    if (s[i] >= 0) {
      const byteIdx = i >>> 3;
      const bitIdx = 7 - (i & 7);
      out[byteIdx] |= 1 << bitIdx;
    }
  }
  return out;
}

/**
 * Convert vector to a hex string for storage / debugging.
 * (Only used in tests / small vectors; full vector hex is 16 KB.)
 */
export function toHex(v: HyperVector): string {
  return Array.from(v)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Convert Hamming distance to similarity score in [0, 1].
 *   score = 1 - 2 * dist / D
 * Interpretation:
 *   dist = 0       → score = 1.0  (perfect match)
 *   dist = D/2     → score = 0.0  (uncorrelated, baseline for random HVs)
 *   dist = D       → score = -1.0 (perfect anti-correlation)
 *
 * We clamp to [0, 1] since negative scores are never useful for ranking.
 */
export function hammingToScore(dist: number): number {
  const raw = 1 - (2 * dist) / HDC_DIM;
  return Math.max(0, Math.min(1, raw));
}
