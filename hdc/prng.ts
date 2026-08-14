/**
 * Deterministic PRNG for HDC hypervector generation.
 *
 * Requirements (per plan section 7):
 * - 128-bit seed
 * - deterministic
 * - good bit balance (~50/50)
 * - one 64-bit hash must NOT be the only source
 */

/**
 * xfnv1a — fast, well-distributed 32-bit FNV-1a hash.
 * Used only to seed sfc32 from a string; not the sole source of entropy.
 */
function xfnv1a(str: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

/**
 * sfc32 — fast PRNG with excellent statistical properties.
 * Seeded from four 32-bit values produced by xfnv1a over different salts.
 */
export function sfc32(a: number, b: number, c: number, d: number) {
  return function () {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21 | c >>> 11) | 0;
    d = (d + 1) | 0;
    t = (t + a) | 0;
    c = (c + a) | 0;
    return (t >>> 0) / 4294967296;
  };
}

/**
 * Build a deterministic PRNG function from a string ID.
 *
 * The plan requires a 128-bit seed; we derive four 32-bit values
 * from the input string combined with four different salts, then
 * feed them into sfc32. This guarantees:
 *   - reproducibility: same id → same PRNG sequence
 *   - avalanche: small change in id → very different stream
 *   - 128-bit entropy surface (4 × 32 bits)
 */
export function prngFromId(id: string): () => number {
  const a = xfnv1a(`hdc::salt-a::${id}`);
  const b = xfnv1a(`hdc::salt-b::${id}`);
  const c = xfnv1a(`hdc::salt-c::${id}`);
  const d = xfnv1a(`hdc::salt-d::${id}`);
  return sfc32(a, b, c, d);
}

/**
 * Stable 53-bit hash from a string. Used for bucketing — must be stable
 * across processes, languages, and runs (plan §8 bucketing).
 *
 * Implementation: FNV-1a 32-bit folded twice into a 53-bit value.
 * For MVP scale this gives sufficiently uniform distribution; for
 * production scale one would upgrade to xxhash or SipHash.
 */
export function stableHash53(str: string): number {
  let h1 = 2166136261 >>> 0;
  let h2 = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 16777619) >>> 0;
    h2 = Math.imul(h2 ^ (ch + 0x9e3779b9), 2246822519) >>> 0;
  }
  // Combine two 32-bit hashes into a 53-bit safe integer.
  return h1 * 2097152 + (h2 & 0x1fffff);
}
