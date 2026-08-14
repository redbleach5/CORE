/**
 * Block store — the heart of the HDC memory.
 *
 * Plan §8:
 *   For a fact (subject, predicate, object, context):
 *     fact_vector = φ(subject) ⊗ φ(object) ⊗ φ(context)
 *
 *   For each (predicate, direction, bucket) block:
 *     S[block] = Σ bipolar(fact_vector) over facts in bucket
 *     K[block] = sign(S[block])
 *
 * Plan §7 / §13:
 *   - S stored as Int16Array (MVP scale: no overflow risk for <32k facts/block).
 *   - Overflow monitored.
 *   - Tombstones: deleted facts are NOT removed from S; instead they are
 *     subtracted. K is rebuilt lazily from S after enough churn.
 *
 * Plan §4 / §10 / §11:
 *   Direct query:    a = K_direct ⊗ φ(subject) ⊗ φ(context)
 *                    → search nearest entities to a in FAISS.
 *   Reverse query:   a = K_reverse ⊗ φ(object) ⊗ φ(context)
 *                    → search nearest entities to a in FAISS.
 *
 * The reason direct K encodes (subject⊗object⊗context):
 *   XOR of K_direct with (subject ⊗ context) recovers (object ⊗ noise),
 *   which is close to φ(object) — so nearest-neighbor over entity vectors
 *   returns the object.
 */

import {
  HDC_DIM,
  HyperVector,
  generateHypervector,
  xorVec,
  bindMany,
  signPackInt16,
  toBipolar,
} from "../hdc/hypervector";
import { store } from "../store/store";
import type { BlockMetadata, Fact, SStorageType } from "../store/types";
import {
  MAX_ENTRIES_PER_BLOCK,
  pickBucketCount,
  directBucket,
  reverseBucket,
  blockId,
} from "./bucketing";

// Cache of generated φ vectors. Generation is deterministic, so caching
// only saves CPU; it does not affect correctness. Cleared on rebuild.
const vectorCache = new Map<string, HyperVector>();

export function phi(id: string): HyperVector {
  let v = vectorCache.get(id);
  if (!v) {
    v = generateHypervector(id);
    vectorCache.set(id, v);
  }
  return v;
}

export function clearVectorCache(): void {
  vectorCache.clear();
}

interface Block {
  predicate_id: string;
  direction: "direct" | "reverse";
  bucket: number;
  s: Int16Array; // length = HDC_DIM
  k: HyperVector | null; // sign(s) — rebuilt on demand
  entry_count: number;
  version: number;
  overflow: boolean;
}

/**
 * In-memory block store. Indexed by blockId.
 *
 * For MVP we keep all blocks "hot" (Int16 S in memory). For production
 * scale, the plan calls for moving cold blocks to K-only + delta-log;
 * the interface here is designed so that swap-out is a local change.
 */
class BlockStore {
  private blocks = new Map<string, Block>();
  private bucketCounts = new Map<string, { direct: number; reverse: number }>();
  private builtAt: string | null = null;
  private version = 0;

  /** Build all blocks from current active facts. Destructive: clears existing. */
  buildAll(): { blockCount: number; entryCount: number; durationMs: number } {
    const t0 = Date.now();
    this.blocks.clear();
    this.bucketCounts.clear();
    vectorCache.clear();

    const facts = store.getActiveFacts();

    // First pass: count entries per (predicate, direction) to pick bucket counts.
    const entriesPerPredicate = new Map<
      string,
      { direct: number; reverse: number }
    >();
    for (const f of facts) {
      const cur =
        entriesPerPredicate.get(f.predicate_id) || { direct: 0, reverse: 0 };
      cur.direct += 1;
      cur.reverse += 1;
      entriesPerPredicate.set(f.predicate_id, cur);
    }

    // Pick bucket counts.
    for (const [pid, counts] of entriesPerPredicate) {
      this.bucketCounts.set(pid, {
        direct: pickBucketCount(counts.direct),
        reverse: pickBucketCount(counts.reverse),
      });
    }

    // Second pass: accumulate S.
    let entryCount = 0;
    for (const f of facts) {
      this.addFactToBlocks(f, false);
      entryCount++;
    }

    // Build K vectors for all blocks.
    for (const b of this.blocks.values()) {
      b.k = signPackInt16(b.s);
    }

    this.builtAt = new Date().toISOString();
    this.version++;

    // Persist metadata into store.
    store.setBlockMetadata(this.snapshotMetadata());

    const durationMs = Date.now() - t0;
    return {
      blockCount: this.blocks.size,
      entryCount,
      durationMs,
    };
  }

  /** Incrementally add a fact to all relevant blocks. */
  addFactToBlocks(fact: Fact, rebuildK = true): void {
    // Direct block.
    const directB = this.getBucketCount(fact.predicate_id, "direct");
    const dBucket = directBucket(fact.subject_id, directB);
    const dId = blockId(fact.predicate_id, "direct", dBucket);
    const dBlock = this.getOrCreateBlock(fact.predicate_id, "direct", dBucket);
    const dFactVec = bindMany([
      phi(fact.subject_id),
      phi(fact.object_id),
      phi(fact.context_id),
    ]);
    accumulateBipolar(dBlock, dFactVec);
    if (rebuildK) dBlock.k = signPackInt16(dBlock.s);

    // Reverse block.
    const revB = this.getBucketCount(fact.predicate_id, "reverse");
    const rBucket = reverseBucket(fact.object_id, revB);
    const rId = blockId(fact.predicate_id, "reverse", rBucket);
    const rBlock = this.getOrCreateBlock(fact.predicate_id, "reverse", rBucket);
    // Same fact vector — XOR is commutative & associative, so direct and
    // reverse blocks share the same fact-vector. The difference is which
    // entity we XOR with at query time (subject vs object).
    accumulateBipolar(rBlock, dFactVec);
    if (rebuildK) rBlock.k = signPackInt16(rBlock.s);

    // Track entry count.
    dBlock.entry_count++;
    rBlock.entry_count++;

    // Overflow monitoring.
    if (dBlock.entry_count > MAX_ENTRIES_PER_BLOCK * 1.5) dBlock.overflow = true;
    if (rBlock.entry_count > MAX_ENTRIES_PER_BLOCK * 1.5) rBlock.overflow = true;

    // Track version.
    dBlock.version++;
    rBlock.version++;
    this.version++;

    // Update metadata in store.
    store.setBlockMetadata(this.snapshotMetadata());
  }

  /** Incrementally remove a fact from blocks (subtraction). */
  removeFactFromBlocks(fact: Fact, rebuildK = true): void {
    const directB = this.getBucketCount(fact.predicate_id, "direct");
    const dBucket = directBucket(fact.subject_id, directB);
    const dBlock = this.getBlock(
      fact.predicate_id,
      "direct",
      dBucket
    );
    const factVec = bindMany([
      phi(fact.subject_id),
      phi(fact.object_id),
      phi(fact.context_id),
    ]);
    if (dBlock) {
      subtractBipolar(dBlock, factVec);
      dBlock.entry_count = Math.max(0, dBlock.entry_count - 1);
      dBlock.version++;
      if (rebuildK) dBlock.k = signPackInt16(dBlock.s);
    }

    const revB = this.getBucketCount(fact.predicate_id, "reverse");
    const rBucket = reverseBucket(fact.object_id, revB);
    const rBlock = this.getBlock(fact.predicate_id, "reverse", rBucket);
    if (rBlock) {
      subtractBipolar(rBlock, factVec);
      rBlock.entry_count = Math.max(0, rBlock.entry_count - 1);
      rBlock.version++;
      if (rebuildK) rBlock.k = signPackInt16(rBlock.s);
    }

    this.version++;
    store.setBlockMetadata(this.snapshotMetadata());
  }

  /**
   * Produce the query vector for a direct lookup.
   *   q = φ(subject) ⊗ φ(context)
   *   a = K_direct ⊗ q
   *
   * Returns null if block doesn't exist (no facts for this predicate/subject).
   */
  buildDirectQueryVector(
    subjectId: string,
    predicateId: string,
    contextId: string
  ): { vector: HyperVector; blockId: string } | null {
    const B = this.getBucketCount(predicateId, "direct");
    const bucket = directBucket(subjectId, B);
    const block = this.getBlock(predicateId, "direct", bucket);
    if (!block || !block.k) return null;
    const q = bindMany([phi(subjectId), phi(contextId)]);
    const a = xorVec(block.k, q);
    return { vector: a, blockId: blockId(predicateId, "direct", bucket) };
  }

  /**
   * Produce the query vector for a reverse lookup.
   *   q = φ(object) ⊗ φ(context)
   *   a = K_reverse ⊗ q
   */
  buildReverseQueryVector(
    objectId: string,
    predicateId: string,
    contextId: string
  ): { vector: HyperVector; blockId: string } | null {
    const B = this.getBucketCount(predicateId, "reverse");
    const bucket = reverseBucket(objectId, B);
    const block = this.getBlock(predicateId, "reverse", bucket);
    if (!block || !block.k) return null;
    const q = bindMany([phi(objectId), phi(contextId)]);
    const a = xorVec(block.k, q);
    return { vector: a, blockId: blockId(predicateId, "reverse", bucket) };
  }

  /**
   * Inspect a single block (for the UI). Returns the K-vector and S-summary.
   */
  inspectBlock(
    predicateId: string,
    direction: "direct" | "reverse",
    bucket: number
  ): {
    block_id: string;
    entry_count: number;
    version: number;
    overflow: boolean;
    s_summary: { min: number; max: number; mean: number; nonzero: number };
    k_popcount: number;
  } | null {
    const block = this.getBlock(predicateId, direction, bucket);
    if (!block) return null;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nonzero = 0;
    for (let i = 0; i < block.s.length; i++) {
      const v = block.s[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (v !== 0) nonzero++;
    }
    let kpop = 0;
    if (block.k) {
      const view = new Uint16Array(block.k.buffer);
      // Reuse popcount LUT
      const LUT = getPopcntLUT();
      for (let i = 0; i < view.length; i++) kpop += LUT[view[i]];
    }
    return {
      block_id: blockId(predicateId, direction, bucket),
      entry_count: block.entry_count,
      version: block.version,
      overflow: block.overflow,
      s_summary: {
        min,
        max,
        mean: sum / block.s.length,
        nonzero,
      },
      k_popcount: kpop,
    };
  }

  listBlocks(): BlockMetadata[] {
    return this.snapshotMetadata();
  }

  getBuiltAt(): string | null {
    return this.builtAt;
  }

  getVersion(): number {
    return this.version;
  }

  // --- Internals ---

  private getBucketCount(
    predicateId: string,
    direction: "direct" | "reverse"
  ): number {
    const c = this.bucketCounts.get(predicateId);
    if (!c) return 1;
    return direction === "direct" ? c.direct : c.reverse;
  }

  private getOrCreateBlock(
    predicateId: string,
    direction: "direct" | "reverse",
    bucket: number
  ): Block {
    const id = blockId(predicateId, direction, bucket);
    let b = this.blocks.get(id);
    if (!b) {
      b = {
        predicate_id: predicateId,
        direction,
        bucket,
        s: new Int16Array(HDC_DIM),
        k: null,
        entry_count: 0,
        version: 0,
        overflow: false,
      };
      this.blocks.set(id, b);
    }
    return b;
  }

  private getBlock(
    predicateId: string,
    direction: "direct" | "reverse",
    bucket: number
  ): Block | undefined {
    return this.blocks.get(blockId(predicateId, direction, bucket));
  }

  private snapshotMetadata(): BlockMetadata[] {
    const out: BlockMetadata[] = [];
    for (const b of this.blocks.values()) {
      let storage: SStorageType = "int16";
      if (b.overflow) storage = "int32";
      out.push({
        predicate_id: b.predicate_id,
        direction: b.direction,
        bucket_id: b.bucket,
        entry_count: b.entry_count,
        s_storage_type: storage,
        built_at: this.builtAt || new Date().toISOString(),
        version: b.version,
      });
    }
    return out;
  }
}

// Popcount LUT reused for block inspection. Lazy-init.
let _popcntLUT: Uint8Array | null = null;
function getPopcntLUT(): Uint8Array {
  if (_popcntLUT) return _popcntLUT;
  _popcntLUT = new Uint8Array(65536);
  for (let i = 0; i < 65536; i++) {
    let n = i;
    let c = 0;
    while (n) {
      n &= n - 1;
      c++;
    }
    _popcntLUT[i] = c;
  }
  return _popcntLUT;
}

/** Add a bipolar vector's contribution to block S. */
function accumulateBipolar(block: Block, v: HyperVector): void {
  const bipolar = toBipolar(v); // Int8Array
  const s = block.s;
  for (let i = 0; i < HDC_DIM; i++) {
    s[i] += bipolar[i];
    // Int16 overflow guard.
    if (s[i] > 32767) s[i] = 32767;
    else if (s[i] < -32768) s[i] = -32768;
  }
}

function subtractBipolar(block: Block, v: HyperVector): void {
  const bipolar = toBipolar(v);
  const s = block.s;
  for (let i = 0; i < HDC_DIM; i++) {
    s[i] -= bipolar[i];
    if (s[i] > 32767) s[i] = 32767;
    else if (s[i] < -32768) s[i] = -32768;
  }
}

/** Singleton. */
export const blockStore = new BlockStore();
