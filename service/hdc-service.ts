/**
 * HDC Service — top-level orchestrator.
 *
 * Owns the lifecycle of:
 *   - DataStore
 *   - BlockStore
 *   - BinaryIndex
 *
 * Exposes the high-level operations:
 *   - query(text | structured)
 *   - assert(text | structured)
 *   - rebuild()
 *   - health()
 *   - stats()
 *
 * Initialised at server startup. Subsequent API route calls reuse the
 * same instance — no rebuild per request.
 */

import { store } from "../store/store";
import { blockStore } from "../blocks/block-store";
import { binaryIndex } from "../index/binary-index";
import { compile } from "../compiler/llm-compiler";
import { linkEntity } from "../linker/entity-linker";
import { textEmbeddingIndex } from "../linker/text-embedding-index";
import {
  directQuery,
  reverseQuery,
  multiHopQuery,
  type RetrievalResult,
} from "../retrieval/engine";
import { runAssert } from "../assert/assert-pipeline";
import { compose } from "../composer/response-composer";
import { telemetry, newRequestId } from "../telemetry/telemetry";
import { DEFAULT_THRESHOLDS } from "../calibration/thresholds";
import { calibrate, getActiveThresholds } from "../calibration/calibration-service";
import type {
  AssertResponse,
  Candidate,
  CompiledRequest,
  HealthResponse,
  QueryResponse,
} from "../store/types";

class HDCService {
  private initialized = false;
  private initializing: Promise<void> | null = null;
  private lastRebuild: string | null = null;
  private llmAvailable = false;
  private readOnly = false; // true during rebuild (plan §28)

  /** Enter read-only mode. Queries still work; asserts and rebuilds are rejected. */
  setReadOnly(on: boolean): void {
    this.readOnly = on;
    console.log(`[hdc-service] read-only mode: ${on ? "ON" : "OFF"}`);
  }

  isReadOnly(): boolean {
    return this.readOnly;
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    if (this.initializing) return this.initializing;

    this.initializing = this.doInit();
    await this.initializing;
  }

  private async doInit(): Promise<void> {
    console.log("[hdc-service] initializing...");
    await store.init();

    const blockResult = blockStore.buildAll();
    console.log(
      `[hdc-service] built ${blockResult.blockCount} blocks, ${blockResult.entryCount} entries in ${blockResult.durationMs}ms`
    );

    const indexResult = binaryIndex.buildAll();
    console.log(
      `[hdc-service] built index with ${indexResult.count} entities in ${indexResult.durationMs}ms`
    );

    // Build text embedding index for entity linker.
    textEmbeddingIndex.buildAll();
    console.log(
      `[hdc-service] built text embedding index with ${textEmbeddingIndex.size()} aliases`
    );

    this.lastRebuild = new Date().toISOString();

    // Run initial calibration to set thresholds from empirical data.
    try {
      const cal = await calibrate();
      console.log(
        `[hdc-service] calibrated: absolute=${cal.thresholds.absolute_threshold} min_score=${cal.thresholds.min_score.toFixed(3)} in ${cal.duration_ms}ms`
      );
    } catch (e) {
      console.warn("[hdc-service] calibration failed, using defaults:", e);
    }

    // LLM is not used — compiler is fully rule-based and offline (no external LLM).
    this.llmAvailable = false;
    console.log(`[hdc-service] LLM available: false (compiler is fully rule-based, no external LLM)`);

    this.initialized = true;
    this.initializing = null;
    console.log("[hdc-service] ready.");
  }

  /**
   * Full query pipeline (plan §32):
   *   user text → rule-based compiler → entity linker → HDC query → binary index →
   *   threshold filtering → beam pruning → provenance → response composer
   */
  async query(text: string): Promise<QueryResponse & { composed: ReturnType<typeof compose> }> {
    await this.init();
    const requestId = newRequestId();
    const t0 = Date.now();

    telemetry.log(requestId, "info", "query.start", { text });

    // 1. Compile.
    const tCompileStart = Date.now();
    const { request: parsed, source, latencyMs: parseLatency } = await compile(text);
    const parseDuration = Date.now() - tCompileStart;
    telemetry.recordMetric("compile_latency_ms", parseDuration);
    telemetry.log(requestId, "info", "query.parsed", { source, parsed });

    if (parsed.type === "clarify") {
      const resp: QueryResponse = {
        status: "clarify",
        candidates: [],
        chain_used: [],
        context_used: "c:default",
        scores: [],
        distances: [],
        provenance: { chain: [], sources: [] },
        warnings: [parsed.message],
        parsed_query: parsed,
        resolved_subject: {
          entity_id: null,
          canonical_name: null,
          confidence: 0,
          candidates: [],
        },
        latency_ms: {
          total: Date.now() - t0,
          parse: parseDuration,
          link: 0,
          retrieval: 0,
        },
      };
      telemetry.recordMetric("total_queries", 1);
      const composed = compose(resp);
      return { ...resp, composed };
    }

    if (parsed.type === "assert") {
      // Caller should hit /api/assert instead, but be defensive.
      const resp: QueryResponse = {
        status: "clarify",
        candidates: [],
        chain_used: [],
        context_used: "c:default",
        scores: [],
        distances: [],
        provenance: { chain: [], sources: [] },
        warnings: ["Compiled as assert — use /api/assert endpoint instead."],
        parsed_query: parsed,
        resolved_subject: {
          entity_id: null,
          canonical_name: null,
          confidence: 0,
          candidates: [],
        },
        latency_ms: {
          total: Date.now() - t0,
          parse: parseDuration,
          link: 0,
          retrieval: 0,
        },
      };
      const composed = compose(resp);
      return { ...resp, composed };
    }

    // 2. Entity link subject.
    const tLinkStart = Date.now();
    const linkResult = linkEntity(parsed.subject_text);
    const linkDuration = Date.now() - tLinkStart;
    telemetry.recordMetric("entity_link_latency_ms", linkDuration);
    telemetry.log(requestId, "info", "query.linked", {
      subject_text: parsed.subject_text,
      entity_id: linkResult.entity_id,
      confidence: linkResult.confidence,
    });

    if (!linkResult.entity_id) {
      const resp: QueryResponse = {
        status: "clarify",
        candidates: [],
        chain_used: parsed.chain,
        context_used: "c:default",
        scores: [],
        distances: [],
        provenance: { chain: [], sources: [] },
        warnings: [
          `Could not resolve entity "${parsed.subject_text}" with sufficient confidence.`,
          ...(linkResult.candidates.length > 0
            ? [`Candidates: ${linkResult.candidates.map((c) => c.canonical_name).join(", ")}`]
            : []),
        ],
        parsed_query: parsed,
        resolved_subject: {
          entity_id: null,
          canonical_name: linkResult.canonical_name,
          confidence: linkResult.confidence,
          candidates: linkResult.candidates,
        },
        latency_ms: {
          total: Date.now() - t0,
          parse: parseDuration,
          link: linkDuration,
          retrieval: 0,
        },
      };
      telemetry.recordMetric("total_queries", 1);
      telemetry.recordMetric("threshold_rejects", 1);
      const composed = compose(resp);
      return { ...resp, composed };
    }

    // 3. Resolve predicates to predicate_ids.
    const predicateIds: string[] = [];
    for (const pName of parsed.chain) {
      const pred = store.findPredicateByCanonical(pName);
      if (pred) predicateIds.push(pred.predicate_id);
      else telemetry.log(requestId, "warn", "query.unknown_predicate", { name: pName });
    }
    if (predicateIds.length === 0) {
      const resp: QueryResponse = {
        status: "clarify",
        candidates: [],
        chain_used: parsed.chain,
        context_used: "c:default",
        scores: [],
        distances: [],
        provenance: { chain: [], sources: [] },
        warnings: ["No valid predicates recognized."],
        parsed_query: parsed,
        resolved_subject: {
          entity_id: linkResult.entity_id,
          canonical_name: linkResult.canonical_name,
          confidence: linkResult.confidence,
          candidates: linkResult.candidates,
        },
        latency_ms: {
          total: Date.now() - t0,
          parse: parseDuration,
          link: linkDuration,
          retrieval: 0,
        },
      };
      const composed = compose(resp);
      return { ...resp, composed };
    }

    // 4. Resolve context.
    let contextId = "c:default";
    if (parsed.context_text) {
      const ctx = store.findContextByCanonical(parsed.context_text);
      if (ctx) contextId = ctx.context_id;
    }

    // 5. Retrieval.
    const tRetStart = Date.now();
    let result;
    let chainUsed: string[] = [];
    if (predicateIds.length === 1) {
      // Single-hop: try direct first; if no candidates, try reverse.
      result = directQuery({
        subject_id: linkResult.entity_id!,
        predicate_id: predicateIds[0],
        context_id: contextId,
        top_k: parsed.top_k,
      });
      chainUsed = parsed.chain;
      if (result.candidates.length === 0) {
        // Try reverse (e.g. "who works at Acme" — Acme is the object, not subject).
        const rev = reverseQuery({
          object_id: linkResult.entity_id!,
          predicate_id: predicateIds[0],
          context_id: contextId,
          top_k: parsed.top_k,
        });
        if (rev.candidates.length > 0) {
          result = rev;
        }
      }
    } else {
      // Multi-hop.
      result = multiHopQuery({
        subject_id: linkResult.entity_id!,
        chain: predicateIds,
        context_id: contextId,
        beam_width: parsed.beam_width,
        top_k: parsed.top_k,
      });
      chainUsed = parsed.chain;
    }
    const retDuration = Date.now() - tRetStart;
    telemetry.recordMetric("retrieval_latency_ms", retDuration);
    telemetry.recordMetric("candidate_count", result.candidates.length);

    // 6. Build response.
    const status: QueryResponse["status"] =
      result.candidates.length === 0
        ? "empty"
        : result.warnings.includes("ambiguous: small margin between top candidates")
        ? "ambiguous"
        : "ok";

    const totalDuration = Date.now() - t0;
    telemetry.recordMetric("query_latency_ms", totalDuration);
    telemetry.recordMetric("total_queries", 1);
    if (result.candidates.length === 0) {
      telemetry.recordMetric("threshold_rejects", 1);
    }

    const resp: QueryResponse = {
      status,
      candidates: result.candidates,
      chain_used: chainUsed,
      context_used: result.context_used,
      context_fallbacks_tried: result.context_fallbacks_tried || [],
      scores: result.candidates.map((c) => c.score),
      distances: result.candidates.map((c) => c.distance),
      provenance: {
        chain: chainUsed,
        sources: result.candidates
          .map((c) => c.provenance.source_id)
          .filter((s): s is string => !!s),
      },
    warnings: result.warnings,
    parsed_query: parsed,
      resolved_subject: {
        entity_id: linkResult.entity_id,
        canonical_name: linkResult.canonical_name,
        confidence: linkResult.confidence,
        candidates: linkResult.candidates,
      },
      latency_ms: {
        total: totalDuration,
        parse: parseDuration,
        link: linkDuration,
        retrieval: retDuration,
      },
    };

    telemetry.log(requestId, "info", "query.done", {
      status,
      candidates: resp.candidates.length,
      latency_total: totalDuration,
    });

    const composed = compose(resp);
    return { ...resp, composed };
  }

  /**
   * Raw structured query — bypass rule-based compiler. Used by the testing UI.
   */
  async rawQuery(
    subject_id: string,
    predicate_id: string,
    direction: "direct" | "reverse",
    context_id?: string,
    top_k?: number
  ): Promise<RetrievalResult> {
    await this.init();
    const params = {
      subject_id,
      object_id: subject_id,
      predicate_id,
      context_id: context_id || "c:default",
      top_k: top_k || 10,
    };
    if (direction === "direct") return directQuery(params);
    return reverseQuery(params);
  }

  /**
   * Raw multi-hop — bypass compiler. Used by the testing UI.
   */
  async rawMultiHop(
    subject_id: string,
    chain: string[],
    context_id?: string,
    beam_width?: number,
    top_k?: number
  ) {
    await this.init();
    return multiHopQuery({
      subject_id,
      chain,
      context_id: context_id || "c:default",
      beam_width: beam_width || 5,
      top_k: top_k || 10,
    });
  }

  /**
   * Assert pipeline (plan §32). Rejects when in read-only mode (plan §28).
   */
  async assert(text: string): Promise<AssertResponse & { parsed_query: CompiledRequest }> {
    await this.init();
    if (this.readOnly) {
      return {
        accepted: false,
        entity_resolution_result: {
          subject: { entity_id: null, canonical_name: null, confidence: 0, created_new: false },
          object: { entity_id: null, canonical_name: null, confidence: 0, created_new: false },
        },
        conflicts: [],
        fact_id: null,
        warnings: ["System is in read-only mode (rebuild in progress)."],
        predicate_id: null,
        context_id: null,
        parsed_query: { type: "clarify", message: "Read-only mode" },
      };
    }
    const requestId = newRequestId();
    telemetry.log(requestId, "info", "assert.start", { text });

    const { request } = await compile(text);
    if (request.type !== "assert") {
      telemetry.log(requestId, "warn", "assert.not_assert", { type: request.type });
      return {
        accepted: false,
        entity_resolution_result: {
          subject: { entity_id: null, canonical_name: null, confidence: 0, created_new: false },
          object: { entity_id: null, canonical_name: null, confidence: 0, created_new: false },
        },
        conflicts: [],
        fact_id: null,
        warnings: ["Compiled request is not an assert."],
        predicate_id: null,
        context_id: null,
        parsed_query: request,
      };
    }
    telemetry.recordMetric("total_asserts", 1);
    const result = await runAssert({
      subject_text: request.subject_text,
      predicate: request.predicate,
      object_text: request.object_text,
      context_text: request.context_text,
      source_text: request.source_text,
      confidence: request.confidence,
      create_missing_entities: true,
    });
    telemetry.log(requestId, "info", "assert.done", { accepted: result.accepted, fact_id: result.fact_id });
    return { ...result, parsed_query: request };
  }

  /**
   * Rebuild all blocks + index from current store state.
   * Plan §28: "Есть read-only mode во время rebuild".
   */
  async rebuild(): Promise<{
    blocks: { count: number; durationMs: number };
    index: { count: number; durationMs: number };
  }> {
    await this.init();
    const wasReadOnly = this.readOnly;
    this.setReadOnly(true);
    try {
      const blockResult = blockStore.buildAll();
      const indexResult = binaryIndex.buildAll();
      textEmbeddingIndex.buildAll();
      this.lastRebuild = new Date().toISOString();
      return {
        blocks: { count: blockResult.blockCount, durationMs: blockResult.durationMs },
        index: { count: indexResult.count, durationMs: indexResult.durationMs },
      };
    } finally {
      this.setReadOnly(wasReadOnly);
    }
  }

  /**
   * Run threshold calibration on the current dataset.
   */
  async calibrate() {
    await this.init();
    return calibrate();
  }

  /**
   * Health check (plan §26).
   */
  async health(): Promise<HealthResponse> {
    await this.init();
    const blockMB = (blockStore.listBlocks().length * 65536 * 2) / (1024 * 1024); // Int16 S per block
    const indexMB = binaryIndex.memoryMB();
    const heapMB = process.memoryUsage().heapUsed / (1024 * 1024);
    const thresholds = getActiveThresholds();
    return {
      status: "ok",
      memory_usage: {
        heap_mb: Math.round(heapMB * 100) / 100,
        blocks_mb: Math.round(blockMB * 100) / 100,
        index_mb: Math.round(indexMB * 100) / 100,
      },
      index_version: `v${binaryIndex.getVersion()}`,
      last_rebuild: this.lastRebuild || "never",
      llm_available: this.llmAvailable,
      thresholds: {
        absolute_threshold: thresholds.absolute_threshold,
        margin_threshold: thresholds.margin_threshold,
        min_score: Math.round(thresholds.min_score * 1000) / 1000,
      },
      stats: {
        entities: store.entities.size,
        aliases: store.aliases.length,
        predicates: store.predicates.size,
        contexts: store.contexts.size,
        facts: store.facts.size,
        sources: store.sources.size,
        blocks: blockStore.listBlocks().length,
      },
    };
  }

  /**
   * Metrics summary (plan §27).
   */
  metrics() {
    return {
      summary: telemetry.getMetricSummary(),
      recent_logs: telemetry.getRecentLogs(50),
    };
  }
}

/** Singleton. */
export const hdcService = new HDCService();

/**
 * Eager init on first import in a server context.
 * Safe to call multiple times.
 */
let initStarted = false;
export function ensureInitialized(): Promise<void> {
  if (!initStarted) {
    initStarted = true;
    return hdcService.init();
  }
  return Promise.resolve();
}
