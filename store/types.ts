/**
 * Core data model types per plan §5.
 *
 * All identifiers are strings (stable, human-readable, prefixed by type).
 * This avoids hash-coupling and makes debugging trivial.
 *
 *   entity_id    := "e:<slug>"
 *   predicate_id := "p:<slug>"
 *   context_id   := "c:<slug>"
 *   source_id    := "s:<slug>"
 *   fact_id      := "f:<uuid-ish>"
 */

export type EntityStatus = "active" | "merged" | "deleted";
export type PredicateStatus = "active" | "deprecated";
export type ContextStatus = "active" | "deprecated";
export type FactStatus = "active" | "deleted" | "conflict" | "superseded";
export type BlockDirection = "direct" | "reverse";
export type SStorageType = "none" | "int16" | "int32" | "delta";

export interface Entity {
  entity_id: string;
  canonical_name: string;
  type: string; // person, organization, location, etc.
  description: string;
  created_at: string;
  updated_at: string;
  status: EntityStatus;
}

export interface Alias {
  alias_id: string;
  entity_id: string;
  alias_text: string;
  language: string;
  source: string;
  confidence: number; // 0..1
}

export interface Predicate {
  predicate_id: string;
  canonical_name: string;
  reverse_predicate_id: string | null;
  domain_type: string;
  range_type: string;
  is_symmetric: boolean;
  is_transitive: boolean;
  status: PredicateStatus;
  description: string;
}

export interface Context {
  context_id: string;
  canonical_name: string;
  parent_context_id: string | null;
  description: string;
  status: ContextStatus;
}

export interface Source {
  source_id: string;
  name: string;
  reliability: number; // 0..1
  url: string | null;
  license: string | null;
  created_at: string;
}

export interface Fact {
  fact_id: string;
  subject_id: string;
  predicate_id: string;
  object_id: string;
  context_id: string;
  source_id: string;
  confidence: number; // 0..1
  timestamp: string;
  status: FactStatus;
  provenance: {
    chain: string[]; // human-readable path, e.g. ["subject","predicate","object"]
    source_id: string;
    asserted_at: string;
    asserted_by: "seed" | "user" | "system";
  };
}

export interface BlockMetadata {
  predicate_id: string;
  direction: BlockDirection;
  bucket_id: number;
  entry_count: number;
  s_storage_type: SStorageType;
  built_at: string;
  version: number;
}

/**
 * Structured query / assert objects emitted by the LLM compiler (plan §17).
 */
export type CompiledRequest =
  | {
      type: "query";
      subject_text: string;
      chain: string[]; // predicate canonical names
      context_text: string | null;
      beam_width: number;
      top_k: number;
    }
  | {
      type: "assert";
      subject_text: string;
      predicate: string;
      object_text: string;
      context_text: string | null;
      source_text: string;
      confidence: number;
    }
  | {
      type: "clarify";
      message: string;
      candidates?: string[];
    };

/**
 * Retrieval candidate as returned by the retrieval engine.
 */
export interface Candidate {
  entity_id: string;
  canonical_name: string;
  distance: number;
  score: number;
  predicate_id: string;
  context_id: string;
  block_id: string;
  provenance: {
    chain: string[];
    source_id: string | null;
    fact_id: string | null;
  };
}

/**
 * Full query response per plan §26.
 */
export interface QueryResponse {
  status: "ok" | "empty" | "ambiguous" | "clarify" | "error";
  candidates: Candidate[];
  chain_used: string[];
  context_used: string;
  context_fallbacks_tried?: string[];
  scores: number[];
  distances: number[];
  provenance: {
    chain: string[];
    sources: string[];
  };
  warnings: string[];
  parsed_query: CompiledRequest;
  resolved_subject: {
    entity_id: string | null;
    canonical_name: string | null;
    confidence: number;
    candidates: { entity_id: string; canonical_name: string; confidence: number }[];
  };
  latency_ms: {
    total: number;
    parse: number;
    link: number;
    retrieval: number;
  };
}

/**
 * Assert response per plan §26.
 */
export interface AssertResponse {
  accepted: boolean;
  entity_resolution_result: {
    subject: {
      entity_id: string | null;
      canonical_name: string | null;
      confidence: number;
      created_new: boolean;
    };
    object: {
      entity_id: string | null;
      canonical_name: string | null;
      confidence: number;
      created_new: boolean;
    };
  };
  conflicts: {
    existing_fact_id: string;
    existing_source: string;
    existing_context: string;
  }[];
  fact_id: string | null;
  warnings: string[];
  predicate_id: string | null;
  context_id: string | null;
}

/**
 * Health response per plan §26.
 */
export interface HealthResponse {
  status: "ok" | "degraded" | "down";
  memory_usage: {
    heap_mb: number;
    blocks_mb: number;
    index_mb: number;
  };
  index_version: string;
  last_rebuild: string;
  llm_available: boolean;
  thresholds?: {
    absolute_threshold: number;
    margin_threshold: number;
    min_score: number;
  };
  stats: {
    entities: number;
    aliases: number;
    predicates: number;
    contexts: number;
    facts: number;
    sources: number;
    blocks: number;
  };
}
