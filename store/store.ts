/**
 * In-memory data store with JSON persistence.
 *
 * Holds the registries defined in plan §5:
 *   - entities (with status / tombstones)
 *   - aliases
 *   - predicates
 *   - contexts (mandatory `default`)
 *   - facts (with status / tombstones)
 *   - sources
 *
 * Mutations are append-only / status-flipping where possible to preserve
 * provenance. Deleted facts stay in the table with status="deleted" so
 * rebuilds can read tombstones (plan §7).
 */

import { promises as fs } from "fs";
import path from "path";
import { homedir } from "os";
import type {
  Entity,
  Alias,
  Predicate,
  Context,
  Fact,
  Source,
  BlockMetadata,
} from "./types";
import {
  SEED_SOURCES,
  SEED_CONTEXTS,
  SEED_PREDICATES,
  SEED_ENTITIES,
  SEED_ALIASES,
  SEED_FACTS,
} from "./seed-data";

const PERSIST_DIR = path.join(homedir(), ".hdc-data");
const PERSIST_FILE = path.join(PERSIST_DIR, "store.json");

export interface StoreSnapshot {
  entities: Entity[];
  aliases: Alias[];
  predicates: Predicate[];
  contexts: Context[];
  facts: Fact[];
  sources: Source[];
  block_metadata: BlockMetadata[];
}

class DataStore {
  entities: Map<string, Entity> = new Map();
  aliases: Alias[] = [];
  predicates: Map<string, Predicate> = new Map();
  contexts: Map<string, Context> = new Map();
  facts: Map<string, Fact> = new Map();
  sources: Map<string, Source> = new Map();
  block_metadata: BlockMetadata[] = [];

  // Indexes for fast lookup
  aliasesByText: Map<string, Alias[]> = new Map(); // normalized alias → aliases
  aliasesByEntity: Map<string, Alias[]> = new Map();
  factsBySubject: Map<string, Fact[]> = new Map();
  factsByObject: Map<string, Fact[]> = new Map();
  factsByPredicate: Map<string, Fact[]> = new Map();

  private initialized = false;
  private dirty = false;

  async init(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    // Try to load persisted state first.
    try {
      const raw = await fs.readFile(PERSIST_FILE, "utf8");
      const snap: StoreSnapshot = JSON.parse(raw);
      this.loadSnapshot(snap);
      console.log(
        `[store] loaded persisted state: ${this.entities.size} entities, ${this.facts.size} facts`
      );
    } catch {
      // No persisted state — seed.
      this.seed();
      console.log(
        `[store] seeded: ${this.entities.size} entities, ${this.facts.size} facts`
      );
      await this.persist();
    }
  }

  private loadSnapshot(snap: StoreSnapshot): void {
    this.entities.clear();
    this.aliases = [];
    this.predicates.clear();
    this.contexts.clear();
    this.facts.clear();
    this.sources.clear();
    this.block_metadata = snap.block_metadata || [];

    for (const e of snap.entities) this.entities.set(e.entity_id, e);
    for (const p of snap.predicates) this.predicates.set(p.predicate_id, p);
    for (const c of snap.contexts) this.contexts.set(c.context_id, c);
    for (const s of snap.sources) this.sources.set(s.source_id, s);
    for (const f of snap.facts) this.facts.set(f.fact_id, f);
    this.aliases = snap.aliases;
    this.rebuildIndexes();
  }

  private seed(): void {
    this.entities.clear();
    this.aliases = [];
    this.predicates.clear();
    this.contexts.clear();
    this.facts.clear();
    this.sources.clear();
    this.block_metadata = [];

    for (const s of SEED_SOURCES) this.sources.set(s.source_id, s);
    for (const c of SEED_CONTEXTS) this.contexts.set(c.context_id, c);
    for (const p of SEED_PREDICATES) this.predicates.set(p.predicate_id, p);
    for (const e of SEED_ENTITIES) this.entities.set(e.entity_id, e);
    this.aliases = [...SEED_ALIASES];
    for (const f of SEED_FACTS) this.facts.set(f.fact_id, f);
    this.rebuildIndexes();
  }

  private rebuildIndexes(): void {
    this.aliasesByText.clear();
    this.aliasesByEntity.clear();
    for (const alias of this.aliases) {
      const norm = normalizeAlias(alias.alias_text);
      if (!this.aliasesByText.has(norm)) this.aliasesByText.set(norm, []);
      this.aliasesByText.get(norm)!.push(alias);
      if (!this.aliasesByEntity.has(alias.entity_id))
        this.aliasesByEntity.set(alias.entity_id, []);
      this.aliasesByEntity.get(alias.entity_id)!.push(alias);
    }
    this.factsBySubject.clear();
    this.factsByObject.clear();
    this.factsByPredicate.clear();
    for (const f of this.facts.values()) {
      if (f.status !== "active") continue;
      pushIdx(this.factsBySubject, f.subject_id, f);
      pushIdx(this.factsByObject, f.object_id, f);
      pushIdx(this.factsByPredicate, f.predicate_id, f);
    }
  }

  private markDirty(): void {
    this.dirty = true;
  }

  async persist(): Promise<void> {
    if (!this.dirty) return;
    const snap: StoreSnapshot = {
      entities: Array.from(this.entities.values()),
      aliases: this.aliases,
      predicates: Array.from(this.predicates.values()),
      contexts: Array.from(this.contexts.values()),
      facts: Array.from(this.facts.values()),
      sources: Array.from(this.sources.values()),
      block_metadata: this.block_metadata,
    };
    await fs.mkdir(PERSIST_DIR, { recursive: true });
    await fs.writeFile(PERSIST_FILE, JSON.stringify(snap, null, 2), "utf8");
    this.dirty = false;
  }

  // --- Mutation API ---

  addEntity(e: Entity): void {
    this.entities.set(e.entity_id, e);
    this.markDirty();
  }

  addAlias(a: Alias): void {
    this.aliases.push(a);
    const norm = normalizeAlias(a.alias_text);
    if (!this.aliasesByText.has(norm)) this.aliasesByText.set(norm, []);
    this.aliasesByText.get(norm)!.push(a);
    if (!this.aliasesByEntity.has(a.entity_id))
      this.aliasesByEntity.set(a.entity_id, []);
    this.aliasesByEntity.get(a.entity_id)!.push(a);
    this.markDirty();
  }

  addFact(f: Fact): void {
    this.facts.set(f.fact_id, f);
    if (f.status === "active") {
      pushIdx(this.factsBySubject, f.subject_id, f);
      pushIdx(this.factsByObject, f.object_id, f);
      pushIdx(this.factsByPredicate, f.predicate_id, f);
    }
    this.markDirty();
  }

  deleteFact(fact_id: string): void {
    const f = this.facts.get(fact_id);
    if (!f) return;
    f.status = "deleted";
    // Rebuild indexes to drop it.
    this.rebuildIndexes();
    this.markDirty();
  }

  setBlockMetadata(meta: BlockMetadata[]): void {
    this.block_metadata = meta;
    this.markDirty();
  }

  // --- Query API ---

  getEntity(id: string): Entity | undefined {
    return this.entities.get(id);
  }

  getPredicate(id: string): Predicate | undefined {
    return this.predicates.get(id);
  }

  findPredicateByCanonical(name: string): Predicate | undefined {
    const norm = name.toLowerCase().trim();
    for (const p of this.predicates.values()) {
      if (p.canonical_name.toLowerCase() === norm) return p;
    }
    return undefined;
  }

  getContext(id: string): Context | undefined {
    return this.contexts.get(id);
  }

  findContextByCanonical(name: string): Context | undefined {
    const norm = name.toLowerCase().trim();
    for (const c of this.contexts.values()) {
      if (c.canonical_name.toLowerCase() === norm) return c;
    }
    return undefined;
  }

  getSource(id: string): Source | undefined {
    return this.sources.get(id);
  }

  findSourceByName(name: string): Source | undefined {
    const norm = name.toLowerCase().trim();
    for (const s of this.sources.values()) {
      if (s.name.toLowerCase() === norm) return s;
    }
    return undefined;
  }

  findAliasesByText(text: string): Alias[] {
    return this.aliasesByText.get(normalizeAlias(text)) || [];
  }

  getAliasesByEntity(entityId: string): Alias[] {
    return this.aliasesByEntity.get(entityId) || [];
  }

  getFactsBySubject(subjectId: string): Fact[] {
    return this.factsBySubject.get(subjectId) || [];
  }

  getFactsByObject(objectId: string): Fact[] {
    return this.factsByObject.get(objectId) || [];
  }

  getFactsByPredicate(predicateId: string): Fact[] {
    return this.factsByPredicate.get(predicateId) || [];
  }

  getActiveFacts(): Fact[] {
    return Array.from(this.facts.values()).filter((f) => f.status === "active");
  }

  // Find a fact by (subject, predicate, object, context) — used for dedup / conflict.
  findFact(
    subjectId: string,
    predicateId: string,
    objectId: string,
    contextId: string
  ): Fact | undefined {
    for (const f of this.getFactsBySubject(subjectId)) {
      if (
        f.predicate_id === predicateId &&
        f.object_id === objectId &&
        f.context_id === contextId
      ) {
        return f;
      }
    }
    return undefined;
  }

  // Snapshot for serialization.
  snapshot(): StoreSnapshot {
    return {
      entities: Array.from(this.entities.values()),
      aliases: this.aliases,
      predicates: Array.from(this.predicates.values()),
      contexts: Array.from(this.contexts.values()),
      facts: Array.from(this.facts.values()),
      sources: Array.from(this.sources.values()),
      block_metadata: this.block_metadata,
    };
  }

  // For testing: force reseed.
  async resetToSeed(): Promise<void> {
    this.seed();
    await this.persist();
  }
}

function pushIdx<K>(m: Map<K, unknown[]>, k: K, v: unknown): void {
  if (!m.has(k)) m.set(k, []);
  (m.get(k) as unknown[]).push(v);
}

/** Normalize an alias for exact-match lookup: lowercase, trim, collapse spaces. */
export function normalizeAlias(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ")
    .replace(/[.,;:!?]+$/g, "")
    .trim();
}

/** Singleton. */
export const store = new DataStore();
