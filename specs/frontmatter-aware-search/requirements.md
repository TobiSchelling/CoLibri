# Frontmatter-Aware Search Requirements

> **Feature:** frontmatter-aware-search
> **Date:** 2026-05-04
> **Repo:** CoLibri
> **Repo Type:** code
> **Spec Directory:** specs/frontmatter-aware-search/
> **Design Doc Location:** [specs/frontmatter-aware-search/design.md](design.md)
> **Status:** Draft

---

## Context

CoLibri's filesystem connector indexes Markdown content but ignores YAML frontmatter — `tags: None` is hardcoded in the envelope. Downstream effects:

- `browse_topics` MCP tool returns nothing useful for filesystem-sourced documents (it reads `metadata_store.tags_json`, which the connector populates with `[]`).
- Search has no way to filter by structured fields like `area: SIT`, `status: active`, `DocumentType: meeting`, `people-involved: [[Maximilian Hamacher]]`.
- "Find me all decisions about X involving Y in Q1" — the kind of query a knowledge-base agent should answer trivially — falls back to text-only matches.

Additionally:

- Search results are returned as **chunks**, not documents. A 4-chunk match in one file produces 4 results, crowding out other relevant files. There's no `group_by: document` option.
- Path-based filtering (e.g. "search only `03_MY_PROJECTS/02_HEIMDALL/...`") is only available at index time via per-source connectors — not at query time.

This feature adds frontmatter parsing in the connector, exposes the parsed fields as search filters, adds doc-level grouping, and exposes the filters through the MCP serving plane.

## Requirements

### REQ-001: Frontmatter Parsing in Filesystem Connector [MUST]
**Pattern:** Event-Driven

When the filesystem connector reads a Markdown file with a leading YAML frontmatter block (`---\n…\n---\n` at the start), the connector shall parse the block and populate `EnvelopeMetadata.tags` from the `tags:` key (if present) and additional structured fields in a new envelope-carried `frontmatter` map.

**Acceptance Criteria:**
- AC-001.1: A file starting with `---\ntags: [foo, bar]\n---\n# Body` produces an envelope with `metadata.tags = ["foo", "bar"]`.
- AC-001.2: A file with no frontmatter (no leading `---`) produces an envelope with `metadata.tags = None` (existing behavior preserved).
- AC-001.3: Malformed frontmatter (unclosed `---`, invalid YAML) is logged at warn-level and the file is indexed with `tags = None` rather than the connector failing.
- AC-001.4: Standard Obsidian-vault frontmatter fields are extracted into a structured map: `area`, `DocumentType`, `status`, `authority`, `people-involved`, `created`, `date`. Unknown fields are tolerated but not propagated.
- AC-001.5: Tag values from `tags:` are normalized: trimmed, deduplicated, empty values dropped, leading `#` stripped.

### REQ-002: Frontmatter Storage in Metadata Store [MUST]
**Pattern:** Ubiquitous

The metadata store shall persist a `frontmatter_json` column on each document row, storing the parsed structured fields as a JSON object for query-time filtering.

**Acceptance Criteria:**
- AC-002.1: A new SQLite migration adds a `frontmatter_json TEXT NOT NULL DEFAULT '{}'` column to the documents table.
- AC-002.2: The migration is idempotent: running it on an already-migrated DB is a no-op.
- AC-002.3: `MetadataStore::upsert_document` writes the frontmatter JSON; `list_documents` returns it as part of the row.
- AC-002.4: Existing rows from earlier schema versions get `'{}'` as their frontmatter_json (default value applies).

### REQ-003: Search Filters [MUST]
**Pattern:** Ubiquitous

The `SearchEngine::search` API shall accept additional optional filters that narrow the result set after vector/BM25 retrieval and before rank merging:

- `path_includes: Option<Vec<String>>` — return only chunks whose document path contains *any* of the listed substrings.
- `path_excludes: Option<Vec<String>>` — drop chunks whose document path contains *any* of the listed substrings.
- `frontmatter_filters: Option<HashMap<String, String>>` — equality match on parsed frontmatter fields (e.g. `{"area": "SIT", "status": "active"}`).
- `since: Option<DateTime>` — drop docs whose `source_updated_at` is older than `since`.

**Acceptance Criteria:**
- AC-003.1: `path_includes = ["03_MY_PROJECTS/02_HEIMDALL"]` returns only docs whose `markdown_path` includes that substring.
- AC-003.2: `path_excludes = ["06_ARCHIVE"]` drops archived docs.
- AC-003.3: `frontmatter_filters = {"area": "SIT"}` returns only docs with that exact area value.
- AC-003.4: `since = "2026-04-01T00:00:00Z"` returns only docs updated on or after that timestamp.
- AC-003.5: Multiple filters combine with AND semantics.
- AC-003.6: Filters apply post-LanceDB retrieval (no schema migration of the chunks table) — implementation reads metadata_store rows by doc_id and filters in memory.

### REQ-004: Document-Level Grouping [MUST]
**Pattern:** State-Driven

The `SearchEngine::search` API shall accept a `group_by_doc: bool` parameter; when true, only the highest-scoring chunk per `doc_id` is returned, with chunk count attached as metadata.

**Acceptance Criteria:**
- AC-004.1: `group_by_doc = true` with a query that matches 5 chunks across 2 docs returns 2 results, each showing the best matching chunk.
- AC-004.2: Each grouped result includes `chunk_count` (how many chunks of this doc matched in the underlying search).
- AC-004.3: `group_by_doc = false` (default) preserves existing chunk-level behavior.
- AC-004.4: Grouping happens *after* filtering and *before* the final `limit` truncation.

### REQ-005: MCP Tool Exposure [MUST]
**Pattern:** Ubiquitous

The MCP tools `search_library` and `search_books` shall expose the new filters and grouping options. Tool descriptions shall include usage guidance for LLM clients.

**Acceptance Criteria:**
- AC-005.1: `search_library` accepts optional input properties: `path_includes` (array of strings), `path_excludes` (array), `frontmatter` (object of `{field: value}`), `since` (ISO 8601 string), `group_by_doc` (bool, default true).
- AC-005.2: `search_books` accepts the same filter set, kept consistent with `search_library`.
- AC-005.3: Default `group_by_doc` for MCP callers is **true** — agents almost always want document-level results, not chunk lists.
- AC-005.4: Filter values are validated; invalid types yield JSON-RPC error responses (no crashes).
- AC-005.5: Tool input schemas describe each filter clearly enough for an LLM client to use them correctly without further docs.

### REQ-006: browse_topics Auto-Population [SHOULD]
**Pattern:** Ubiquitous

Once REQ-001 is in place, the existing `browse_topics` MCP tool shall return non-empty results for filesystem-sourced documents that have frontmatter `tags`, with no further changes required to its implementation.

**Acceptance Criteria:**
- AC-006.1: After a `colibri sync` of a vault with tagged content, `browse_topics` returns at least one tag.
- AC-006.2: `browse_topics(classification = "internal")` filter still works as before.

### REQ-007: Doc-Level Result Schema [MUST]
**Pattern:** Ubiquitous

When `group_by_doc = true`, the `SearchResult` struct (and MCP JSON output) shall include the document's frontmatter map alongside the matching chunk, so callers can render a richer hit without a follow-up read.

**Acceptance Criteria:**
- AC-007.1: Each grouped result includes `frontmatter: HashMap<String, JsonValue>` populated from the metadata store (may be empty).
- AC-007.2: The MCP JSON response includes the `frontmatter` field for each hit.
- AC-007.3: Backwards compatibility: `group_by_doc = false` does not introduce the `frontmatter` field — existing chunk-mode response shape is unchanged.

## Non-Goals

- **No LanceDB schema migration.** Filtering happens in-memory after retrieval. This avoids reindexing the existing 181k chunks and keeps the change scoped to metadata.
- **No range/comparison filters beyond `since`.** Equality on frontmatter fields and substring on paths cover the high-value cases. Range filters can come later.
- **No filter on chunk-level tags.** Tags live on documents, not chunks.
- **No new connector types.** This is purely a filesystem-connector + serving-plane change.
- **No reranking changes.** Hybrid search behavior unchanged; filtering is additive.

## Backward Compatibility

- Existing CLI calls (`colibri search QUERY`) without filter arguments behave identically to today.
- Existing MCP clients calling `search_library` / `search_books` without the new fields get the same results as before, except `group_by_doc` defaults to `true` in MCP — chunk-mode behavior reverts on `group_by_doc: false`.
- Schema migration is auto-applied at startup; old DBs upgrade transparently. Pre-migration data has empty `frontmatter_json` until next `colibri sync`.

## Performance Expectations

- Frontmatter parsing per file: < 1ms per typical Markdown file (parse a small YAML header).
- Post-LanceDB filter join with metadata store: O(top_k_per_backend) lookups by doc_id (already indexed), well under 50ms total for default top_k.
- Doc-level grouping: in-memory dedup of N hits where N ≤ top_k_per_backend × num_backends. O(N).
- Net wall-clock impact on a typical query: < 100ms additional for full filter+group path.
