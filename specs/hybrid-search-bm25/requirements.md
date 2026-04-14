# Hybrid Search (BM25) Requirements

> **Feature:** hybrid-search-bm25
> **Date:** 2026-04-13
> **Repo:** CoLibri
> **Repo Type:** code
> **Spec Directory:** specs/hybrid-search-bm25/
> **Design Doc Location:** [specs/hybrid-search-bm25/design.md](design.md)
> **Status:** Draft

---

## Context

CoLibri's search is currently pure vector-based (L2 distance on LanceDB with Ollama embeddings). This works well for semantic queries but falls short on exact keyword matches — searching for specific terms like "ATAM", "C4", API names, test IDs, or author names can miss relevant results because embeddings fuzz over precise terminology. Adding BM25 full-text search alongside vector search (hybrid search) would combine the strengths of both approaches: semantic understanding for conceptual queries and exact keyword matching for precise lookups. LanceDB supports FTS natively, and bge-m3 was designed for hybrid retrieval, making this a natural extension with low infrastructure cost.

## Requirements

### REQ-001: FTS Index Creation [MUST]
**Pattern:** Event-Driven

When a sync operation completes chunking and vector indexing for a profile, the system shall create or update a full-text search (BM25) index on the `text` column of that profile's LanceDB `chunks` table.

**Acceptance Criteria:**
- [ ] AC-001.1: After `colibri sync`, a FTS index exists on the `text` column of each profile's chunks table and can be queried via LanceDB's FTS API
- [ ] AC-001.2: Running `colibri sync` twice without changes does not error on the existing FTS index
- [ ] AC-001.3: A `--force` rebuild recreates the FTS index alongside the vector index

---

### REQ-002: Search Modes [MUST]
**Pattern:** Ubiquitous

The SearchEngine shall support three search modes: `hybrid` (BM25 + vector fusion), `semantic` (vector-only, current behavior), and `keyword` (BM25-only).

**Acceptance Criteria:**
- [ ] AC-002.1: `semantic` mode returns results identical to the current vector-only search behavior
- [ ] AC-002.2: `keyword` mode returns results ranked by BM25 score without embedding the query
- [ ] AC-002.3: `hybrid` mode returns results that combine both BM25 and vector rankings

---

### REQ-003: Default to Hybrid [MUST]
**Pattern:** Ubiquitous

The SearchEngine shall default to `hybrid` mode when no search mode is specified.

**Acceptance Criteria:**
- [ ] AC-003.1: Calling `search()` without a mode parameter produces hybrid results
- [ ] AC-003.2: MCP tools with no `mode` parameter use hybrid search

---

### REQ-004: Reciprocal Rank Fusion [MUST]
**Pattern:** State-Driven

While in hybrid mode, the SearchEngine shall combine BM25 and vector search results using Reciprocal Rank Fusion (RRF). Implementation delegates to LanceDB's native hybrid query with built-in RRF reranker.

**Acceptance Criteria:**
- [ ] AC-004.1: Hybrid mode uses LanceDB's native RRF fusion (via chained `.full_text_search()` + `.nearest_to()`)
- [ ] AC-004.2: Results present in only one signal (BM25 or vector) still appear in the fused results
- [ ] AC-004.3: Final results are sorted by descending RRF score

---

### REQ-005: MCP Tool Mode Parameter [MUST]
**Pattern:** Optional

Where a `mode` parameter is provided to `search_library` or `search_books` MCP tools, the system shall use the specified search mode (`hybrid`, `semantic`, or `keyword`) for that query.

**Acceptance Criteria:**
- [ ] AC-005.1: `search_library` accepts an optional `mode` string parameter with values `hybrid`, `semantic`, `keyword`
- [ ] AC-005.2: `search_books` accepts the same optional `mode` parameter
- [ ] AC-005.3: An invalid mode value returns a JSON-RPC error with a descriptive message

---

### REQ-006: CLI Mode Flag [MUST]
**Pattern:** Optional

Where a `--mode` flag is provided to the `colibri search` CLI command, the system shall use the specified search mode for that query.

**Acceptance Criteria:**
- [ ] AC-006.1: `colibri search --mode keyword "ATAM"` returns BM25-only results
- [ ] AC-006.2: `colibri search --mode semantic "architecture quality"` returns vector-only results
- [ ] AC-006.3: `colibri search "query"` (no flag) defaults to hybrid mode

---

### REQ-007: Multi-Profile FTS [MUST]
**Pattern:** Ubiquitous

The FTS index shall be created per embedding profile, consistent with the existing multi-profile architecture. Hybrid search shall query all queryable profiles concurrently, same as vector search does today.

**Acceptance Criteria:**
- [ ] AC-007.1: Each profile's LanceDB table has its own FTS index
- [ ] AC-007.2: Hybrid search results include matches from all queryable profiles
- [ ] AC-007.3: If one profile's FTS query fails, the system logs a warning and continues with other profiles (graceful degradation)

---

### REQ-008: Embedding Failure Fallback [SHOULD]
**Pattern:** Unwanted

If vector embedding fails during a hybrid search query, then the system shall fall back to keyword-only search and log a warning to stderr.

**Acceptance Criteria:**
- [ ] AC-008.1: When Ollama is unreachable, `hybrid` mode degrades to `keyword` results instead of returning an error
- [ ] AC-008.2: A warning message is logged to stderr indicating the fallback

---

### REQ-009: Score Transparency [SHOULD]
**Pattern:** Ubiquitous

Search results shall include the search mode used and the fused score. In hybrid mode, individual component ranks should be available for debugging.

**Acceptance Criteria:**
- [ ] AC-009.1: SearchResult includes a `search_mode` field indicating which mode produced the result
- [ ] AC-009.2: MCP tool responses include the search mode in the response metadata

---

## Deliverables Contract

The following deliverables are committed for this feature:

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 1 | requirements.md | ☑ Committed | This document |
| 2 | design.md | ☑ Committed | |
| 3 | Architecture diagram | ☑ Committed | |
| 5 | Implementation plan | ☑ Committed | |
| 6 | Unit tests (TDD) | ☑ Committed | |
| 7 | Integration tests | ☑ Committed | |
| 10 | Linter clean | ☑ Committed | |
| 12 | Code review | ☑ Committed | |
| 13 | Documentation update | ☑ Committed | |
| 14 | Changelog entry | ☑ Committed | |
| 17 | verification.md | ☑ Committed | |
| 18 | Acceptance criteria met | ☑ Committed | |

**Mandatory (cannot skip):** requirements.md, verification.md, acceptance criteria met

## Constraints

- LanceDB's native FTS must be used — no external search infrastructure (Elasticsearch, Meilisearch, etc.)
- Must work with the existing Ollama-based local embedding pipeline
- Must preserve the multi-profile architecture (classification-based routing)
- RRF fusion delegated to LanceDB's built-in reranker (not custom implementation)
- Local-first philosophy: no cloud dependencies introduced

## Out of Scope

- Weighted linear fusion (potential follow-up to RRF)
- bge-m3 sparse vector / ColBERT vector support (future extension)
- Per-field FTS boosting (e.g., weighting title higher than body text)
- Search analytics or A/B testing between modes
- FTS on metadata columns (tags, title, source_name) — only the `text` column
- Custom RRF implementation (using LanceDB's built-in reranker instead)

## Traceability

| REQ | Design Section | Plan Task | Test / Verification |
|-----|---------------|-----------|---------------------|
| REQ-001 | §3 Indexer Changes | Task 7 | `colibri sync --force` creates FTS index |
| REQ-002 | §2 Search Engine Changes | Tasks 1, 4 | `cargo test search_mode` + manual test |
| REQ-003 | §1 New Types | Task 1 | `test search_mode_default_is_hybrid` |
| REQ-004 | §2 Per-backend branching | Task 4 | `.full_text_search().nearest_to()` chain |
| REQ-005 | §4 MCP Server Changes | Task 6 | MCP mode param + invalid value error |
| REQ-006 | §5 CLI Changes | Task 5 | `colibri search --mode keyword "ATAM"` |
| REQ-007 | §3 Multi-profile FTS | Task 7 | Per-profile loop in indexer |
| REQ-008 | §2 Hybrid fallback | Task 4 | Hybrid Err(e) → keyword fallback |
| REQ-009 | §1 SearchResult, §4 MCP | Tasks 2, 6 | search_mode field in results + response |
