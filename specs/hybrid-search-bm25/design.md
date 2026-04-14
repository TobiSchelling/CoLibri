# Hybrid Search (BM25) — Design Specification

> **Feature:** hybrid-search-bm25
> **Date:** 2026-04-13
> **Requirements:** [specs/hybrid-search-bm25/requirements.md](requirements.md)
> **Status:** Draft

---

## Overview

Extend CoLibri's search from pure vector-based (L2 distance on LanceDB) to hybrid search combining BM25 full-text search with vector similarity. Uses LanceDB 0.23's native FTS and hybrid query capabilities — no additional infrastructure needed.

**Key design decision:** LanceDB natively supports hybrid queries (chaining `.full_text_search()` + `.nearest_to()` on a single query), with built-in RRF fusion. We use this rather than implementing manual RRF, reducing complexity and leveraging battle-tested internals. Note: REQ-004 specifies RRF with k=60 — LanceDB's internal RRF implementation uses its own k constant. We accept LanceDB's default rather than implementing custom RRF, trading tunability for simplicity.

**Scoring behavior by mode:**
- **Semantic:** `similarity_threshold` config applies (score = `exp(-distance)`, range 0–1)
- **Keyword/Hybrid:** `similarity_threshold` does NOT apply (BM25/RRF scores have different scales). All results returned up to `limit`.

## Architecture

### Data Flow — Current vs. New

```
CURRENT (semantic only):
  Query → Ollama embed → vector_search(vec) → L2 distance → exp(-d) score → results

NEW (three modes):
  Query + mode
    ├─ Semantic:  Ollama embed → vector_search(vec) → _distance → exp(-d) score → results
    ├─ Keyword:   FTS query → full_text_search(q) → _score (BM25) → results
    └─ Hybrid:    Ollama embed + FTS query → full_text_search(q).nearest_to(vec)
                  → LanceDB internal RRF → _score (fused) → results

  All modes: → deduplicate → metadata enrich → filter → return
```

### FTS Index Lifecycle

```
colibri sync
  └─ per profile:
       ├─ chunk documents
       ├─ embed chunks → Ollama
       ├─ write to LanceDB table (create or append)
       └─ create FTS index on "text" column   ← NEW
            └─ table.create_index(&["text"], Index::FTS(default))
                 .replace(true)  // idempotent
                 .execute().await
```

## Detailed Design

### 1. New Types

#### SearchMode enum (`src/query.rs`)

```rust
/// Search mode controlling how queries are executed against LanceDB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// BM25 full-text search combined with vector similarity via LanceDB native RRF.
    #[default]
    Hybrid,
    /// Vector-only search using embedding similarity (original behavior).
    Semantic,
    /// BM25 full-text search only — no embedding needed.
    Keyword,
}
```

Implements:
- `Default` → `Hybrid` (REQ-003)
- `FromStr` / `Display` for CLI parsing and MCP responses
- `clap::ValueEnum` for CLI flag integration
- `Serialize` / `Deserialize` for JSON output

#### SearchResult extension (`src/query.rs`)

Add `search_mode` field:

```rust
pub struct SearchResult {
    pub text: String,
    pub file: String,
    pub title: String,
    pub doc_type: String,
    pub classification: String,
    pub score: f64,
    pub search_mode: SearchMode,  // NEW — which mode produced this result
}
```

### 2. Search Engine Changes (`src/query.rs`)

#### Method signature

```rust
pub async fn search(
    &self,
    query: &str,
    classification: Option<&str>,
    doc_type: Option<&str>,
    limit: usize,
    mode: SearchMode,  // NEW parameter
) -> Result<Vec<SearchResult>, ColibriError>
```

Convenience methods updated:

```rust
pub async fn search_library(&self, query: &str, limit: usize, mode: SearchMode) -> Result<...>
pub async fn search_books(&self, query: &str, limit: usize, mode: SearchMode) -> Result<...>
```

#### Per-backend query branching

Inside the per-backend search loop, branch on mode:

```rust
let batches = match mode {
    SearchMode::Semantic => {
        let query_vector = embed_query(query, &backend).await?;
        backend.table.vector_search(query_vector)
            .limit(per_backend_limit)
            .execute().await?
    }
    SearchMode::Keyword => {
        let fts_query = FullTextSearchQuery::new(query.to_string());
        backend.table.query()
            .full_text_search(fts_query)
            .limit(per_backend_limit)
            .execute().await?
    }
    SearchMode::Hybrid => {
        match embed_query(query, &backend).await {
            Ok(query_vector) => {
                let fts_query = FullTextSearchQuery::new(query.to_string());
                backend.table.query()
                    .full_text_search(fts_query)
                    .nearest_to(&query_vector)?
                    .limit(per_backend_limit)
                    .execute().await?
            }
            Err(e) => {
                // REQ-008: fallback to keyword on embedding failure
                eprintln!("Warning: embedding failed, falling back to keyword search: {e}");
                let fts_query = FullTextSearchQuery::new(query.to_string());
                backend.table.query()
                    .full_text_search(fts_query)
                    .limit(per_backend_limit)
                    .execute().await?
            }
        }
    }
};
```

#### Score extraction

The `collect_search_hits()` function must handle two score columns:

| Mode | Column | Conversion |
|------|--------|------------|
| Semantic | `_distance` (Float32) | `score = (-distance).exp()` |
| Keyword | `_score` (Float32) | `score = _score` (BM25 score, used directly) |
| Hybrid | `_score` (Float32) | `score = _score` (LanceDB's fused RRF score) |

Logic: try `_score` first (present in keyword/hybrid). If absent, fall back to `_distance` (semantic). This makes the function mode-agnostic.

The existing `similarity_threshold` filter applies to semantic mode only (where the score is a normalized similarity). For keyword and hybrid modes, all results are returned (BM25/RRF scores have different scales).

### 3. Indexer Changes (`src/indexer.rs`)

#### FTS index creation

After writing vector data to LanceDB (both full rebuild and incremental paths), create the FTS index:

```rust
use lancedb::index::Index;

// After table write completes
table.create_index(&["text"], Index::FTS(Default::default()))
    .replace(true)
    .execute()
    .await?;
```

**FTS index builder defaults** (from `lancedb::index::FtsIndexBuilder`):
- Tokenizer: `simple` (whitespace + punctuation splitting)
- Language: English
- Stemming: enabled
- Stop-word removal: enabled
- Case folding: enabled (lowercase)
- ASCII folding: enabled (accent normalization)

These defaults are appropriate for CoLibri's content (English technical books and documentation).

**Placement:** Both paths in `index_profile()` end with the same FTS creation call:
- Full rebuild: after `db.create_table(TABLE_NAME, batches).mode(Overwrite)`
- Incremental: after `tbl.delete(...)` + `tbl.add(batches)`

**Error handling:** If FTS index creation fails, log the error and continue. The vector index is still usable — users can search in `semantic` mode. This prevents an FTS-specific issue from blocking the entire sync pipeline.

### 4. MCP Server Changes (`src/mcp.rs`)

#### Tool definitions update

Add `mode` parameter to `search_library` and `search_books`:

```json
{
  "name": "mode",
  "description": "Search mode: 'hybrid' (default, combines keyword + semantic), 'semantic' (vector similarity only), or 'keyword' (BM25 text matching only)",
  "type": "string",
  "enum": ["hybrid", "semantic", "keyword"]
}
```

This parameter is optional — omitting it defaults to `hybrid`.

#### Response format update

Add `search_mode` to search responses:

```json
{
  "query": "ATAM evaluation",
  "search_mode": "hybrid",
  "total_results": 5,
  "results": [
    {
      "title": "Software Architecture in Practice",
      "score": 0.0312,
      "type": "book",
      "file": "/path/to/file.md",
      "text": "The ATAM method evaluates..."
    }
  ]
}
```

#### Error handling

Invalid mode values return a JSON-RPC error:
```json
{
  "code": -32602,
  "message": "Invalid mode 'fuzzy'. Must be one of: hybrid, semantic, keyword"
}
```

### 5. CLI Changes (`src/cli/mod.rs`, `src/cli/search.rs`)

#### New `--mode` flag

```rust
Search {
    query: String,
    #[arg(short, long, default_value_t = 5)]
    limit: usize,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    doc_type: Option<String>,
    #[arg(long)]
    classification: Option<String>,
    #[arg(long, value_enum, default_value_t = SearchMode::Hybrid)]
    mode: SearchMode,
}
```

The `run()` function passes `mode` through to `engine.search()`. Display output is unchanged — scores already appear in the output.

## Error Handling Summary

| Scenario | Mode | Behavior |
|----------|------|----------|
| Ollama unreachable | Hybrid | Degrade to keyword, log warning to stderr |
| Ollama unreachable | Semantic | Return `ColibriError` (embedding required) |
| Ollama unreachable | Keyword | Works normally (no embedding needed) |
| No FTS index | Keyword/Hybrid | Return error suggesting `colibri sync` |
| FTS index creation fails | (sync) | Log error, continue — vector index still usable |
| One profile FTS fails | Any | Log warning, return results from other profiles |

## Testing Strategy

### Unit Tests (`src/query.rs`)

- `SearchMode::from_str("hybrid")` → `Ok(Hybrid)`, same for semantic/keyword
- `SearchMode::from_str("invalid")` → `Err`
- `SearchMode::default()` → `Hybrid`
- `SearchMode::display()` → lowercase string
- Score extraction handles `_score` and `_distance` columns

### Integration Tests (`tests/hybrid_search.rs`)

- Create in-memory LanceDB table with known text content and vectors
- Build FTS index on text column
- **Semantic mode**: returns results ranked by vector similarity
- **Keyword mode**: returns results for exact term matches (no embedding called)
- **Hybrid mode**: returns results combining both signals
- **Keyword finds what semantic misses**: insert doc with specific term (e.g., "ATAM"), verify keyword finds it even if embedding similarity is low
- **Embedding failure fallback**: simulate Ollama failure in hybrid mode, verify keyword results returned

### Indexer Tests (`src/indexer.rs`)

- FTS index created after full rebuild
- FTS index created after incremental update
- FTS index survives re-sync (idempotent via `replace(true)`)
- FTS index failure doesn't block sync

### MCP Tests

- Valid mode parameter parsed correctly
- Missing mode defaults to hybrid
- Invalid mode returns JSON-RPC error with code -32602

## Files Modified

| File | Change |
|------|--------|
| `src/query.rs` | Add `SearchMode` enum, update `search()` signature, mode branching in query execution, score extraction for `_score`/`_distance` |
| `src/indexer.rs` | Add FTS index creation after table writes |
| `src/mcp.rs` | Add `mode` parameter to search tools, `search_mode` in responses |
| `src/cli/mod.rs` | Add `--mode` flag to Search command |
| `src/cli/search.rs` | Pass mode through to `engine.search()` |
| `Cargo.toml` | No changes needed — lancedb 0.23 already supports FTS |
| `tests/hybrid_search.rs` | New integration test file |

## Dependencies

No new crate dependencies. LanceDB 0.23.1 includes full FTS support via its existing `lancedb::index::Index::FTS` variant and `FullTextSearchQuery` type.

## Migration

**Existing users:** After upgrading, the first `colibri sync` (or `colibri sync --force`) will create FTS indexes. Until then, keyword and hybrid modes will return an error suggesting a sync. Semantic mode continues to work immediately (backward compatible).

**No schema migration needed:** FTS is an index overlay on the existing `text` column, not a schema change.
