# Hybrid Search (BM25) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid BM25 + vector search to CoLibri using LanceDB's native FTS and hybrid query support.

**Architecture:** Three search modes (hybrid, semantic, keyword) controlled by a `SearchMode` enum. Mode branching happens inside `SearchEngine::search()`. FTS index created automatically during `colibri sync`. LanceDB handles RRF fusion internally for hybrid mode.

**Tech Stack:** Rust, LanceDB 0.23 (existing — FTS support built in), clap (CLI), JSON-RPC (MCP)

**Design spec:** `specs/hybrid-search-bm25/design.md`
**Requirements:** `specs/hybrid-search-bm25/requirements.md`

---

### Task 1: Add SearchMode Enum

**Files:**
- Modify: `src/query.rs:1-14` (imports) and after line 30 (new enum)

- [ ] **Step 1: Add SearchMode enum and imports**

Add to imports at top of `src/query.rs`:

```rust
use std::fmt;
use std::str::FromStr;
```

Add the `SearchMode` enum after the `SearchResult` struct (after line 30):

```rust
/// Controls how search queries are executed against LanceDB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum SearchMode {
    /// BM25 + vector combined via LanceDB native RRF.
    #[default]
    Hybrid,
    /// Vector-only search (original behavior).
    Semantic,
    /// BM25 full-text search only.
    Keyword,
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Hybrid => write!(f, "hybrid"),
            SearchMode::Semantic => write!(f, "semantic"),
            SearchMode::Keyword => write!(f, "keyword"),
        }
    }
}

impl FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(SearchMode::Hybrid),
            "semantic" => Ok(SearchMode::Semantic),
            "keyword" => Ok(SearchMode::Keyword),
            other => Err(format!(
                "Invalid search mode '{other}'. Must be one of: hybrid, semantic, keyword"
            )),
        }
    }
}

impl clap::ValueEnum for SearchMode {
    fn value_variants<'a>() -> &'a [Self] {
        &[SearchMode::Hybrid, SearchMode::Semantic, SearchMode::Keyword]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            SearchMode::Hybrid => Some(clap::builder::PossibleValue::new("hybrid")),
            SearchMode::Semantic => Some(clap::builder::PossibleValue::new("semantic")),
            SearchMode::Keyword => Some(clap::builder::PossibleValue::new("keyword")),
        }
    }
}
```

- [ ] **Step 2: Add unit tests for SearchMode**

Add a `#[cfg(test)]` module at the bottom of `src/query.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_mode_default_is_hybrid() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    #[test]
    fn search_mode_from_str_valid() {
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("keyword".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert_eq!("HYBRID".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("Semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
    }

    #[test]
    fn search_mode_from_str_invalid() {
        let err = "fuzzy".parse::<SearchMode>().unwrap_err();
        assert!(err.contains("Invalid search mode"));
        assert!(err.contains("fuzzy"));
    }

    #[test]
    fn search_mode_display() {
        assert_eq!(SearchMode::Hybrid.to_string(), "hybrid");
        assert_eq!(SearchMode::Semantic.to_string(), "semantic");
        assert_eq!(SearchMode::Keyword.to_string(), "keyword");
    }
}
```

- [ ] **Step 3: Run tests to verify**

Run: `cargo test search_mode`
Expected: All 4 tests pass.

- [ ] **Step 4: Run type check**

Run: `cargo check`
Expected: Compiles cleanly. The new enum is defined but not yet used — no warnings expected since it's referenced by tests.

- [ ] **Step 5: Commit**

```bash
git add src/query.rs
git commit -m "feat(search): add SearchMode enum with hybrid/semantic/keyword variants"
```

---

### Task 2: Update SearchResult and search() Signature

**Files:**
- Modify: `src/query.rs:21-30` (SearchResult struct)
- Modify: `src/query.rs:127-133` (search method signature)
- Modify: `src/query.rs:245-252` (SearchResult construction)
- Modify: `src/query.rs:267-282` (convenience methods)

- [ ] **Step 1: Add search_mode field to SearchResult**

In `src/query.rs`, update the `SearchResult` struct (currently lines 21-30):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub file: String,
    pub title: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub classification: String,
    pub score: f64,
    pub search_mode: SearchMode,
}
```

- [ ] **Step 2: Add mode parameter to search()**

Update the `search()` method signature (currently line 127):

```rust
pub async fn search(
    &self,
    query: &str,
    classification: Option<&str>,
    doc_type: Option<&str>,
    limit: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>, ColibriError> {
```

- [ ] **Step 3: Update SearchResult construction inside search()**

Find the `SearchResult` construction in the dedup loop (currently lines 245-252) and add the `search_mode` field:

```rust
let result = SearchResult {
    text: hit.text,
    file: doc.markdown_path.clone(),
    title: doc.title.clone(),
    doc_type: doc.doc_type.clone(),
    classification: doc.classification.clone(),
    score: hit.score,
    search_mode: mode,
};
```

- [ ] **Step 4: Update convenience methods**

Update `search_books()` (lines 267-273):

```rust
pub async fn search_books(
    &self,
    query: &str,
    limit: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>, ColibriError> {
    self.search(query, None, Some("book"), limit, mode).await
}
```

Update `search_library()` (lines 276-282):

```rust
pub async fn search_library(
    &self,
    query: &str,
    limit: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>, ColibriError> {
    self.search(query, None, None, limit, mode).await
}
```

- [ ] **Step 5: Run type check**

Run: `cargo check`
Expected: Compile errors in `src/mcp.rs` and `src/cli/search.rs` because callers don't pass `mode` yet. This is expected — we'll fix callers in subsequent tasks.

- [ ] **Step 6: Commit (even with downstream errors)**

```bash
git add src/query.rs
git commit -m "feat(search): add mode parameter to search() and search_mode to SearchResult"
```

---

### Task 3: Update collect_search_hits for FTS Scores

**Files:**
- Modify: `src/query.rs:364-397` (collect_search_hits function)

- [ ] **Step 1: Update collect_search_hits to handle both _score and _distance**

Replace the `collect_search_hits` function (lines 364-397):

```rust
fn collect_search_hits(
    batches: &[RecordBatch],
    similarity_threshold: f64,
    mode: SearchMode,
    out: &mut Vec<SearchHit>,
) {
    for batch in batches {
        let num_rows = batch.num_rows();

        let doc_id_col = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let text_col = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        // FTS/hybrid results use _score; vector results use _distance.
        let score_col = batch
            .column_by_name("_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        let dist_col = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        for i in 0..num_rows {
            let score = if let Some(sc) = score_col {
                // _score from BM25 or RRF fusion — use directly
                sc.value(i) as f64
            } else {
                // _distance from vector search — convert to similarity
                let distance = dist_col.map(|c| c.value(i) as f64).unwrap_or(0.0);
                (-distance).exp()
            };

            // Only apply similarity_threshold for semantic mode (where scores are 0-1).
            if mode == SearchMode::Semantic && score < similarity_threshold {
                continue;
            }

            let doc_id = doc_id_col.map(|c| c.value(i)).unwrap_or("");
            out.push(SearchHit {
                doc_id: doc_id.to_string(),
                text: text_col.map(|c| c.value(i).to_string()).unwrap_or_default(),
                score: (score * 10000.0).round() / 10000.0,
            });
        }
    }
}
```

- [ ] **Step 2: Update the call site in search()**

Find the line calling `collect_search_hits` (currently line 205):

```rust
// Old:
collect_search_hits(&batches, self.config.similarity_threshold, &mut merged);

// New:
collect_search_hits(&batches, self.config.similarity_threshold, mode, &mut merged);
```

- [ ] **Step 3: Run type check**

Run: `cargo check`
Expected: Still compile errors from downstream callers (mcp.rs, cli/search.rs) but no new errors in query.rs.

- [ ] **Step 4: Commit**

```bash
git add src/query.rs
git commit -m "feat(search): handle FTS _score column alongside vector _distance"
```

---

### Task 4: Implement Per-Backend Query Branching

**Files:**
- Modify: `src/query.rs:11` (imports)
- Modify: `src/query.rs:143-206` (backend search loop)

- [ ] **Step 1: Add LanceDB FTS imports**

Update the imports at the top of `src/query.rs`. Add to the existing lancedb import line:

```rust
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::scalar::FullTextSearchQuery;
```

- [ ] **Step 2: Refactor the backend loop to branch on mode**

Replace the backend search loop body (the part from `let query_vector = match embed_texts` through `collect_search_hits`) inside `search()`. The new loop body should be:

```rust
for backend in &self.backends {
    // Refresh to latest LanceDB version so externally-rebuilt indexes are visible.
    if let Err(e) = backend.table.checkout_latest().await {
        warn!(
            "Failed to refresh table for profile '{}': {e}",
            backend.profile_id
        );
    }

    let batches: Vec<RecordBatch> = match mode {
        SearchMode::Semantic => {
            // Vector-only search (original behavior)
            let query_vector = match embed_texts(
                &[query.to_string()],
                &backend.embedding_model,
                &backend.embedding_endpoint,
            )
            .await
            {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    warn!(
                        "Skipping profile '{}': embedding returned empty vector",
                        backend.profile_id
                    );
                    continue;
                }
                Err(e) => {
                    warn!(
                        "Skipping profile '{}': embed error: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            let search = backend
                .table
                .vector_search(query_vector[0].clone())
                .map_err(|e| ColibriError::Query(format!("Vector search failed: {e}")))?
                .limit(per_backend_limit);

            match search.execute().await {
                Ok(stream) => match stream.try_collect().await {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Skipping profile '{}': collect error: {e}", backend.profile_id);
                        continue;
                    }
                },
                Err(e) => {
                    warn!("Skipping profile '{}': query error: {e}", backend.profile_id);
                    continue;
                }
            }
        }
        SearchMode::Keyword => {
            // BM25 full-text search only — no embedding needed
            let fts_query = FullTextSearchQuery::new(query.to_string());
            let search = backend
                .table
                .query()
                .full_text_search(fts_query)
                .limit(per_backend_limit);

            match search.execute().await {
                Ok(stream) => match stream.try_collect().await {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Skipping profile '{}': FTS collect error: {e}", backend.profile_id);
                        continue;
                    }
                },
                Err(e) => {
                    warn!("Skipping profile '{}': FTS query error: {e}", backend.profile_id);
                    continue;
                }
            }
        }
        SearchMode::Hybrid => {
            // Combined FTS + vector — LanceDB handles RRF fusion.
            // On embed failure, falls back to keyword-only (REQ-008).
            match embed_texts(
                &[query.to_string()],
                &backend.embedding_model,
                &backend.embedding_endpoint,
            )
            .await
            {
                Ok(v) if !v.is_empty() => {
                    let fts_query = FullTextSearchQuery::new(query.to_string());
                    let search = backend
                        .table
                        .query()
                        .full_text_search(fts_query)
                        .nearest_to(v[0].as_slice())
                        .map_err(|e| ColibriError::Query(format!("Hybrid search failed: {e}")))?
                        .limit(per_backend_limit);

                    match search.execute().await {
                        Ok(stream) => match stream.try_collect().await {
                            Ok(b) => b,
                            Err(e) => {
                                warn!("Skipping profile '{}': hybrid collect error: {e}", backend.profile_id);
                                continue;
                            }
                        },
                        Err(e) => {
                            warn!("Skipping profile '{}': hybrid query error: {e}", backend.profile_id);
                            continue;
                        }
                    }
                }
                Ok(_) => {
                    warn!(
                        "Skipping profile '{}': embedding returned empty vector",
                        backend.profile_id
                    );
                    continue;
                }
                Err(e) => {
                    // REQ-008: fallback to keyword on embedding failure
                    warn!(
                        "Embedding failed for profile '{}', falling back to keyword search: {e}",
                        backend.profile_id
                    );
                    let fts_query = FullTextSearchQuery::new(query.to_string());
                    let search = backend
                        .table
                        .query()
                        .full_text_search(fts_query)
                        .limit(per_backend_limit);

                    match search.execute().await {
                        Ok(stream) => match stream.try_collect().await {
                            Ok(b) => b,
                            Err(e) => {
                                warn!("Skipping profile '{}': FTS fallback error: {e}", backend.profile_id);
                                continue;
                            }
                        },
                        Err(e) => {
                            warn!("Skipping profile '{}': FTS fallback query error: {e}", backend.profile_id);
                            continue;
                        }
                    }
                }
            }
        }
    };

    succeeded += 1;
    collect_search_hits(&batches, self.config.similarity_threshold, mode, &mut merged);
}
```

- [ ] **Step 3: Run type check**

Run: `cargo check`
Expected: Errors only in downstream callers (mcp.rs, cli/search.rs). The query.rs file itself should compile.

- [ ] **Step 4: Commit**

```bash
git add src/query.rs
git commit -m "feat(search): implement per-backend query branching for all three modes"
```

---

### Task 5: Update CLI to Pass SearchMode

**Files:**
- Modify: `src/cli/mod.rs:154-174` (Search command definition)
- Modify: `src/cli/search.rs:1-56` (search runner)
- Modify: `src/main.rs:91-97` (dispatch)

- [ ] **Step 1: Add mode flag to CLI Search command**

In `src/cli/mod.rs`, add the import at the top:

```rust
use crate::query::SearchMode;
```

Update the `Search` variant (lines 154-174) to add the `mode` field:

```rust
/// Search the indexed library
Search {
    /// Search query
    query: String,

    /// Maximum results to return
    #[arg(short, long, default_value_t = 5)]
    limit: usize,

    /// Output as JSON
    #[arg(long)]
    json: bool,

    /// Filter by document type
    #[arg(long)]
    doc_type: Option<String>,

    /// Filter by classification
    #[arg(long)]
    classification: Option<String>,

    /// Search mode: hybrid (default), semantic, or keyword
    #[arg(long, value_enum, default_value_t = SearchMode::Hybrid)]
    mode: SearchMode,
},
```

- [ ] **Step 2: Update search::run() to accept mode**

Update `src/cli/search.rs`:

```rust
//! `colibri search` — search command with hybrid/semantic/keyword modes.

use crate::config::load_config;
use crate::query::{SearchEngine, SearchMode};

pub async fn run(
    query: String,
    limit: usize,
    json: bool,
    doc_type: Option<String>,
    classification: Option<String>,
    mode: SearchMode,
) -> anyhow::Result<()> {
    let config = load_config()?;
    let engine = SearchEngine::new(&config).await?;

    let limit = limit.min(config.top_k);
    let results = engine
        .search(
            &query,
            classification.as_deref(),
            doc_type.as_deref(),
            limit,
            mode,
        )
        .await?;

    if json {
        let output = serde_json::json!({
            "query": query,
            "search_mode": mode.to_string(),
            "total_results": results.len(),
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        if results.is_empty() {
            eprintln!("No results found for: {query}");
            return Ok(());
        }

        for (i, result) in results.iter().enumerate() {
            println!("{}. {} (score: {:.4})", i + 1, result.title, result.score);
            println!("   File: {}", result.file);
            if !result.classification.is_empty() {
                println!("   Classification: {}", result.classification);
            }
            // Show truncated text preview
            let preview: String = result.text.chars().take(200).collect();
            let ellipsis = if result.text.len() > 200 { "..." } else { "" };
            println!("   {preview}{ellipsis}");
            println!();
        }

        eprintln!("{} result(s) found", results.len());
    }

    Ok(())
}
```

- [ ] **Step 3: Update main.rs dispatch**

In `src/main.rs`, find the `Commands::Search` match arm (line 91) and add `mode`:

```rust
cli::Commands::Search {
    query,
    limit,
    json,
    doc_type,
    classification,
    mode,
} => cli::search::run(query, limit, json, doc_type, classification, mode).await,
```

- [ ] **Step 4: Run type check**

Run: `cargo check`
Expected: Errors only in `src/mcp.rs` (callers not yet updated).

- [ ] **Step 5: Commit**

```bash
git add src/cli/mod.rs src/cli/search.rs src/main.rs
git commit -m "feat(cli): add --mode flag to colibri search command"
```

---

### Task 6: Update MCP Server Tools

**Files:**
- Modify: `src/mcp.rs:162-234` (tool definitions)
- Modify: `src/mcp.rs:237-343` (tool call handler)

- [ ] **Step 1: Add mode parameter to tool definitions**

In `src/mcp.rs`, add `SearchMode` to imports (around line 14):

```rust
use crate::query::{SearchEngine, SearchMode};
```

Update `handle_tools_list()` — add `mode` property to both `search_library` and `search_books` input schemas. In each tool's `"properties"` object, add after `"limit"`:

```json
"mode": {
    "type": "string",
    "description": "Search mode: 'hybrid' (default, combines keyword + semantic), 'semantic' (vector similarity only), or 'keyword' (BM25 text matching only)",
    "enum": ["hybrid", "semantic", "keyword"]
}
```

Also update the tool descriptions:
- `search_library`: Change "Performs semantic search" to "Performs hybrid search (combining BM25 keyword matching with semantic vector search) by default"
- `search_books`: Change "Filters to documents with type 'book'." to "Filters to documents with type 'book'. Supports hybrid, semantic, and keyword search modes."

- [ ] **Step 2: Update handle_tools_call to parse mode**

In `handle_tools_call()`, add a helper to parse the mode from arguments. Before the `match tool_name` block, add:

```rust
let mode = arguments
    .get("mode")
    .and_then(|m| m.as_str())
    .map(|m| m.parse::<SearchMode>())
    .transpose()
    .map_err(|e| format!("{e}"));

let mode = match mode {
    Ok(m) => m.unwrap_or_default(),
    Err(e) => {
        return error_response(id, -32602, &e);
    }
};
```

- [ ] **Step 3: Update search_library and search_books call sites**

Update `search_library` arm (around line 260):

```rust
match engine.search_library(query, limit, mode).await {
```

Add `search_mode` to the output JSON:

```rust
let output = json!({
    "query": query,
    "search_mode": mode.to_string(),
    "total_results": results.len(),
    "results": results,
});
```

Apply the same changes to the `search_books` arm (around line 283):

```rust
match engine.search_books(query, limit, mode).await {
    Ok(results) => {
        let output = json!({
            "query": query,
            "search_mode": mode.to_string(),
            "total_results": results.len(),
            "results": results,
        });
        Ok(serde_json::to_string_pretty(&output).unwrap_or_default())
    }
    Err(e) => Err(format!("{e}")),
}
```

- [ ] **Step 4: Run full type check and lint**

Run: `cargo check && cargo clippy -- -D warnings`
Expected: Clean compilation and no clippy warnings.

- [ ] **Step 5: Run existing tests**

Run: `cargo test`
Expected: All existing tests pass. The MCP startup tests don't call search, so they're unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/mcp.rs
git commit -m "feat(mcp): add mode parameter to search_library and search_books tools"
```

---

### Task 7: Add FTS Index Creation to Indexer

**Files:**
- Modify: `src/indexer.rs:15-16` (imports)
- Modify: `src/indexer.rs:450-471` (after table write)

- [ ] **Step 1: Add LanceDB index import**

In `src/indexer.rs`, add to the imports (after line 15):

```rust
use lancedb::index::Index;
```

- [ ] **Step 2: Add FTS index creation after table writes**

After the table write block (lines 450-471), add FTS index creation. The three branches (full rebuild, incremental with existing table, new table) all end up with a table reference. Add the following after the entire `if profile_full_rebuild { ... } else if let Some(tbl) { ... } else { ... }` block:

```rust
// Create FTS index on the text column for keyword/hybrid search.
let fts_table = if profile_full_rebuild {
    db.open_table(TABLE_NAME).execute().await?
} else if let Some(tbl) = table.as_ref() {
    tbl.clone()
} else {
    db.open_table(TABLE_NAME).execute().await?
};

if let Err(e) = fts_table
    .create_index(&["text"], Index::FTS(Default::default()))
    .replace(true)
    .execute()
    .await
{
    on_progress(IndexEvent::Warning {
        message: format!(
            "FTS index creation failed for profile '{profile_id}' (keyword/hybrid search may not work): {e}"
        ),
    });
}
```

- [ ] **Step 3: Run type check**

Run: `cargo check`
Expected: Clean compilation.

- [ ] **Step 4: Run lint**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

- [ ] **Step 5: Commit**

```bash
git add src/indexer.rs
git commit -m "feat(indexer): create FTS index on text column during sync"
```

---

### Task 8: Full Build and Test

**Files:**
- No new files — validation only

- [ ] **Step 1: Run full build**

Run: `cargo build`
Expected: Clean compilation.

- [ ] **Step 2: Run full test suite**

Run: `cargo test`
Expected: All tests pass, including the new SearchMode tests from Task 1.

- [ ] **Step 3: Run lint**

Run: `cargo clippy -- -D warnings`
Expected: No clippy warnings.

- [ ] **Step 4: Run format check**

Run: `cargo fmt --check`
Expected: No formatting issues.

- [ ] **Step 5: Fix any issues found in steps 1-4, then commit**

If any fixes needed:
```bash
cargo fmt
git add -A
git commit -m "fix: address lint/format issues from hybrid search implementation"
```

---

### Task 9: Manual Integration Test

**Files:**
- No files — manual verification

- [ ] **Step 1: Rebuild the index with FTS**

Run: `colibri sync --force`
Expected: Sync completes. FTS index created for each profile (no FTS-related warnings).

- [ ] **Step 2: Test semantic mode (original behavior)**

Run: `colibri search --mode semantic "software architecture"`
Expected: Results ranked by vector similarity, similar to pre-change behavior.

- [ ] **Step 3: Test keyword mode**

Run: `colibri search --mode keyword "ATAM"`
Expected: Results containing the exact term "ATAM" ranked by BM25 score.

- [ ] **Step 4: Test hybrid mode (default)**

Run: `colibri search "architecture quality evaluation"`
Expected: Results combining semantic and keyword signals. No `--mode` flag needed (defaults to hybrid).

- [ ] **Step 5: Test JSON output includes search_mode**

Run: `colibri search --json --mode keyword "ATAM"`
Expected: JSON output contains `"search_mode": "keyword"` field.

- [ ] **Step 6: Test invalid mode**

Run: `colibri search --mode fuzzy "test"`
Expected: Clap error message listing valid values (hybrid, semantic, keyword).

---

### Task 10: Update Requirements Traceability and CLAUDE.md

**Files:**
- Modify: `specs/hybrid-search-bm25/requirements.md` (traceability table)
- Modify: `CLAUDE.md` (architecture section)

- [ ] **Step 1: Update traceability table in requirements.md**

Update the traceability table at the bottom of `specs/hybrid-search-bm25/requirements.md`:

```markdown
## Traceability

| REQ | Design Section | Plan Task | Test / Verification |
|-----|---------------|-----------|---------------------|
| REQ-001 | §3 Indexer Changes | Task 7 | Task 9 Step 1 |
| REQ-002 | §2 Search Engine Changes | Task 4 | Task 9 Steps 2-4 |
| REQ-003 | §1 New Types | Task 1 | test search_mode_default_is_hybrid |
| REQ-004 | §2 Per-backend branching | Task 4 | Task 9 Step 4 |
| REQ-005 | §4 MCP Server Changes | Task 6 | Task 9 Step 5 |
| REQ-006 | §5 CLI Changes | Task 5 | Task 9 Steps 2-6 |
| REQ-007 | §3 Multi-profile FTS | Task 7 | Task 9 Step 1 |
| REQ-008 | §2 Hybrid fallback | Task 4 | Manual (disconnect Ollama) |
| REQ-009 | §1 SearchResult, §4 MCP | Tasks 2, 6 | Task 9 Step 5 |
```

- [ ] **Step 2: Update CLAUDE.md architecture section**

In `CLAUDE.md`, update the "Data Flow" section to reflect hybrid search:

Add after the existing data flow:

```
SearchEngine ← LanceDB ← MCP Server / CLI
     ↑              ↑
Ollama embed    FTS index (BM25)
(semantic)      (keyword)
     └──── hybrid mode: both + RRF fusion ────┘
```

Update the SearchEngine description in "Key Types & Boundaries":

```
- **`SearchEngine`** (`query.rs`): Wraps LanceDB table. Supports three search modes: `hybrid` (BM25 + vector via LanceDB native RRF), `semantic` (vector-only, L2 distance → similarity via `exp(-distance)`), `keyword` (BM25 full-text only). Defaults to hybrid. Filters by `similarity_threshold` in semantic mode only.
```

- [ ] **Step 3: Commit**

```bash
git add specs/hybrid-search-bm25/requirements.md CLAUDE.md
git commit -m "docs: update traceability and architecture docs for hybrid search"
```
