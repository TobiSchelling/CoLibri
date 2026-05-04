# Frontmatter-Aware Search Design

> **Feature:** frontmatter-aware-search
> **Companion to:** [requirements.md](requirements.md), [plan.md](plan.md)

## Architecture

Three layers touched, in order:

```
┌─────────────────────────────────────────────────────────────┐
│  SERVING (CLI + MCP)                                        │
│  - Tool schemas accept new filter inputs                     │
│  - SearchEngine::search() takes SearchFilter, group_by_doc   │
├─────────────────────────────────────────────────────────────┤
│  QUERY                                                       │
│  - Post-retrieval filter application (path/frontmatter/since)│
│  - Doc-level grouping (dedup by doc_id, keep best chunk)     │
│  - Joins with metadata_store at result time                  │
├─────────────────────────────────────────────────────────────┤
│  STORAGE                                                     │
│  - metadata_store: new column `frontmatter_json TEXT`        │
│  - LanceDB chunks schema UNCHANGED — no reindex needed       │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                   │
│  - Filesystem connector parses YAML frontmatter              │
│  - Strips frontmatter from markdown before chunking          │
│  - Populates EnvelopeMetadata.tags + new `frontmatter` map   │
└─────────────────────────────────────────────────────────────┘
```

The key non-obvious decision: **filtering is applied at query time via metadata_store join, not by adding columns to the LanceDB chunks table.** This avoids a full reindex of the existing 181k chunks. Per-search overhead is bounded by `top_k_per_backend × num_backends` doc lookups, which is ≤ 500 by default.

## Data model changes

### `EnvelopeMetadata` (new field)

```rust
pub struct EnvelopeMetadata {
    pub doc_type: String,
    pub classification: String,
    pub tags: Option<Vec<String>>,
    pub language: Option<String>,
    pub acl_tags: Option<Vec<String>>,
    pub frontmatter: Option<serde_json::Map<String, serde_json::Value>>, // NEW
}
```

The `frontmatter` field carries the parsed YAML mapping from the source file. Preserves type fidelity (string, number, list, nested map).

### `metadata_store` document row

Add column:

```sql
ALTER TABLE documents ADD COLUMN frontmatter_json TEXT NOT NULL DEFAULT '{}';
```

Persisted as a JSON object string. Query-time deserialized to `serde_json::Value` for filter matching.

### Migration mechanism

Re-uses the existing `migration_log` table pattern (visible in earlier schema bumps). New migration ID `0007_add_frontmatter_json` (verify next sequence number when implementing). Idempotent: checks `migration_log` for prior application before applying.

### LanceDB chunks schema

**Unchanged.** Stays `(doc_id, text, vector)` — the doc_id is the join key.

## API changes

### `SearchEngine::search` signature evolution

Today:

```rust
pub async fn search(
    &self,
    query: &str,
    classification: Option<&str>,
    doc_type: Option<&str>,
    limit: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>, ColibriError>
```

After:

```rust
pub async fn search(
    &self,
    query: &str,
    filter: SearchFilter,
    group_by_doc: bool,
    limit: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>, ColibriError>
```

`SearchFilter` absorbs the existing `classification` and `doc_type` to keep the call site clean:

```rust
#[derive(Debug, Default, Clone)]
pub struct SearchFilter {
    pub classification: Option<String>,
    pub doc_type: Option<String>,
    pub path_includes: Vec<String>,    // empty = no filter
    pub path_excludes: Vec<String>,
    pub frontmatter: BTreeMap<String, String>,  // empty = no filter
    pub since: Option<DateTime<Utc>>,
}
```

`Default::default()` yields the no-filter case — preserving the cheap-call ergonomics today's API has.

### `SearchResult` (new optional fields)

```rust
pub struct SearchResult {
    // existing
    pub doc_id: String,
    pub title: String,
    pub markdown_path: String,
    pub classification: String,
    pub doc_type: String,
    pub score: f32,
    pub text: String,
    pub search_mode: SearchMode,
    // new (populated only when group_by_doc = true)
    pub chunk_count: Option<usize>,
    pub frontmatter: Option<serde_json::Map<String, serde_json::Value>>,
}
```

Backwards compatibility: serializes `chunk_count` and `frontmatter` only when populated (use `#[serde(skip_serializing_if = "Option::is_none")]`).

## Query flow

```
search(query, filter, group_by_doc, limit, mode)
    │
    ├─ For each backend (one per embedding profile):
    │       ├─ LanceDB hybrid/vector/keyword search → top per_backend_limit raw hits
    │       └─ Push hits into merged: Vec<SearchHit>
    │
    ├─ Apply filter (filter_hits):
    │       ├─ Look up metadata_store rows for unique doc_ids in `merged`
    │       ├─ Drop hits whose doc fails any filter predicate
    │       │   • classification mismatch
    │       │   • doc_type mismatch
    │       │   • path_includes (no substring match) → drop
    │       │   • path_excludes (any substring match) → drop
    │       │   • frontmatter[k] != v → drop
    │       │   • source_updated_at < since → drop
    │       └─ Returns filtered: Vec<SearchHit>
    │
    ├─ If group_by_doc:
    │       ├─ Bucket filtered by doc_id, count chunks per bucket.
    │       ├─ Keep best-scoring hit per bucket.
    │       ├─ Attach chunk_count and frontmatter.
    │       └─ Returns grouped: Vec<SearchHit>
    │
    ├─ Sort by score desc.
    └─ Truncate to `limit`. Return.
```

## YAML frontmatter parser (filesystem connector)

Use `serde_yaml` (already in Cargo deps tree, used elsewhere). Algorithm:

```rust
fn parse_frontmatter(text: &str) -> (Option<Vec<String>>, Option<serde_json::Map<String, serde_json::Value>>, &str) {
    // Returns (tags, frontmatter, body)
    if !text.starts_with("---\n") { return (None, None, text); }
    let after_marker = &text[4..];
    let Some(end_idx) = after_marker.find("\n---\n") else { return (None, None, text); };
    let yaml_block = &after_marker[..end_idx];
    let body = &after_marker[end_idx + 5..];

    let parsed: serde_yaml::Value = match serde_yaml::from_str(yaml_block) {
        Ok(v) => v,
        Err(e) => { warn!("frontmatter parse error: {e}"); return (None, None, text); }
    };

    // Extract tags
    let tags = parsed.get("tags")
        .and_then(|t| t.as_sequence())
        .map(|seq| seq.iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.trim().trim_start_matches('#').to_string())
            .filter(|s| !s.is_empty())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>());

    // Convert top-level mapping to JSON for portability
    let frontmatter = serde_json::to_value(&parsed)
        .ok()
        .and_then(|v| v.as_object().cloned());

    (tags, frontmatter, body)
}
```

Edge cases handled by REQ-001 acceptance criteria:
- File not starting with `---` → return body unchanged.
- Missing closing `---` → log warn, return whole file as body, no metadata.
- Invalid YAML → log warn, body unchanged, no metadata.
- `tags` as scalar string instead of list → tolerated (treated as None).

## MCP schema additions

`search_library` input schema gains:

```json
{
  "path_includes": {
    "type": "array", "items": {"type": "string"},
    "description": "Only return docs whose path contains any of these substrings."
  },
  "path_excludes": { "type": "array", "items": {"type": "string"} },
  "frontmatter": {
    "type": "object", "additionalProperties": {"type": "string"},
    "description": "Equality match on frontmatter fields. e.g. {\"area\":\"SIT\",\"status\":\"active\"}."
  },
  "since": {
    "type": "string", "format": "date-time",
    "description": "Only docs updated on or after this RFC 3339 timestamp."
  },
  "group_by_doc": {
    "type": "boolean", "default": true,
    "description": "Return one result per document (best chunk + chunk_count). Default true; set false for chunk-level results."
  }
}
```

Same shape applied to `search_books`.

## Tests

| File | What it covers |
|---|---|
| `src/connectors/filesystem.rs` (test mod) | Frontmatter parsing: valid, missing, malformed, scalar-tags, body-only |
| `src/metadata_store.rs` (test mod) | Migration idempotent; round-trip frontmatter_json |
| `src/query.rs` (test mod) | Filter application: each filter type; combined; group_by_doc dedup |
| `src/mcp.rs` (test mod) | Tool input parsing; type validation errors |

A new integration-style test under `tests/` exercises end-to-end: synthetic vault → sync → search with filter → assert hit set.

## Risks / open questions

1. **Schema migration on existing DB**: needs careful testing on a real CoLibri home. Migration log must be checked first; idempotency is REQ-002 AC. Plan to test against `~/.local/share/colibri/metadata.db` before bumping version.

2. **Frontmatter that contains arbitrary nested YAML**: storing as opaque JSON works for storage; equality filter is string-based. Future: deeper queries (Dataview-style) require typed schema. Out of scope here.

3. **`browse_topics` re-population**: existing indexed docs from before this feature have `tags_json: '[]'`. After upgrade, browse_topics returns nothing useful until next `colibri sync`. Documented in plan.md step 11.

4. **Hybrid search rank fusion behavior**: filter happens post-fusion. If filter drops 90% of top-k_per_backend hits, ranks beyond that aren't considered. Mitigation: existing `per_backend_limit = top_k.saturating_mul(5).max(limit*5).min(500)` already over-fetches by 5x; that buffer absorbs typical filter selectivity. Document this assumption in the design — extreme filter selectivity may need a different approach (re-query with a higher k, or push filters into LanceDB which would need schema changes — back to A1's "no reindex" non-goal).
