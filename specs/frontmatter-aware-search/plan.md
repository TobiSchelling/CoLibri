# Frontmatter-Aware Search Plan

> **Feature:** frontmatter-aware-search
> **Companion to:** [requirements.md](requirements.md), [design.md](design.md)

Implementation sequence. Each step lands in a single commit; tests run between commits.

## Step 1 — Envelope: add `frontmatter` field

- `src/envelope.rs`: add `frontmatter: Option<serde_json::Map<String, serde_json::Value>>` to `EnvelopeMetadata`.
- Update existing `sample_envelope()` test helper. Set new field to `None` for backward-compat tests.

**Verifies:** `make check && make test`.

## Step 2 — Filesystem connector: parse YAML frontmatter, strip from body

- `src/connectors/filesystem.rs`:
  - Add `parse_frontmatter(text) -> (tags, frontmatter, body)` per design.md.
  - In `sync()` for `.md`/`.markdown`, parse first, then chunk only the body.
  - Populate `EnvelopeMetadata.tags` and `.frontmatter`.
- Tests: 4 cases (valid, no-frontmatter, malformed, scalar-tags).

**Verifies:** `make check && make test`. Manual: `colibri sync` against a small vault yields envelopes with frontmatter populated.

## Step 3 — Metadata store: add `frontmatter_json` column + migration

- `src/metadata_store.rs`:
  - Add migration with next ID after current max.
  - Extend `DocumentRow` with `frontmatter_json: String`.
  - Update `upsert_document` SQL: write the column.
  - Update `list_documents` and `get_documents` to return it.
- Tests: migration idempotent; round-trip values.

**Verifies:** `make check && make test`.

## Step 4 — Indexer: thread frontmatter through to metadata store

- `src/indexer.rs`: serialize `envelope.metadata.frontmatter` (or `None` → `'{}'`) and pass to `upsert_document`.

**Verifies:** `make check && make test`. Manual: sync a vault, query SQLite — `frontmatter_json` non-empty for tagged files.

## Step 5 — Search filter struct and application

- `src/query.rs`:
  - Introduce `SearchFilter` struct.
  - Change `SearchEngine::search` signature.
  - Add `MetadataStore::get_documents_by_ids(&[String]) -> Result<HashMap<String, DocumentRow>>` if not present.
  - Apply filters post-LanceDB-retrieval, pre-rank-merge.
  - Update all in-tree call sites (CLI search, MCP, tests).
- Tests: each filter type, combined, edge cases.

**Verifies:** `make check && make test`.

## Step 6 — Doc-level grouping

- `src/query.rs`: add `group_by_doc: bool` parameter; implement bucketing + chunk_count + frontmatter attachment.
- Extend `SearchResult` with optional `chunk_count` and `frontmatter` fields (with `#[serde(skip_serializing_if = "Option::is_none")]`).
- Tests: multi-chunk-per-doc fixtures.

**Verifies:** `make check && make test`.

## Step 7 — CLI: expose filters

- CLI search command flags: `--path-includes`, `--path-excludes`, `--frontmatter KEY=VALUE` (repeatable), `--since`, `--group-by-doc`.
- Help text per flag.
- Default `group_by_doc = false` for CLI.

**Verifies:** `make check && make test`. Manual: end-to-end `colibri search "..." --frontmatter area=SIT --group-by-doc`.

## Step 8 — MCP: expose filters with new defaults

- `src/mcp.rs`:
  - Update `handle_tools_list` schema for `search_library` and `search_books` (new properties).
  - Update `handle_tools_call` to parse the new fields and pass to `search()`.
  - Default `group_by_doc = true` for MCP callers.
  - Validate input types; return JSON-RPC errors for malformed input.
- Tests: tool-call dispatch with each filter; error cases.

**Verifies:** `make check && make test`. Manual: `colibri serve --check`, then a probe via `mcp__colibri__search_library`.

## Step 9 — `browse_topics` regression test

- Add an end-to-end test asserting `browse_topics()` returns ≥ 1 tag after sync of a fixture with tagged frontmatter.

**Verifies:** `make check && make test`.

## Step 10 — Bump version, docs, formula

- `Cargo.toml` + `Cargo.lock`: 0.13.0 → 0.14.0.
- `README.md`: brief mention of new filter capabilities under Commands section.
- ADR or changelog entry: `docs/adr/<n>-frontmatter-aware-search.md` recording the decision.
- Homebrew tap formula update: deferred until after GitHub release tag is cut (resource SHA depends on tarball).

**Verifies:** `make check && make test && make lint && make format`.

## Step 11 — Local rollout

1. Build: `cargo build --release`.
2. Stop running `colibri serve` if any.
3. Replace local binary: `cp target/release/colibri /opt/homebrew/bin/colibri` (or `make install` if a target exists).
4. Run `colibri migrate` to apply the new schema migration.
5. Run `colibri sync && colibri index` to re-ingest with frontmatter populated. Wait for completion (10-30 min).
6. Verify CLI: `colibri search "Aegis" --group-by-doc --frontmatter area=SIT`.
7. Verify MCP from this Claude Code session.
8. Update homebrew tap (separate repo) once a GitHub release tag is cut.

## Verification gates (run between every step)

```
make check    # cargo check
make test     # cargo test
make lint     # cargo clippy --all-targets --all-features -- -D warnings
make format   # cargo fmt --check
```

Don't proceed to step N+1 until step N passes all four. Public-API changes (steps 1, 5, 6) ripple to call sites — fix them in the same step.

## Estimated effort

| Steps | What | Effort |
|---|---|---|
| 1–4 | Data model + storage | 3-4h |
| 5–6 | Search engine | 3-4h |
| 7 | CLI | 1h |
| 8 | MCP | 1-2h |
| 9 | Regression test | 30min |
| 10–11 | Release + rollout | 1-2h |
| **Total** | | **9-13h** |

## Out-of-scope

- Range / numeric / boolean-operator filters.
- Reranker awareness of filters.
- Frontmatter schema validation against external registries (e.g. PKM `09_SYSTEM/STATE_MACHINES/`).
- Performance optimization for huge filter result sets (>10k matched chunks pre-filter).
- LanceDB chunk-schema changes.
