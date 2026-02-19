# CoLibri Architecture Execution Plan

## 1. Outcome and Scope

Build CoLibri as a three-plane system that can:

1. Ingest content from many static and changing sources.
2. Convert all content into a canonical markdown representation.
3. Index and serve that corpus reliably through MCP, CLI, and future APIs.

Success criteria:
- New source integrations can be added as plugins without core rewrites.
- Re-indexing for new models/chunking strategies is safe and repeatable.
- Data is portable: one defined root can be backed up/restored or copied to another machine.

## 2. Architecture Decisions

## 2.1 Three Planes (Hard Separation)

1. Ingestion Plane (source adapters + conversion)
2. Indexing Plane (normalize/chunk/embed/upsert)
3. Serving Plane (retrieval/ranking interfaces: MCP/CLI/API)

Each plane has strict contracts and independent versioning.

## 2.2 Language Strategy (Challenged and Decided)

Decision:
- Keep Rust for index/serve core.
- Add Python plugin runtime + SDK for ingestion plugins.

Rationale:
- Source connectors and converters (Docling, MarkItDown, python-pptx, many SaaS SDKs) are ecosystem-heavy in Python.
- Rust remains ideal for stable binaries, runtime efficiency, and serving/indexing performance.
- A Rust-only ingestion strategy will likely slow connector velocity.

Implementation pattern:
- Rust host orchestrates jobs and lifecycle.
- Python plugins run out-of-process via a versioned contract (JSONL over stdio for v1).
- Optional future runtimes (WASM, gRPC sidecars) can be added later without breaking the core model.

## 2.3 Technology Re-Selection (Greenfield Rule)

Because storage compatibility is intentionally broken, all major infra choices must be re-evaluated:
- Vector store
- Embedding/runtime serving
- Reranking stack
- Job orchestration

Decision gate:
- No legacy tool is carried forward by default.
- Every selected component must win a measurable benchmark on quality, latency, operational burden, and portability.

Shortlist to evaluate:
- Vector store: Qdrant, pgvector, LanceDB
- Embedding serving: vLLM, HuggingFace TEI, Ollama
- Managed embedding API option: OpenAI-compatible provider (for non-airgapped deployments)

Baseline recommendation:
- Keep interfaces provider-agnostic (`VectorStore`, `Embedder`, `Reranker` traits) and select implementation by config.

## 2.4 Embedding Deployment Modes and Safety Routing

Both local and cloud embedding are first-class deployment modes.

Policy:
- Sensitive data classes must use local embedding runtime only.
- Lower-risk classes may use cloud embedding.
- Selection is enforced by a routing policy, not operator discretion at query time.

Operating model:
- Define embedding profiles (for example: `local_secure`, `cloud_fast`).
- Route documents/sources to a profile using data classification tags.
- Store `embedding_profile_id` on indexed artifacts for traceability.

Critical rule:
- Never mix vectors from different embedding profiles/models in the same index generation.
- Maintain separate generations (or collections) per embedding profile and cut over independently.

Example policy mapping:
- `restricted` -> `local_secure`
- `confidential` -> `local_secure`
- `internal` -> `cloud_fast`
- `public` -> `cloud_fast`

## 3. Contracts and Data Model

## 3.1 Canonical Document Envelope (v1)

All ingestion plugins must emit this shape:

```json
{
  "schema_version": 1,
  "source": {
    "plugin_id": "confluence",
    "connector_instance": "team-wiki-prod",
    "external_id": "SPACE:12345",
    "uri": "https://example.atlassian.net/wiki/..."
  },
  "document": {
    "doc_id": "stable-uuid-or-deterministic-hash",
    "title": "Architecture Decision",
    "markdown": "# ...",
    "content_hash": "sha256:...",
    "source_updated_at": "2026-02-18T07:00:00Z",
    "deleted": false
  },
  "metadata": {
    "doc_type": "wiki",
    "tags": ["architecture"],
    "language": "en",
    "acl_tags": ["team:platform"]
  }
}
```

Rules:
- `doc_id` must be stable across sync runs.
- `content_hash` is computed after conversion to markdown.
- Deletions are represented as tombstones (`deleted=true`) and must be preserved.

## 3.2 Plugin Manifest (v1)

Each plugin ships a manifest with:
- `plugin_id`, `version`, `runtime`, `entrypoint`
- capabilities: `snapshot`, `incremental`, `webhook`
- auth modes: oauth/token/basic/service-account
- cursor strategy and retry policy
- config schema for validation

## 3.3 Canonical Storage Model

Use a managed, relocatable application store:

```text
$COLIBRI_HOME/
  manifest.json
  metadata.db
  canonical/
  indexes/
  state/
  backups/
  logs/
```

Default: `~/.local/share/colibri`  
Override: `COLIBRI_HOME=/path/to/portable-root`

Recommended table set in `metadata.db` (SQLite first):
- `schema_versions(component, version, applied_at)`
- `sources(source_id, plugin_id, config_json, enabled, created_at, updated_at)`
- `sync_state(source_id, cursor, last_success_at, last_error_at, error_json)`
- `documents(doc_id, source_id, external_id, title, content_hash, source_updated_at, deleted, acl_json, classification, meta_json, updated_at)`
- `document_blobs(doc_id, markdown_path, size_bytes, checksum)`
- `embedding_profiles(profile_id, provider, model, endpoint, locality, config_json, created_at, updated_at)`
- `routing_policies(policy_id, rules_json, is_active, updated_at)`
- `index_generations(generation_id, embedding_profile_id, pipeline_version_json, status, created_at, activated_at)`
- `document_index_state(doc_id, generation_id, status, chunk_count, embedded_at)`
- `migration_log(component, from_version, to_version, applied_at, success, notes)`

Markdown body storage:
- Store markdown in `canonical/<source_id>/<doc_id>.md`
- Keep paths relative in DB to maintain portability.

## 4. Versioning and Migration Strategy

## 4.1 Independent Version Axes

1. Canonical schema version: envelope + metadata store shape.
2. Pipeline version: chunking/preprocessing/embedding/reranking settings.
3. Serving API version: MCP/CLI/API contract compatibility.

## 4.2 Migration Rules

- Forward-only migrations.
- Every migration is idempotent.
- `colibri migrate --dry-run` required in CI for migration PRs.
- `colibri migrate` creates pre-migration backup in `backups/`.
- Startup behavior: refuse to run on unsupported downgrade states.

## 4.3 Index Generation Model

Vector indexes are immutable generations:
- Example: `gen_2026_02_18_bge-m3_chunk3000_v2`
- A pointer in `manifest.json` marks `active_generation`.

Cutover flow:
1. Build new generation in shadow.
2. Run quality checks against evaluation set.
3. Flip `active_generation` atomically.
4. Keep N previous generations for rollback.

Re-index triggers:
- Required: embedding model or chunking/preprocessing change.
- Optional/none: serving adapter-only changes.

## 5. Runtime Flows

## 5.1 Ingestion Job

1. Resolve enabled sources.
2. Invoke plugins with source config + prior cursor.
3. Validate envelope schema.
4. Upsert canonical metadata + write markdown blob.
5. Persist cursor and sync metrics.

## 5.2 Indexing Job

1. Detect changed docs by `content_hash` or tombstone.
2. Resolve `embedding_profile_id` via active routing policy and document classification.
3. Chunk + embed using selected pipeline version for that profile.
4. Upsert/remove vector entries in target generation.
5. Mark per-document index state.

## 5.3 Serving Query

1. Read `active_generation`.
2. Embed query using the embedding profile bound to that generation.
3. Retrieve/filter/rank.
4. Return through MCP or CLI adapters.

## 6. Security and Isolation Baseline

- Plugins run in isolated subprocesses with:
  - execution timeout
  - memory budget
  - structured stderr capture
- Secrets never persisted in plugin logs.
- Credentials stored in OS keychain or encrypted config references.
- ACL tags carried from ingestion through retrieval filters.

## 7. Execution Roadmap

## Phase 0 (1 week): Foundations

Deliverables:
- Architecture RFC accepted.
- Tooling bake-off design accepted (datasets, metrics, thresholds, candidate stacks).
- `COLIBRI_HOME` root layout implemented.
- `metadata.db` bootstrap + `schema_versions` table.
- `manifest.json` with active generation pointer.

Acceptance:
- Fresh bootstrap creates deterministic directory structure.
- Move/copy `COLIBRI_HOME` to another machine preserves startup viability.

## Phase 1 (2 weeks): Canonical Store + Migrations

Deliverables:
- Canonical envelope validator.
- Document persistence (`documents`, `document_blobs`, tombstones).
- Migration framework + CLI commands.

Acceptance:
- `colibri migrate --dry-run` reports pending/applied state.
- Re-running migration is no-op.

## Phase 1.5 (1 week): Tooling Bake-Off and Selection

Deliverables:
- Representative evaluation corpus and query set.
- Scored comparison of shortlisted vector stores and embedding runtimes.
- ADRs documenting selected defaults and rejected alternatives.
- Embedding policy design approved (classification taxonomy + routing rules).

Acceptance:
- Selected stack meets defined quality and latency targets.
- Operational runbook exists for local and production-like setup.
- Safety tests prove restricted/confidential classes cannot be routed to cloud profiles.

## Phase 2 (2-3 weeks): Plugin Runtime (Python v1)

Deliverables:
- Plugin manifest spec and loader.
- JSONL stdio protocol host<->plugin.
- Python SDK + example plugins:
  - `filesystem_markdown`
  - one SaaS connector (recommend Confluence or Jira)

Acceptance:
- Plugin errors are isolated and do not crash host.
- Incremental sync with cursor works end-to-end.

## Phase 3 (2 weeks): Index Generation Management

Deliverables:
- Generation creation, shadow build, activation, rollback.
- `index_generations` and `document_index_state` wiring.

Acceptance:
- Two generations can coexist.
- Atomic switch of `active_generation` is observable by CLI/MCP queries.

## Phase 4 (1-2 weeks): Serving Alignment

Deliverables:
- Query path routes by active generation config.
- Shared retrieval core for MCP + CLI.

Acceptance:
- No model mismatch between query embedding and active index.
- Existing CLI and MCP behavior remains backward compatible.

## Phase 5 (ongoing): Quality + Scale

Deliverables:
- Evaluation dataset and retrieval metrics.
- Plugin catalog growth (GitLab, Confluence, shared folders, PPTX, Jira, Zephyr).

Acceptance:
- Promotion gate for new generation based on retrieval KPIs.

## 8. Immediate Backlog (Execution-Ready)

Priority 1:
- Add `COLIBRI_HOME` root resolver + portability tests.
- Add metadata DB bootstrap/migration module.
- Introduce `CanonicalDocument` and validator crate/module.

Priority 2:
- Implement `index generation` abstraction in indexer/query paths.
- Add `active_generation` read/write in manifest management.
- Add embedding profile/routing policy resolution in indexing pipeline.

Priority 3:
- Create `plugins/` workspace:
  - `plugins/spec/` (manifest + envelope JSON schemas)
  - `plugins/python-sdk/`
  - `plugins/examples/filesystem_markdown/`

Priority 4:
- Add CLI commands:
  - `colibri migrate --dry-run`
  - `colibri migrate`
  - `colibri index --generation <id>`
  - `colibri index activate <generation-id>`

## 9. Definition of Done for “Execution Prepared”

This plan is execution-ready when:
- All core contracts are versioned and documented.
- Migration/cutover commands exist and are tested.
- At least two real plugins run through the canonical->index->serve path.
- Portable backup/restore workflow is verified on a second machine.

## 10. Recommended First Sprint Board

1. RFC: Three-plane architecture + polyglot rationale.
2. Implement `COLIBRI_HOME` root and bootstrapping.
3. Add migration engine skeleton and `schema_versions`.
4. Define envelope/manifest JSON schemas.
5. Refactor indexer to consume canonical store (not direct source reads).
6. Add generation pointer and shadow index wiring.
7. Create Python SDK skeleton + filesystem plugin.

## 11. Ready-to-Start Checklist

Start implementation only after these are confirmed:
- Finalized data classification taxonomy (`restricted/confidential/internal/public`).
- Approved embedding profiles (`local_secure`, `cloud_fast`) and default models.
- Agreed benchmark KPIs for tooling bake-off (quality, p95 latency, ops complexity, portability).
- Chosen initial source connectors for v1 (`filesystem_markdown` + one SaaS connector).
- Ownership assigned for Phase 0 and Phase 1.5 deliverables.
