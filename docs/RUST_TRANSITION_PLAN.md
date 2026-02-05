# Rust Transition Plan (Rough)

This is a pragmatic, staged plan to transition CoLibri from Python to Rust while minimizing deployment friction and keeping the product usable throughout the migration. It emphasizes incremental milestones, clear decision points, and a path to a Homebrew-friendly single-binary distribution.

## Goals

1. Produce a single-binary CLI that is easy to ship via Homebrew.
2. Preserve core user workflows (import → index → search → serve API/MCP).
3. Keep data format stability or provide explicit migration tooling.
4. Reduce deployment complexity (Python dependency packaging, sdists, etc.).

## Non-Goals (for v1 of Rust port)

- Rewriting every optional integration immediately (e.g., niche importers).
- Perfect feature parity on day one.
- Breaking existing library/index formats without a migration path.

## Strategy Overview

Two viable tracks. We can start with Track A and switch to Track B if import complexity blocks progress.

**Track A: Full Rust Port (Preferred long-term)**
- Rust CLI + indexer + query engine + API/MCP.
- Native embedding client (Ollama HTTP).
- Native import pipeline for PDF/EPUB.

**Track B: Hybrid (Fastest path)**
- Rust CLI + query + API/MCP.
- Keep Python import pipeline as a sidecar (local process or minimal service).
- Later replace Python import with Rust when ready.

## Phase 0 — Alignment (1–2 days)

- Confirm “must-have” features for first Rust release.
- Decide target distribution (single binary) and supported platforms.
- Identify any Python-only blockers (e.g., PDF extraction quality).

Deliverables:
- Finalized scope for v0.3 (Rust MVP) with prioritized feature list.

## Phase 1 — Inventory & Test Harness (2–4 days)

- Inventory current CLI commands and behaviors.
- Add a small golden-test suite for CLI behaviors (inputs → outputs).
- Capture minimal datasets to validate index/search parity.

Deliverables:
- CLI behavior checklist.
- Golden test dataset and expected outputs.

## Phase 2 — Target Architecture (3–5 days)

Define a Rust architecture that mirrors current components but improves packaging:

- CLI: `clap`
- Config: `serde` + `toml` or `yaml`
- HTTP API: `axum` or `actix-web`
- MCP: simple stdio protocol implementation
- Storage/index: evaluate Rust-native vector store options (see “Decisions”)
- Embeddings: Ollama HTTP client (`reqwest`)

Deliverables:
- `docs/ARCHITECTURE.md` update or new diagram.
- Decision record for storage/index choice.

## Phase 3 — Rust CLI Skeleton (3–5 days)

- Create `colibri-rs` crate with subcommands:
  - `doctor`, `import`, `index`, `search`, `serve`, `capabilities`
- Wire config loading and filesystem paths.
- Add structured logging and error reporting.

Deliverables:
- CLI shell with stable command surface.
- Config layout compatible with current defaults.

## Phase 4 — Embeddings & Query Core (5–10 days)

- Implement Ollama embedding client.
- Implement query engine (vector search + metadata filtering).
- Provide JSON output flag parity with current CLI.

Deliverables:
- `search` works end-to-end with a stubbed index.
- Baseline performance and latency checks.

## Phase 5 — Index & Storage (5–15 days)

- Choose and integrate storage:
  - Option 1: Rust-native vector DB
  - Option 2: LanceDB via Rust bindings (if viable)
  - Option 3: Embedded DB + vector extension (e.g., SQLite + vector)
- Implement index metadata/versioning and manifests.
- Provide migration layer or reindex strategy.

Deliverables:
- Index creation + read/query.
- Index metadata format and versioned migration plan.

## Phase 6 — Import Pipeline (5–20 days)

**If Track A (full Rust):**
- PDF extraction (evaluate Rust PDF libraries, quality vs. speed).
- EPUB extraction (Rust HTML parsing + markdown conversion).

**If Track B (hybrid):**
- Keep current Python import pipeline as a separate process.
- Rust CLI calls Python importer and consumes extracted markdown.

Deliverables:
- `import` produces same Markdown layout as current system (or documented changes).

## Phase 7 — API & MCP (3–7 days)

- Port REST API endpoints.
- Port MCP stdio server behaviors.
- Add feature flags to allow Python/legacy fallback if needed.

Deliverables:
- API parity for core operations.
- MCP capabilities matched to current surface.

## Phase 8 — Packaging & Release (3–5 days)

- Build single-binary releases for macOS and Linux.
- Homebrew formula for binary install (no Python resources).
- Signed release artifacts + checksums.

Deliverables:
- Homebrew formula for binary.
- Release checklist and automated CI build.

## Phase 9 — Migration & Deprecation (2–4 days)

- Migration notes for configs and indices.
- Optional `colibri migrate` command to reindex or convert.
- Deprecation timeline for Python package.

Deliverables:
- Clear migration guide.
- Deprecation notice and support policy.

## Key Decisions (To Resolve Early)

1. **Vector store**: Rust-native vs. LanceDB Rust vs. embedded DB.
2. **Import pipeline**: full Rust vs. hybrid Python.
3. **Compatibility**: reindex vs. migrate existing LanceDB data.
4. **CLI parity**: strict vs. best-effort in v1.

## Risk Notes

- PDF extraction quality may be the hardest Rust parity gap.
- Index migration from LanceDB might be non-trivial.
- Rebuilding feature parity will take time; keep scope tight.

## Suggested First Milestone (2–3 weeks)

“Rust MVP”
- `colibri doctor`, `colibri search`, and `colibri serve`
- Uses Ollama embeddings
- Can index a small Markdown corpus
- Produces a single binary that installs cleanly via Homebrew
