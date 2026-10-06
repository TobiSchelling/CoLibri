# Implementation Plan: P3 (mirrors, `update`, `status`)

> Spec: `requirements.md` REQ-009..REQ-016, plus AC-006.2 carried over from P1+P2.
> Branch: `feat/ffi-p1-storage`. Written alongside the implementation; records the decisions taken.

## Decisions

- **Mirrors replace the filesystem connector now**, not in P5. Keeping both would mean two folder-ingestion paths with separate tests until P5. `type: filesystem` in `connectors:` now fails with a pointer to `mirrors:`. `colibri sync` remains for the Zephyr connector only.
- **Plan, then apply.** Planning is read-only: text formats (`.md`, `.markdown`, `.txt`, `.yaml`, `.yml`) are read and hashed; convertible formats (`.pdf`, `.epub`, `.docx`, `.pptx`) take a size+mtime fast path, then SHA-256. Conversion happens only in apply, through a cache keyed by SHA-256 (`conversions` table + `<home>/conversions/`).
- **What counts as seen** (never pruned): every file matched by include/exclude, including online-only placeholders, unreadable files, unsupported types and failed conversions.
- **Prune guard:** no prune when the root is missing (run status `error`), when the walk was incomplete, or when deletions exceed `max(min_count, max_fraction × active)` without `--allow-mass-prune`. Blocked prunes are recorded as `prune_blocked` problems.
- **Problems** are replaced per collection on every run, so `status` always shows the latest state.
- **Exit codes:** `update` exits 1 when a mirror could not be processed (`error`) or indexing had errors; problems alone (`partial`) exit 0.
- **Tool checks** (docling/pandoc/markitdown) shared by `doctor` and `bootstrap` via `cli::missing_tools`.

### After code review

- Content changes and metadata-only changes are separate: unchanged content with a new location, size/mtime or `doc_type` is a metadata update (no re-embed, no reconversion). Fixes stale `source_path` after a folder move and endless re-hashing after a `doc_type` change.
- The conversion cache is content-addressed and written atomically; a cached file is used even if its row was rolled back.
- Pruned canonical files are deleted only after the final commit. A pruned file that still exists is recorded with reason `excluded`.
- Prune is also blocked when a mirror that had documents suddenly reads as empty (unmounted or signed-out cloud folder).
- Mirror paths must be absolute (or `~/...`); matching is case-insensitive; directory symlinks are followed with loop protection; broken symlinks are notices, not walk failures; only `/**` patterns exclude whole subtrees by probe.
- `status` counts chunks of any searchable document as live (same rule as the indexer's orphan purge).
- Known limitation: a file whose conversion fails is retried on every update (no negative cache); revisit with `--reconvert` in P4.

## Tasks

1. Move frontmatter, PlantUML and conversion code from `connectors/filesystem.rs` into `src/ingest/{frontmatter,plantuml,convert}.rs` with their tests; delete the filesystem connector.
2. `ingest/walk.rs`: include/exclude glob filter (`**/` also matches the root), excluded directories not descended, completeness flag, online-only detection (`SF_DATALESS`). Tests incl. unreadable directory and AC-016.1.
3. `ingest/convert.rs`: `Converter` trait, `ExternalConverter`, `file_sha256`, `convert_cached`; test that identical bytes convert once.
4. `metadata_store.rs`: `list_documents_in`, collection runs, problems, conversions.
5. `config.rs`: typed `mirrors:` and `prune:` with `deny_unknown_fields`, name/glob validation, `expand_tilde`.
6. `ingest/mirror.rs`: `reconcile_mirror` with tests for AC-009, AC-010, AC-011.1-3, AC-012.1, AC-013.1, YAML and unsupported types.
7. `ingest/update.rs` + `cli/update.rs`: mirrors then index; dry run read-only; update-level test (AC-009.1/.2, AC-010.1).
8. `cli/status.rs`: per-collection counts, last run, problems, orphan chunks (AC-006.2, AC-014.1).
9. `doctor` lists mirrors; `bootstrap --init-path` writes a mirror; `reset` also removes `conversions/`.
10. Integration tests: `tests/cli_storage.rs` switched to mirrors + `update`; new `tests/cli_mirrors.rs` (dry run, frontmatter/YAML search, prune, status).
11. README/CLAUDE.md; code review; `/verify` for REQ-009..016 and AC-006.2.
