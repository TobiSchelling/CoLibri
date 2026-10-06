# Files-First Ingestion Verification Report

> **Feature:** files-first-ingestion
> **Date:** 2026-10-06 (P1+P2), 2026-10-06 (P3)
> **Spec:** specs/files-first-ingestion/requirements.md
> **Verified by:** Claude (claude-opus-5-5)
> **Status:** PASSED for P1+P2 (REQ-001..008) and P3 (REQ-009..016, AC-006.2); REQ-017..031 not yet implemented

---

## Phase P1+P2: Acceptance Criteria Results

All evidence below was produced in this session on branch `feat/ffi-p1-storage` (uncommitted working tree before the phase commit). Unit tests were run individually with `cargo test --bin colibri <name>`, integration tests with `cargo test --test cli_storage`.

| REQ | Priority | Criterion | Method | Result | Evidence |
|-----|----------|-----------|--------|--------|----------|
| REQ-001 | MUST | AC-001.1: non-SQLite `metadata.db` stays byte-identical, write commands error | `metadata_store::tests::non_sqlite_file_is_rejected_and_untouched`, `config::tests::open_for_write_leaves_corrupt_metadata_db_untouched` | PASS | both `1 passed`; error contains `colibri reset`, bytes compared |
| REQ-001 | MUST | AC-001.2: other `user_version` / pre-v7 schema reported with `colibri reset`, file unchanged | `metadata_store::tests::old_schema_is_rejected_and_untouched`; `cli_storage::pre_v7_metadata_db_is_reported_and_left_unchanged` | PASS | unit `1 passed`; CLI `sync` fails, stderr contains `colibri reset`, file identical |
| REQ-002 | MUST | AC-002.1: search/serve paths create or modify no file in the data dir | `query::tests::read_paths_filter_by_collection_and_leave_data_dir_untouched` | PASS | `1 passed`; recursive size+mtime snapshot equal after `SearchEngine::new`, search, `list_books`, `browse_topics` |
| REQ-002 | MUST | AC-002.2: `colibri search` without data exits non-zero and creates nothing | `cli_storage::sync_search_doctor_reset_round_trip` (first step), `config::tests::load_config_creates_nothing`, `metadata_store::tests::open_ro_requires_existing_db_and_creates_nothing` | PASS | CLI search fails and the data dir does not exist afterwards; unit tests `1 passed` each |
| REQ-003 | MUST | AC-003.1: second writer fails within 1 s naming the holder PID | `lock::tests::second_writer_fails_fast_with_holder_pid` | PASS | `1 passed`; asserts elapsed < 1 s and `pid <n>` in message |
| REQ-003 | MUST | AC-003.2: read succeeds while a write transaction is open | `metadata_store::tests::read_succeeds_while_write_transaction_is_open` | PASS | `1 passed`; reader sees last committed state, then the new row after commit |
| REQ-004 | MUST | AC-004.1: reset removes `metadata.db`, `canonical/`, `index/`; config and mirror folder untouched | `cli::reset::tests::reset_removes_owned_data_only`, `cli_storage` reset step | PASS | unit `1 passed`; CLI: data gone, source file and config exist |
| REQ-004 | MUST | AC-004.2: no or wrong confirmation deletes nothing | `cli::reset::tests::confirmation_requires_exact_word`, `cli_storage` (`reset` with stdin `yes`) | PASS | unit `1 passed`; CLI prints `Aborted`, `metadata.db` still exists |
| REQ-005 | MUST | AC-005.1: failure on doc 3 of 5 keeps docs 1-2; rerun embeds 3-5 only | `indexer::tests::interrupted_run_resumes_with_remaining_documents`, plus `index_meta_is_written_before_embedding`, `missing_table_is_rebuilt_even_if_documents_claim_indexed` | PASS | `1 passed` each; rerun reports 3 indexed / 2 skipped and exactly 3 embed calls |
| REQ-006 | MUST | AC-006.1: chunks of unknown doc ids are purged, live chunks stay | `indexer::tests::orphan_chunks_are_purged` | PASS | `1 passed`; `orphans_removed == 1`, table ids == {keep} |
| REQ-006 | MUST | AC-006.2: `colibri status` shows `orphan chunks: 0` | — | SKIP | `colibri status` is delivered in P3 (REQ-014). The purge itself is proven by AC-006.1; the count is in `IndexResult.orphans_removed`. Re-verify in P3. |
| REQ-007 | MUST | AC-007.1: `--collection books` (CLI) and MCP `collection` return only that collection | `query::tests::filter_collection_and_doc_type`, `query::tests::read_paths_filter_by_collection_and_leave_data_dir_untouched`, `mcp::tests::filter_extras_parse_collection_and_reject_bad_types`, `cli_storage` (all results `collection == notes`) | PASS | all pass; e2e search with `collection: books` returns 1 of 2 docs |
| REQ-007 | MUST | AC-007.2: no `classification`, `routing_policy`, `embedding_profiles`, `active_generation` identifiers in `src/` | `grep -rnE "\bclassification\b\|routing_policy\|embedding_profiles\|active_generation" src` | PASS | grep exit 1 (no matches) |
| REQ-008 | MUST | AC-008.1: four tool names, parameter names asserted | `mcp::tests::tool_names_and_parameters` | PASS | `1 passed`; `search_library` adds `collection`, others unchanged; `browse_topics` takes `collection` |
| REQ-008 | MUST | AC-008.2: `list_books` returns `title`, `authors`, `source_path`, `chunks` | `query::tests::read_paths_filter_by_collection_and_leave_data_dir_untouched` | PASS | serialized keys == `[authors, chunks, source_path, title]`, authors `["Scott Chacon","Ben Straub"]` |

### Summary (phase P1+P2)

| Priority | Pass | Fail | Skip | Total |
|----------|------|------|------|-------|
| MUST | 14 | 0 | 1 | 15 |
| SHOULD | 0 | 0 | 0 | 0 |
| COULD | 0 | 0 | 0 | 0 |
| **Total** | **14** | **0** | **1** | **15** |

### Not in this phase

REQ-017..REQ-024 (P4 library), REQ-025..REQ-030 (P5 fetchers), REQ-031 (rebuild on the real setup) are not implemented yet and were not verified.

## Deliverables Check

| # | Deliverable | Expected Location | Status | Notes |
|---|-------------|-------------------|--------|-------|
| 1 | requirements.md | specs/files-first-ingestion/requirements.md | PRODUCED | Approved 2026-10-05 |
| 2 | design.md | specs/files-first-ingestion/design.md | PRODUCED | Approved plan |
| 5 | Implementation plan | specs/files-first-ingestion/plan-p1-p2.md | PRODUCED | P1+P2 combined, rationale in the plan |
| 6 | Unit tests (TDD) | `#[cfg(test)]` modules in `src/` | PRODUCED | 154/154 pass |
| 7 | Integration tests | tests/cli_storage.rs, tests/common/mod.rs | PRODUCED | 2/2 pass; real binary + fake Ollama HTTP server |
| 10 | Linter clean | — | PRODUCED | `cargo fmt --check` OK; `cargo clippy -- -D warnings` exit 0 (bin and `--tests`) |
| 12 | Code review | — | PRODUCED | Reviewer agent on the uncommitted diff; 5 important findings fixed (resumable first run via early index meta, missing table re-embed, prune grace window, only NotADatabase treated as foreign + short ingest transactions, FTS repair) plus atomic meta write, reset guards for generic dir names, dry run surfacing an unusable DB, dot-safe collection paths |
| 13 | Documentation update | README.md, CLAUDE.md | PRODUCED | Commands, config (`embedding:`), collections, write lock, data layout, `reset` |
| 17 | verification.md | specs/files-first-ingestion/verification.md | PRODUCED | This document (phase section) |
| 18 | Acceptance criteria met | — | PARTIAL | Met for REQ-001..008; later phases pending |

## Traceability

| REQ | Referenced in Tests | Referenced in Commits | Referenced in Plan |
|-----|--------------------|-----------------------|--------------------|
| REQ-001 | src/metadata_store.rs, src/config.rs (AC-001.x) | phase commit | plan-p1-p2 task 2 |
| REQ-002 | src/query.rs, src/config.rs, tests/cli_storage.rs | phase commit | task 9 |
| REQ-003 | src/lock.rs, src/metadata_store.rs | phase commit | task 3 |
| REQ-004 | src/cli/reset.rs, tests/cli_storage.rs | phase commit | task 8 |
| REQ-005 | src/indexer.rs | phase commit | task 6 |
| REQ-006 | src/indexer.rs | phase commit | task 6 |
| REQ-007 | src/query.rs | phase commit | tasks 4, 7, 10 |
| REQ-008 | src/mcp.rs, src/query.rs | phase commit | task 7 |

- ⚠ AC-006.2 depends on `colibri status` (P3); carry it into the P3 verification.

## Gate Decision (phase P1+P2)

**All MUST criteria pass (or skip with reason):** YES
**All mandatory deliverables produced:** YES (requirements.md, verification.md; acceptance criteria met for this phase)
**No MUST criteria failed:** YES

---

**Result: PASSED (phase P1+P2).** AC-006.2 was re-verified in P3 (below).

---

## Phase P3: Acceptance Criteria Results

Evidence produced in this session on `feat/ffi-p1-storage` (uncommitted tree before the P3 commit): unit tests run individually with `cargo test --bin colibri <name>`, integration via `cargo test --test cli_mirrors` / `--test cli_storage`; full suite 158 unit + 3 integration tests pass, `cargo fmt --check` OK, `cargo clippy -- -D warnings` exit 0 (bin and `--tests`).

| REQ | Priority | Criterion | Method | Result | Evidence |
|-----|----------|-----------|--------|--------|----------|
| REQ-006 | MUST | AC-006.2: `colibri status` shows `orphan chunks: 0` | `cli_mirrors::update_prune_dry_run_and_status` | PASS | `status --json` `orphan_chunks == 0` after update; human output contains `orphan chunks: 0`; with a pending prune it reports exactly the pruned doc's 1 chunk and not the outdated chunk of an edited doc |
| REQ-009 | MUST | AC-009.1: 3 files added; one edit → 1 changed, only that doc embedded | `mirror::tests::adds_changes_and_leaves_unchanged_files_alone`, `update::tests::update_embeds_only_changes_and_drops_deleted_documents` | PASS | counts (3,0,0,0) then (0,1,2,0); index `(indexed, unchanged) == (1, 2)` |
| REQ-009 | MUST | AC-009.2: no changes → 0 writes, 0 embeddings, 0 converter calls | same two tests | PASS | rows identical and canonical mtimes unchanged after the third run; converter calls stay 1; fake embedder 0 calls |
| REQ-010 | MUST | AC-010.1: deleted file not returned by search, no chunks | `mirror::tests::deleted_file_is_pruned_and_reappearing_file_is_reactivated`, update test, `cli_mirrors` | PASS | status removed/`source_deleted`, `indexed_hash == None`; CLI keyword search for the deleted note returns 0 |
| REQ-011 | MUST | AC-011.1: missing root → 0 pruned, problem in status | `mirror::tests::missing_root_prunes_nothing_and_reports_a_problem`, `cli_mirrors` (`gone` mirror) | PASS | status `error`, `root_missing` problem in `status --json` |
| REQ-011 | MUST | AC-011.2: unreadable subdirectory → 0 pruned | `mirror::tests::unreadable_subdirectory_blocks_prune`, `walk::tests::unreadable_directory_marks_walk_incomplete` | PASS | `prune_candidates == 1`, `pruned == 0`, `prune_blocked` set |
| REQ-011 | MUST | AC-011.3: 30 of 40 deleted (min 25, 0.2) → 0 pruned + problem; `--allow-mass-prune` → 30 | `mirror::tests::mass_deletion_is_blocked_unless_allowed` | PASS | message "exceed the limit of 25"; override prunes 30; problems cleared |
| REQ-012 | MUST | AC-012.1: moved folder + new `path` → 0 added, 0 pruned | `mirror::tests::moving_the_mirror_folder_keeps_document_ids` | PASS | counts (0,0,2,0); `source_path` points at the new location |
| REQ-013 | MUST | AC-013.1: dry run with pending changes writes nothing, lists counts | `cli_mirrors` (dry run after first update), `mirror::tests::dry_run_reports_without_writing` | PASS | reported added/changed/pruned = 1/1/1; recursive size+mtime snapshot of the data dir (incl. `metadata.db`) unchanged; no converter calls |
| REQ-014 | MUST | AC-014.1: `status --json` per-collection fields | `cli_mirrors` | PASS (mirrors) | two mirrors and one problem checked: kind, active, removed, pending_index, last_run_status, problems. The library row is verified with P4 (REQ-017+) |
| REQ-015 | SHOULD | AC-015.1: frontmatter filter | `cli_mirrors` | PASS | `status=active` → 1 result, `status=archived` → 0 |
| REQ-015 | SHOULD | AC-015.2: `.yaml` file found by keyword | `cli_mirrors`, `mirror::tests::yaml_is_fenced_and_unsupported_types_are_problems` | PASS | `payments-team` returns `arch/billing-service.yaml` |
| REQ-016 | SHOULD | AC-016.1: dataless flag check; reconcile treats such files as seen | `walk::tests::dataless_flag_detection`, `mirror::tests::online_only_files_are_reported_and_never_pruned` | PASS | flag true/false cases; online-only existing doc not pruned, `online_only` problem recorded |

### Summary (phase P3)

| Priority | Pass | Fail | Skip | Total |
|----------|------|------|------|-------|
| MUST | 10 | 0 | 0 | 10 |
| SHOULD | 3 | 0 | 0 | 3 |
| **Total** | **13** | **0** | **0** | **13** |

### Deliverables (phase P3)

| # | Deliverable | Location | Status | Notes |
|---|-------------|----------|--------|-------|
| 5 | Implementation plan | specs/files-first-ingestion/plan-p3.md | PRODUCED | Written alongside implementation; records review-driven decisions |
| 6 | Unit tests | `src/ingest/*`, `src/cli/status.rs` paths | PRODUCED | 158/158 pass |
| 7 | Integration tests | tests/cli_mirrors.rs, tests/cli_storage.rs (switched to mirrors) | PRODUCED | 3/3 pass |
| 10 | Linter clean | — | PRODUCED | fmt + clippy (bin, tests) clean |
| 12 | Code review | — | PRODUCED | 6 important findings fixed (metadata-only updates for doc_type/source_path, orphan definition in status, cache robustness + post-commit file deletion, absolute mirror paths, case-insensitive matching) plus prune hardening (empty-folder guard, `excluded` reason, symlink handling, subtree-exclude probe) |
| 13 | Documentation update | README.md, CLAUDE.md | PRODUCED | mirrors, `update`, `status`, prune guard, config |

Known limitation (accepted, tracked for P4): a file whose conversion fails is retried on every update.

**Gate (phase P3):** all MUST criteria pass, no MUST failed, deliverables produced. **Result: PASSED (phase P3).**
