# Files-First Ingestion Requirements

> **Feature:** files-first-ingestion
> **Date:** 2026-10-05
> **Repo:** CoLibri
> **Repo Type:** code
> **Spec Directory:** specs/files-first-ingestion/
> **Design Doc Location:** specs/files-first-ingestion/design.md (approved plan, phases P1-P5)
> **Status:** In Progress (P1+P2, P3, P4 verified 2026-10-06; P5 verified 2026-10-08; REQ-031 rebuild checks partly open)

---

## Context

CoLibri's connector model re-fetches and re-converts every source on every sync, never propagates deletions, and keys documents on the root path, so moves duplicate content (73 books were indexed twice, 807 of 825 vault notes went stale) and a metadata wipe left the index unusable. Books and living sources need different semantics: books are added once and survive source moves, while the vault, architecture docs, Zephyr and later Confluence content must update incrementally with deletions. The outcome is a files-first design with two verbs (`colibri add` for the library, `colibri update` for mirrors), a `colibri status` overview, and fetchers that write remote content as markdown files DEVONthink can index.

## Requirements

### Storage foundation (P1)

### REQ-001: Metadata DB is never destroyed implicitly [MUST]
**Pattern:** Unwanted

If the metadata DB cannot be opened, is not a CoLibri database, or has a different schema version, then CoLibri shall exit with an error that names `colibri reset` and shall leave the DB file unmodified in place.

**Acceptance Criteria:**
- [ ] AC-001.1: A test writes non-SQLite bytes to `metadata.db`; every write command returns an error and the file is byte-identical afterwards.
- [ ] AC-001.2: A test with a DB whose `user_version` differs from the current schema version gets an error message containing `colibri reset`; the file is unchanged.

---

### REQ-002: Read paths never write [MUST]
**Pattern:** Ubiquitous

The `search`, `serve`, `list` and `status` commands and all MCP tools shall open the metadata DB read-only and shall not create or modify any file in the data directory.

**Acceptance Criteria:**
- [ ] AC-002.1: A test runs the search and serve initialization paths against a temp data dir and asserts that no file in it changed (existence and mtime).
- [ ] AC-002.2: With no data dir present, `colibri search x` exits with an error and creates no files.

---

### REQ-003: Single writer with fast failure [MUST]
**Pattern:** State-Driven

While a write command (`add`, `update`, `remove`, `reset`, `index`) holds the write lock, a second write command shall exit immediately with an error naming the PID of the lock holder, and read commands shall continue to return results.

**Acceptance Criteria:**
- [ ] AC-003.1: A test holds the lock and starts a second writer; it fails within 1 s with a message containing the holder PID.
- [ ] AC-003.2: A test runs a read query while a writer holds an open write transaction; the read succeeds.

---

### REQ-004: Explicit reset [MUST]
**Pattern:** Event-Driven

When the user runs `colibri reset` and types the confirmation word, CoLibri shall delete the metadata DB, the canonical store and the index, and shall not touch any other path (config, mirror folders, library sources).

**Acceptance Criteria:**
- [ ] AC-004.1: A test runs reset on a populated temp home; `metadata.db`, `canonical/` and `index/` are gone, and the config file and a mirror folder are untouched.
- [ ] AC-004.2: Reset without the confirmation (or with a wrong word) deletes nothing.

---

### Index and search (P2)

### REQ-005: Resumable indexing [MUST]
**Pattern:** Event-Driven

When an indexing run fails partway, CoLibri shall keep every document whose chunks were already committed as indexed, so a rerun embeds only the remaining documents.

**Acceptance Criteria:**
- [ ] AC-005.1: A test with a fake embedder failing on document 3 of 5 shows documents 1-2 indexed; the rerun embeds exactly documents 3-5.

---

### REQ-006: No orphan chunks [MUST]
**Pattern:** Event-Driven

When an indexing run completes, CoLibri shall delete LanceDB chunks whose `doc_id` has no active document, and `colibri status` shall report the orphan chunk count.

**Acceptance Criteria:**
- [ ] AC-006.1: A test inserts chunks for an unknown doc_id; after an index run they are gone and chunks of active documents remain.
- [ ] AC-006.2: `colibri status` shows `orphan chunks: 0` after a completed run.

---

### REQ-007: One embedding setup, collection filter [MUST]
**Pattern:** Ubiquitous

CoLibri shall use a single embedding configuration (`embedding.endpoint`, `embedding.model`; `ollama:` accepted as alias) and one index, without classification, embedding profiles, routing or generations, and search shall filter by `collection` and `doc_type`.

**Acceptance Criteria:**
- [ ] AC-007.1: `colibri search --collection books "x"` and MCP `search_library` with `collection: books` return only documents of collection `books`.
- [ ] AC-007.2: Grepping `src/` finds no `classification`, `routing_policy`, `embedding_profiles` or `active_generation` identifiers.

---

### REQ-008: MCP tool compatibility [MUST]
**Pattern:** Ubiquitous

The MCP server shall keep the tools `search_library`, `search_books`, `list_books` and `browse_topics` with their current parameters (except `classification`, replaced by `collection`), and `list_books` shall return title, authors, source path and chunk count.

**Acceptance Criteria:**
- [ ] AC-008.1: `tools/list` returns the four tool names; a test asserts the parameter names of each.
- [ ] AC-008.2: `list_books` on a test library returns `title`, `authors`, `source_path`, `chunks` per book.

---

### Mirrors (P3)

### REQ-009: Incremental mirror reconcile [MUST]
**Pattern:** Event-Driven

When `colibri update` runs, CoLibri shall, for each mirror, ingest new files, re-ingest files whose content changed, and leave unchanged files untouched.

**Acceptance Criteria:**
- [ ] AC-009.1: A test mirror with 3 files: first update adds 3; after editing one file, the second update reports 1 changed and embeds only that document.
- [ ] AC-009.2: A third update with no changes reports 0 writes and 0 embeddings and makes 0 converter calls.

---

### REQ-010: Deletions propagate [MUST]
**Pattern:** Event-Driven

When a file is no longer present in a mirror whose walk completed without errors, CoLibri shall tombstone its document and remove its chunks from the index.

**Acceptance Criteria:**
- [ ] AC-010.1: A test deletes one file from a mirror; after update the document is not returned by search and has no chunks.

---

### REQ-011: Guarded prune [MUST]
**Pattern:** Unwanted

If a mirror root is missing, a directory below it cannot be read, or the deletions would exceed `max(prune.min_count, prune.max_fraction × active documents)`, then CoLibri shall prune nothing for that mirror and shall record a problem; `--allow-mass-prune` shall lift only the threshold check.

**Acceptance Criteria:**
- [ ] AC-011.1: Missing root: update prunes 0 documents and `status` lists a problem for the mirror.
- [ ] AC-011.2: Unreadable subdirectory: update prunes 0 documents.
- [ ] AC-011.3: Deleting 30 of 40 files (min_count 25, max_fraction 0.2) prunes 0 and records a problem; rerunning with `--allow-mass-prune` prunes 30.

---

### REQ-012: Stable mirror identity [MUST]
**Pattern:** Ubiquitous

Mirror document ids shall be `<mirror name>:<path relative to the mirror root>`, so changing a mirror's `path` in config does not change document ids.

**Acceptance Criteria:**
- [ ] AC-012.1: A test moves a mirror folder and updates `path`; the next update reports 0 added and 0 pruned.

---

### REQ-013: Dry run [MUST]
**Pattern:** State-Driven

While `--dry-run` is set, `colibri update` shall report the planned adds, changes and prunes per collection and shall write nothing.

**Acceptance Criteria:**
- [ ] AC-013.1: A test compares data dir file mtimes and DB row counts before and after `update --dry-run` with pending changes; nothing changed, and the report lists the expected counts.

---

### REQ-014: Status overview [MUST]
**Pattern:** Event-Driven

When the user runs `colibri status`, CoLibri shall show per collection the counts of active, removed and source-missing documents, the last run time and outcome, the number of documents pending indexing, open problems, and the orphan chunk count.

**Acceptance Criteria:**
- [ ] AC-014.1: On a test home with one library, two mirrors and one recorded problem, `status --json` contains these fields per collection with the expected values.

---

### REQ-015: Frontmatter and YAML files [SHOULD]
**Pattern:** Ubiquitous

CoLibri shall store YAML frontmatter of markdown files as filterable metadata for every collection, and shall index `.yaml`/`.yml` files listed in a mirror's `include`.

**Acceptance Criteria:**
- [ ] AC-015.1: A mirror file with frontmatter `status: active` is returned by `search --frontmatter status=active` and not by `status=draft`.
- [ ] AC-015.2: A `.yaml` file in an architecture mirror is returned by a keyword search for a term it contains.

---

### REQ-016: Online-only files [SHOULD]
**Pattern:** Unwanted

If a source file is an online-only cloud placeholder, then CoLibri shall skip it, record a problem, and not treat it as missing.

**Acceptance Criteria:**
- [ ] AC-016.1: A unit test of the placeholder check returns true for a metadata value with the `SF_DATALESS` flag and false otherwise; the reconcile treats such files as seen.

---

### Library (P4)

### REQ-017: Sweep adds only new books [MUST]
**Pattern:** Event-Driven

When the user runs `colibri add` without arguments, CoLibri shall scan the configured library roots and add only books that are not yet known, identifying a book by the calibre uuid in its `metadata.opf` and otherwise by the SHA-256 of the file.

**Acceptance Criteria:**
- [ ] AC-017.1: A second sweep over an unchanged test library makes 0 converter calls and reports 0 new books.
- [ ] AC-017.2: Moving a calibre book folder (same uuid) updates its source path without conversion or re-embedding.
- [ ] AC-017.3: A loose EPUB without OPF gets a sha-based identity; adding it again from another path reports it as already known.

---

### REQ-018: Explicit add [MUST]
**Pattern:** Event-Driven

When the user runs `colibri add <file|folder>...`, CoLibri shall add the given books, report already-known ones as skipped, and index the new ones in the same run (unless `--no-index`).

**Acceptance Criteria:**
- [ ] AC-018.1: `colibri add book.epub` on a test home makes the book searchable without another command; repeating it reports `already in library`.

---

### REQ-019: One format per calibre book [MUST]
**Pattern:** Ubiquitous

CoLibri shall choose exactly one format per calibre book by `library.prefer_formats` (default EPUB before PDF), shall ignore `.original_epub`, `.opf`, covers, `.caltrash` and `.calnotes`, and shall record a problem for books that only have an `.acsm` file.

**Acceptance Criteria:**
- [ ] AC-019.1: A test book folder with EPUB and PDF yields one document with `format = epub`.
- [ ] AC-019.2: An ACSM-only folder yields no document and one problem entry of kind `drm_placeholder`.

---

### REQ-020: Book metadata from calibre [MUST]
**Pattern:** Ubiquitous

CoLibri shall take title, authors, language and tags of a calibre book from its `metadata.opf` and return them in `list`, `list_books` and search results.

**Acceptance Criteria:**
- [ ] AC-020.1: The Pro Git OPF fixture yields title `Pro Git` and authors `["Scott Chacon", "Ben Straub"]`; an OPF with `&amp;` in a name decodes to `&`.

---

### REQ-021: Sticky removal [MUST]
**Pattern:** Event-Driven

When the user removes a book with `colibri remove`, CoLibri shall drop it from search results immediately, shall not re-add it on later sweeps, and shall re-add it only on an explicit `colibri add <path>`.

**Acceptance Criteria:**
- [ ] AC-021.1: After `remove`, search returns no chunk of the book; a sweep reports it as removed and adds nothing; `add <path>` restores it.

---

### REQ-022: Books survive source loss [MUST]
**Pattern:** Unwanted

If a library book's source file disappears, then CoLibri shall keep the book searchable and mark it as source-missing in `status` and `list`.

**Acceptance Criteria:**
- [ ] AC-022.1: A test deletes a book's source file; after a sweep, search still returns its chunks and `status` counts 1 source-missing.

---

### REQ-023: Convert once [MUST]
**Pattern:** Ubiquitous

CoLibri shall never convert identical source bytes twice, unless the user passes `--reconvert`.

**Acceptance Criteria:**
- [ ] AC-023.1: With a counting fake converter, re-adding the same bytes under a different path or identity triggers 0 additional conversions; `--reconvert` triggers exactly 1.

---

### REQ-024: Changed book sources [SHOULD]
**Pattern:** Event-Driven

When the chosen EPUB of a book changes, CoLibri shall reconvert it and re-embed only if the resulting markdown changed; when a chosen PDF changes, CoLibri shall flag the book in `status` and reconvert only with `--reconvert`.

**Acceptance Criteria:**
- [ ] AC-024.1: Changed EPUB bytes with identical converted markdown cause 1 conversion and 0 embeddings.
- [ ] AC-024.2: Changed PDF bytes cause 0 conversions and one `source_changed` problem.

---

### Fetchers (P5)

### REQ-025: Zephyr fetcher writes files [MUST]
**Pattern:** Optional

Where a mirror has `fetch.type: zephyr_scale`, `colibri update` shall write one `<KEY>.md` per test case, with YAML frontmatter (key, name, folder, status, priority, labels, updated_on), into the mirror folder (default `<mirrors_dir>/<name>`) and then reconcile that folder.

**Acceptance Criteria:**
- [ ] AC-025.1: A fetch test with a canned API listing of 3 test cases writes 3 files whose frontmatter parses and contains the listed keys.
- [ ] AC-025.2: After update, `search --frontmatter status=<value>` filters Zephyr documents.

---

### REQ-026: Fetch safety [MUST]
**Pattern:** Unwanted

If a fetcher's listing is incomplete (no final page with `isLast: true`, or fewer items than `total`) or the fetch fails, then the fetcher shall delete no files; if steps or links of a test case cannot be fetched, the fetcher shall keep the existing file for that test case.

**Acceptance Criteria:**
- [ ] AC-026.1: Pagination tests: a page without `isLast` continues via `next`; a listing shorter than `total` is reported incomplete and deletes nothing.
- [ ] AC-026.2: A test where the steps request fails leaves the previous `<KEY>.md` byte-identical and records a problem.

---

### REQ-027: Fetchers own only marked folders [MUST]
**Pattern:** Unwanted

If a mirror folder lacks the `.colibri-mirror` marker, then a fetcher shall not delete or overwrite files in it; fetchers shall not rewrite files whose content is unchanged.

**Acceptance Criteria:**
- [ ] AC-027.1: Fetch into a non-empty folder without a marker fails with an error and changes nothing.
- [ ] AC-027.2: A second fetch with identical API data leaves all file mtimes unchanged.

---

### REQ-028: Incremental Zephyr fetch [SHOULD]
**Pattern:** Event-Driven

When a test case's fingerprint (list item, folder path, status and priority names) is unchanged since the last fetch and the last full refresh is younger than `full_refresh_days`, the fetcher shall not request its steps and links again.

**Acceptance Criteria:**
- [ ] AC-028.1: A second fetch with an unchanged listing makes 0 step/link requests; changing one item's `updatedOn` makes requests for that item only.

---

### REQ-029: Script fetchers [SHOULD]
**Pattern:** Optional

Where a mirror has `fetch.type: command`, `colibri update` shall run the configured argv with `{path}` replaced by the mirror folder, and shall treat a non-zero exit as a partial fetch that disables pruning for that mirror.

**Acceptance Criteria:**
- [ ] AC-029.1: A command that writes two files and exits 0 results in two documents; a command that exits 1 after deleting a file results in 0 pruned documents and a problem entry.

---

### REQ-030: Clear config migration [MUST]
**Pattern:** Unwanted

If the config contains the removed `connectors:` key or an unknown key in `library`/`mirrors`, then CoLibri shall exit with an error that names the key and points to the new `library:`/`mirrors:` structure.

**Acceptance Criteria:**
- [ ] AC-030.1: Loading a config with `connectors:` fails with a message containing `mirrors:`; a typo `exlude:` in a mirror fails naming `exlude`.

---

### End-to-end (rebuild after P5)

### REQ-031: Rebuild on the real setup [MUST]
**Pattern:** Event-Driven

When the user's data is rebuilt with the new config (`colibri reset`, then `colibri update`), CoLibri shall contain the calibre books, the vault, the architecture artifacts and the Zephyr CTSLAB test cases, with working search over all of them.

**Acceptance Criteria:**
- [ ] AC-031.1: `colibri list --books` shows the calibre books with EPUB or PDF (about 96) with authors; `status` lists ACSM-only books as problems.
- [ ] AC-031.2: Keyword, semantic and hybrid searches each return hits whose source paths exist on disk; MCP `search_books` works from a fresh Claude session.
- [ ] AC-031.3: Deleting a scratch note in the vault followed by `colibri update` prunes exactly that document; `colibri add` afterwards reports 0 new books.

---

## Deliverables Contract

The following deliverables are committed for this feature:

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 1 | requirements.md | ☑ Committed | This document |
| 2 | design.md | ☑ Committed | Approved plan copied to spec dir |
| 5 | Implementation plan | ☑ Committed | Per-phase task plan before each phase |
| 6 | Unit tests (TDD) | ☑ Committed | Tempdir fixtures, fake converter/embedder/API |
| 7 | Integration tests | ☑ Committed | `tests/` running add/update/search/status against a temp home with a fake embedder |
| 10 | Linter clean | ☑ Committed | `cargo clippy -- -D warnings`, `cargo fmt --check` |
| 12 | Code review | ☑ Committed | Per phase before merge |
| 13 | Documentation update | ☑ Committed | README and CLAUDE.md describe the new commands, config and data layout |
| 17 | verification.md | ☑ Committed | From /verify |
| 18 | Acceptance criteria met | ☑ Committed | All MUST criteria |

**Mandatory (cannot skip):** requirements.md, verification.md, acceptance criteria met

## Constraints

- Ollama remains the only embedding backend; macOS arm64 only; Rust toolchain pinned to 1.93.0.
- No migration of existing data: the user agreed to `colibri reset` and a full rebuild.
- MCP tool names stay the same (REQ-008).
- Library sources (calibre on OneDrive) are only read, never written.
- Mirror folders written by fetchers live outside `~/PKM` and outside OneDrive (default `~/Documents/CoLibri/mirrors`); CoLibri only writes inside folders it marked.
- Each phase P1-P5 is merged only when `make test lint` passes and the phase's criteria pass.

## Out of Scope

- P6 polish beyond README/CLAUDE.md: `bootstrap` to `init`, `docs/user`/`tour`/`instructions` rewrite, ADR updates, deleting `plugins/` and old plan docs (separate follow-up).
- Rename reuse (moving chunks for renamed files instead of re-embedding).
- A built-in Confluence fetcher (only the generic `command` fetcher is in scope).
- Non-Ollama embedding providers, multiple embedding profiles.
- Configuring DEVONthink (the user indexes the mirror folder and calibre library manually).

## Traceability

| REQ | Design Section | Plan Task | Test / Verification |
|-----|---------------|-----------|---------------------|
| REQ-001 | P1 | plan-p1-p2 #2 | metadata_store::tests::{non_sqlite_file_is_rejected_and_untouched, old_schema_is_rejected_and_untouched}; cli_storage::pre_v7_metadata_db_is_reported_and_left_unchanged |
| REQ-002 | P0, P1 | #9 | query::tests::read_paths_filter_by_collection_and_leave_data_dir_untouched; cli_storage round trip |
| REQ-003 | P1 | #3 | lock::tests::second_writer_fails_fast_with_holder_pid; metadata_store::tests::read_succeeds_while_write_transaction_is_open |
| REQ-004 | P1 | #8 | cli::reset::tests; cli_storage reset steps |
| REQ-005 | P2 | #6 | indexer::tests::interrupted_run_resumes_with_remaining_documents |
| REQ-006 | P2, P3 | #6 | indexer::tests::orphan_chunks_are_purged; cli_mirrors status orphan_chunks |
| REQ-007 | P2 | #4, #7, #10 | query::tests::filter_collection_and_doc_type; grep gate |
| REQ-008 | P2 | #7 | mcp::tests::tool_names_and_parameters; query e2e list_books shape |
| REQ-009 | P3 | plan-p3 #6, #7 | mirror::tests::adds_changes_and_leaves_unchanged_files_alone; update::tests |
| REQ-010 | P3 | #6, #7 | mirror::tests::deleted_file_is_pruned_and_reappearing_file_is_reactivated; cli_mirrors |
| REQ-011 | P3 | #6 | mirror::tests::{missing_root_*, unreadable_subdirectory_*, mass_deletion_*} |
| REQ-012 | P3 | #6 | mirror::tests::moving_the_mirror_folder_keeps_document_ids |
| REQ-013 | P3 | #6, #10 | cli_mirrors (dry run with existing data) |
| REQ-014 | P3, P4 | #8 | cli_mirrors, cli_library status --json |
| REQ-015 | P3 | #6, #10 | cli_mirrors frontmatter + yaml search |
| REQ-016 | P3 | #2, #6 | walk::tests::dataless_flag_detection; mirror::tests::online_only_* |
| REQ-017 | P4 | plan-p4 #4 | library::tests::{sweep_adds_*, moved_calibre_book_*, loose_files_*}; cli_library |
| REQ-018 | P4 | #6 | cli_library |
| REQ-019 | P4 | #4 | library::tests::one_format_per_book_and_drm_only_books_are_problems |
| REQ-020 | P4 | #1 | calibre::tests::parses_calibre_opf |
| REQ-021 | P4 | #4, #6 | library::tests::removed_books_stay_removed_until_added_explicitly; cli_library |
| REQ-022 | P4 | #4 | library::tests::books_whose_file_disappears_stay_searchable; cli_library |
| REQ-023 | P4 | #2, #4 | library::tests::same_bytes_*; convert::tests |
| REQ-024 | P4 | #4 | library::tests::changed_epub_is_reconverted_but_changed_pdf_is_flagged |
| REQ-025 | P5 | plan-p5 #3, #4, #9 | zephyr::tests::{writes_one_markdown_file_per_test_case, fetched_test_cases_reconcile_into_documents}; cli_fetch |
| REQ-026 | P5 | #2, #4 | api::tests::pagination_*; zephyr::tests::{deletes_only_after_complete_listings, failed_steps_keep_the_previous_file, mass_deletion_is_blocked_without_override} |
| REQ-027 | P5 | #1, #4 | fetch::tests::*; zephyr::tests::unchanged_cases_cost_no_requests_and_no_writes |
| REQ-028 | P5 | #4, #9 | zephyr::tests::{unchanged_cases_*, full_refresh_*, failed_full_refresh_*}; cli_fetch |
| REQ-029 | P5 | #5, #7, #9 | command::tests; cli_fetch::script_fetchers_fill_folders_and_failures_block_pruning |
| REQ-030 | P5 | #6 | config::tests::mirrors_are_resolved_and_validated |
| REQ-031 | Rebuild | | |
