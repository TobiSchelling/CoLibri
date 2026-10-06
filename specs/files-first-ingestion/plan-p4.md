# Implementation Plan: P4 (library)

> Spec: `requirements.md` REQ-017..REQ-024, plus the library row of AC-014.1.
> Branch: `feat/ffi-p1-storage`. Written alongside the implementation; records the decisions taken.

## Decisions

- **No `layout:` setting.** Every folder is grouped by directory: a folder with a `metadata.opf` carrying a uuid is one calibre book; any other EPUB/PDF/DOCX is a loose book. This covers calibre libraries and plain folders without configuration.
- **Identity:** `books:calibre:<uuid>` or `books:sha:<first 16 hex of SHA-256>`. Loose files reuse a known identity when path, size and mtime match, so a sweep does not hash every file.
- **One format per calibre book** by `prefer_formats` (default epub, pdf, docx); a format pinned by an explicit `add <file>` wins. `.original_epub`, covers and `.caltrash`/`.calnotes` are ignored; `.acsm`-only folders are `drm_placeholder` problems.
- **Changes:** same size+mtime → no conversion (metadata from the OPF is still refreshed); same bytes elsewhere in the same folder → format/metadata update; changed EPUB/DOCX → convert again, re-embed only if the markdown hash changed; changed PDF → `source_changed` problem, convert only with `--reconvert`.
- **Duplicates:** the same book found in another folder while the original still exists is reported (`duplicate`) on sweeps and as "already in library" on explicit adds; the stored source path is not switched.
- **Sticky removal:** `remove` sets status removed, reason `user`; canonical markdown stays, so an explicit `add <file>` restores the book without converting. Chunks are dropped by the next `update`/`index`; search hides the book immediately.
- **Never deleted:** books not found in a sweep stay searchable; `source_missing` follows whether the stored source file exists.
- **Negative conversion cache** (fixes the P3 known limitation): a failed conversion is recorded and not retried until `--retry-failed` or `--reconvert`; a missing converter tool is not recorded.
- `colibri update` also sweeps the library (`update books` for the library only); `colibri import` is now a hidden alias of `add`.
- **Paths:** explicit adds keep the given absolute path (no symlink resolution) so they match sweep paths; file identity comparisons use canonical paths.

### After code review

- A file already tracked at a path keeps its identity when its bytes change (no second document; PDF edits are flagged, EPUB edits reconverted).
- A folder whose `metadata.opf` is unreadable or lacks a uuid is skipped with a `bad_metadata` problem instead of falling back to loose identities.
- Explicit folder adds never clear a pinned format; only an explicit file add pins.
- The same-folder exemption from duplicate detection applies to calibre books only; overlapping library roots are a config error; a stored path string is kept when it names the same file.
- Online-only files are read only when a conversion is unavoidable; new online-only loose files are reported, not hashed.
- Books under a missing root or unreadable folder are not marked source-missing.
- Restore rewrites a missing canonical file (from the conversion cache); `--reconvert` clears each content hash once per run; dry runs build the same record as real runs.
- Metadata schema bumped to v8 (new `conversions.error` column).

## Tasks

1. `ingest/calibre.rs`: OPF parsing (uuid, calibre id, title, authors with role `aut`, language, subjects, publisher, date, ISBN) with `&amp;` decoding; tests (AC-020.1).
2. `ingest/convert.rs` + `metadata_store.rs`: negative cache (`conversions.error`), `check_available`, `clear_cached`; tests.
3. `config.rs`: `library:` (`doc_type`, `prefer_formats`, `roots`), `LIBRARY_COLLECTION`, absolute roots, `books` reserved for mirrors.
4. `ingest/library.rs`: `sweep`, `add_paths`, `find_books`, `remove_book`; unit tests for AC-017.1-3, AC-019.1-2, AC-021.1, AC-022.1, AC-023.1, AC-024.1-2, upgrade/pinning, dry run, missing root.
5. `ingest/update.rs`: library sweep in `update`; `--retry-failed`.
6. CLI `add` (alias `import`), `list`, `remove`; delete `cli/import.rs`; `status` shows the library row.
7. `tests/cli_library.rs` with real EPUBs built by pandoc (CI installs pandoc; the test fails on CI if pandoc is missing).
8. README/CLAUDE.md; code review; `/verify` for REQ-017..024 and the AC-014.1 library row.
