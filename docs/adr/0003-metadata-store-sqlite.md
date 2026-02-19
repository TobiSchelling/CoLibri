# ADR 0003: SQLite Metadata Store (via `sqlite3` CLI)

Date: 2026-02-19

## Context

CoLibri needs a portable metadata store for:
- canonical documents + blobs
- sync state + cursors
- generation lifecycle state

## Decision

Use SQLite as `metadata.db` and execute SQL via the system `sqlite3` binary.

## Consequences

Pros:
- No additional Rust DB crates.
- Portable single-file DB.

Cons:
- Requires `sqlite3` on PATH.
- SQL execution errors must be surfaced clearly (`colibri doctor` checks this).

