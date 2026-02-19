# ADR 0002: Canonical Document Envelope v1 (JSONL)

Date: 2026-02-19

## Context

CoLibri needs a stable contract between ingestion and the canonical store.
The contract must support:
- Stable document identity
- Incremental sync via cursors
- Deletions (tombstones)

## Decision

Adopt `document-envelope` schema v1 emitted as JSONL on stdout:
- One envelope per line
- Optional cursor control line: `{ "type": "cursor", "cursor": {...} }`

## Consequences

- Easy streaming and backpressure in the host.
- Strict validation is required to reject malformed outputs early.

