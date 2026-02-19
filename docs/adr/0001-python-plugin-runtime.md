# ADR 0001: Python Plugin Runtime for Ingestion

Date: 2026-02-19

## Context

Ingestion connectors and document conversion libraries are ecosystem-heavy in Python (e.g., SaaS SDKs, document conversion toolchains).
CoLibri’s indexing and serving core benefits from Rust’s stable binary distribution and runtime efficiency.

## Decision

Use a Python plugin runtime for the ingestion plane:
- Plugins run out-of-process.
- Host<->plugin contract is versioned and uses JSON (request) + JSONL (envelopes) over stdio for v1.

## Consequences

Pros:
- Faster connector velocity.
- Core stays small and stable.

Cons:
- Additional runtime dependency (Python) for ingestion.
- Host must defend against plugin failures (timeouts, output caps, validation).

