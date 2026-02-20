# Concepts

## Import vs ingest (plugins)

- **Import**: a convenience command for turning a *single file* into Markdown and ingesting it into the canonical store (one-off).
- **Ingest (plugins)**: a repeatable sync process that produces **canonical Markdown** and tracks **incremental state** (cursor) per source.

## Canonical store

The canonical store is CoLibri’s managed, markdown-only representation of your corpus.

- Written by `colibri plugins ingest|sync|sync-all`
- Read by `colibri index`
- Designed for portability (`COLIBRI_HOME` can be copied/moved)

## Corpus

Your “corpus” is the set of canonical Markdown documents CoLibri considers indexable.

## Index, profiles, and routing

- **Index**: the vector store content produced from canonical Markdown.
- **Embedding profile**: (provider, endpoint, model, locality) for embedding.
- **Routing policy**: maps document `classification` → embedding profile.

Safety rule:
- `restricted` and `confidential` must route to **local** embedding profiles.

## Generations

A **generation** is a versioned index layout.

- Build a new generation in parallel (shadow build)
- Activate when ready (atomic switch)
- Keep older generations for rollback

## Serve

`colibri serve` starts an MCP server. It checks alignment between active generation and embedding profiles and refuses to start if nothing is queryable.

## Plugins

Plugins are ingestion connectors. They run out-of-process and emit document envelopes as JSONL.

Typical plugin responsibilities:
- discover source items
- fetch content
- convert to Markdown
- attach metadata (classification, doc_type, tags)
- emit tombstones for deletions (when possible)
