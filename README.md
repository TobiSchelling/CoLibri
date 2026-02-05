# CoLibri

CoLibri (COntext LIBRary) is a local-first RAG system for technical books and notes: import PDF/EPUB → Markdown, index into LanceDB, and query it from the command line.

This `README.md` stays intentionally short. The “how-to” lives in `docs/`.

## Quick Start

```bash
colibri doctor
colibri import ~/Downloads/book.pdf
colibri index
colibri search "clean architecture"
```

Setup, configuration, and troubleshooting: `docs/INSTALLATION.md` and `docs/CONFIGURATION.md`.

## Agent Instructions (Important)

If you are a CLI-native coding agent, treat CoLibri as an external context tool.

```bash
colibri agent-guide --format markdown
colibri capabilities --json
colibri changes --since <revision> --json
colibri search "your query" -n 10 --json
```

Recommended integration loop:

1. Call `colibri capabilities --json` once and cache `revision` (and optionally `digest`).
2. Before/while working, poll `colibri changes --since <revision> --json` and refresh capabilities when it advances.

## Docs

- Install & use: `docs/INSTALLATION.md`
- Configuration (sources, modes, data dir): `docs/CONFIGURATION.md`
- Architecture & internals: `docs/ARCHITECTURE.md`
- Maintainer notes: `docs/MAINTENANCE.md`
- Releasing / Homebrew tap (maintainers): `docs/RELEASING.md`
- Repo agent/dev guide: `AGENTS.md`
