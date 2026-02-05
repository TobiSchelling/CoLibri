#!/usr/bin/env python3
"""CLI for managing the CoLibri RAG system."""

import json
import os
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from colibri.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    LANCEDB_DIR,
    LIBRARY_PATH,
    OLLAMA_BASE_URL,
    SOURCES,
    TRANSLATION_MODEL,
    get_books_source,
)
from colibri.index_meta import read_index_meta
from colibri.index_state import read_change_events
from colibri.manifest import make_key, source_id_for_root, split_key

console = Console()


def _import_document(
    file_path: Path,
    library_path: Path,
    title: str | None,
    translate: bool = False,
    translate_model: str | None = None,
    translate_backend: str = "ollama",
) -> Path:
    """Import a document using the processor registry.

    Args:
        file_path: Path to the document
        library_path: Path to the library
        title: Optional title override
        translate: Whether to translate content to English
        translate_model: Model name/alias for translation
        translate_backend: Translation backend ("ollama" or "claude")

    Returns:
        Path to the created markdown file

    Raises:
        SystemExit: If format is unsupported
    """
    from colibri.processors import ProcessorRegistry
    from colibri.processors.utils import write_to_library

    processor = ProcessorRegistry.get_processor(file_path)

    if not processor:
        supported = ", ".join(ProcessorRegistry.supported_extensions())
        console.print(f"[red]Unsupported file type: {file_path.suffix}[/red]")
        console.print(f"[dim]Supported formats: {supported}[/dim]")
        raise SystemExit(1)

    # Derive books directory from the book source profile
    books_src = get_books_source()
    books_dir = Path(books_src.path) if books_src else library_path / "Books"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {file_path.name}...", total=None)

        # Extract content
        content = processor.extract(file_path)

        # Override title if provided
        if title:
            content.title = title

        progress.update(task, description="Writing to library...")

    # Translate if requested (outside the extraction progress context)
    if translate:
        from colibri.translate import translate_content

        content = translate_content(
            content,
            model=translate_model or TRANSLATION_MODEL,
            backend=translate_backend,
        )

    # Write to library
    output_path = write_to_library(content, books_dir)

    console.print(f"[green]Created:[/green] {output_path}")
    return output_path


@click.group()
@click.version_option()
def cli() -> None:
    """CoLibri (COntext LIBRary) - Local RAG for technical books and notes.

    Tip for CLI-native coding agents: run `colibri agent-guide`.
    """
    pass


@cli.command("import")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--library", type=click.Path(exists=True, path_type=Path), help="Library path"
)
@click.option("--title", help="Book title (defaults to filename/metadata)")
@click.option("--translate", is_flag=True, help="Translate content to English before saving")
@click.option("--translate-model", help="Model for translation (auto-detects if omitted)")
@click.option(
    "--translate-backend",
    type=click.Choice(["ollama", "claude"]),
    default="ollama",
    help="Translation backend (default: ollama)",
)
def import_book(
    file_path: Path,
    library: Path | None,
    title: str | None,
    translate: bool,
    translate_model: str | None,
    translate_backend: str,
) -> None:
    """Import a book (PDF or EPUB) into the library.

    Automatically detects the file format and uses the appropriate processor.
    """
    library_path = library or LIBRARY_PATH
    output_path = _import_document(
        file_path,
        library_path,
        title,
        translate=translate,
        translate_model=translate_model,
        translate_backend=translate_backend,
    )

    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  1. Review the imported file: {output_path}")
    console.print("  2. Run: [cyan]colibri index[/cyan] to update the search index")


@cli.command("formats")
def list_formats() -> None:
    """List supported import formats."""
    from colibri.processors import ProcessorRegistry

    console.print("[bold]Supported Import Formats[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Format", style="cyan")
    table.add_column("Extensions")

    for name, extensions in ProcessorRegistry.list_processors():
        table.add_row(name, ", ".join(extensions))

    console.print(table)


@cli.command()
@click.option("--folder", help="Index only this source (by display name)")
@click.option("--force", is_flag=True, help="Full re-index, ignoring all modes")
def index(folder: str | None, force: bool) -> None:
    """Index the library for semantic search.

    By default, respects per-source modes (static, incremental, append_only).
    Use --folder to target a specific source. Use --force for full rebuild.
    """
    from colibri.indexer import index_library

    index_library(
        folder_filter=folder,
        force=force,
    )


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
def reindex(file_path: Path) -> None:
    """Re-index a single file.

    Useful for quickly updating the index after editing a specific note.
    """
    from colibri.indexer import reindex_file
    from colibri.manifest import Manifest, get_manifest_path

    chunks = reindex_file(file_path)

    # Update manifest
    manifest_path = get_manifest_path()
    manifest = Manifest.load(manifest_path)
    # Use path relative to the source that owns it
    for src in SOURCES:
        try:
            src_root = Path(src.path).resolve()
            rel_path = str(file_path.resolve().relative_to(src_root))
            sid = source_id_for_root(src_root)
            manifest.record_file(make_key(sid, rel_path), file_path, chunks)
            break
        except ValueError:
            continue
    manifest.save(manifest_path)

    console.print(f"[green]Re-indexed {file_path.name}: {chunks} chunks[/green]")


@cli.command("agent-guide")
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["text", "markdown"]),
    default="text",
    show_default=True,
    help="Output format for copy/paste into an agent prompt",
)
def agent_guide(out_format: str) -> None:
    """Show instructions for CLI-native coding agents."""
    guide = """Tool: CoLibri (`colibri` CLI)

What it is:
- Local retrieval system over your indexed Markdown library (books + notes). Use it to ground answers in your actual corpus.

When to use it (high priority):
- The task needs \"what we already decided / documented\" or \"what does the library say\".
- You need citations/grounding for technical claims, architecture decisions, project conventions, or historical context.
- The user references a book/note/topic that might exist in their library.

Discovery (corpus is dynamic):
- First, check readiness: `colibri status` (or `colibri status --json`)
- Introspect the indexed corpus: `colibri capabilities --json`
- Detect newly indexed content: `colibri changes --since <revision> --json`
- Inspect configured sources (what can be indexed): `colibri config`
- Inspect indexed book catalog (what's currently in the DB): `colibri books` (or `colibri books --json`)
- If the user says they added/changed files, or results look stale: run `colibri index` (or `colibri reindex <file>` for a single file)

Query workflow:
1) Run `colibri search \"<query>\" -n 10` (or `--books-only`, and/or `--json`)
2) For relevant hits, open the referenced `source_file` from the configured source roots and quote/summarize with file paths.
3) If no results: broaden query terms and/or confirm indexing is up to date.

Safety:
- Do not run `colibri setup` unless explicitly asked (it may write to user config under `~/.config` and MCP config).
"""
    if out_format == "markdown":
        click.echo("## CoLibri: Agent Instructions\n")
        click.echo("```text")
        click.echo(guide.rstrip())
        click.echo("```")
        return

    click.echo(guide.rstrip())


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output capabilities as JSON")
@click.option("-n", "--top-tags", default=15, show_default=True, help="Top tags to include")
def capabilities(as_json: bool, top_tags: int) -> None:
    """Describe the currently indexed corpus and effective retrieval capabilities."""
    from colibri.config import get_config_path

    meta = read_index_meta(LANCEDB_DIR)
    try:
        from importlib.metadata import version as pkg_version

        colibri_version = pkg_version("colibri")
    except Exception:
        colibri_version = None

    # Manifest state (cheap + works without index)
    from colibri.manifest import Manifest, get_manifest_path

    manifest_path = get_manifest_path()
    manifest = Manifest.load(manifest_path)

    index_exists = bool(LANCEDB_DIR.exists() and any(LANCEDB_DIR.iterdir()))

    # Prefer precomputed summary stored during indexing (fast path).
    tags = list(meta.get("top_tags") or [])[: max(0, top_tags)]
    doc_count = meta.get("doc_count")
    books_count = meta.get("books_count")
    doc_type_counts = meta.get("doc_type_counts")

    # Backwards-compatible fallback for older indexes: compute tags from the table.
    if index_exists and not tags:
        try:
            from colibri.query import SearchEngine

            engine = SearchEngine()
            tags = engine.browse_topics()[: max(0, top_tags)]
        except Exception:
            tags = []

    caps = {
        "version": colibri_version,
        "config_path": str(get_config_path()),
        "sources": [
            {
                "name": s.display_name,
                "path": s.path,
                "mode": s.mode.value,
                "doc_type": s.doc_type,
                "extensions": list(s.extensions),
            }
            for s in SOURCES
        ],
        "index": {
            "exists": index_exists,
            "path": str(LANCEDB_DIR),
            "schema_version": meta.get("schema_version"),
            "embedding_model": meta.get("embedding_model", EMBEDDING_MODEL),
            "revision": meta.get("revision", 0),
            "digest": meta.get("digest"),
            "last_indexed_at": meta.get("last_indexed_at") or manifest.indexed_at or None,
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "file_count": meta.get("file_count", len(manifest.files)),
            "chunk_count": meta.get("chunk_count"),
            "doc_count": doc_count,
            "books_count": books_count,
            "doc_type_counts": doc_type_counts,
        },
        "data_dir": str(DATA_DIR),
        "manifest": {
            "path": str(manifest_path),
            "file_count": len(manifest.files),
            "indexed_at": manifest.indexed_at or None,
        },
        "corpus_sketch": {
            "top_tags": tags,
        },
    }

    if as_json:
        click.echo(json.dumps(caps, indent=2, ensure_ascii=True))
        return

    console.print("[bold]CoLibri Capabilities[/bold]\n")
    console.print(f"[dim]Config: {caps['config_path']}[/dim]")
    console.print(f"[dim]Data dir: {caps['data_dir']}[/dim]")
    console.print(f"[dim]Index: {caps['index']['path']}[/dim]\n")

    idx = caps["index"]
    if idx["exists"]:
        console.print(
            f"[green]Index ready[/green] - revision {idx['revision']} "
            f"(digest: {idx['digest']})"
        )
    else:
        console.print("[yellow]Index missing[/yellow] - run: colibri index")

    if idx.get("last_indexed_at"):
        console.print(f"[dim]Last indexed: {idx['last_indexed_at']}[/dim]")
    if idx.get("doc_count") is not None:
        console.print(
            f"[dim]Documents: {idx['doc_count']}  Books: {idx.get('books_count', 0)}[/dim]"
        )
    if tags:
        console.print("\n[bold]Top tags[/bold]")
        for t in tags[: max(0, top_tags)]:
            console.print(f"  - {t['tag']} ({t['document_count']})")


@cli.command()
@click.option("--since", "since_revision", default=0, show_default=True, help="Only show revisions > this")
@click.option("-n", "--limit", default=50, show_default=True, help="Max events to return")
@click.option("--json", "as_json", is_flag=True, help="Output change events as JSON")
def changes(since_revision: int, limit: int, as_json: bool) -> None:
    """Show what changed in the indexed corpus since a revision."""
    events = read_change_events(data_dir=DATA_DIR, since_revision=since_revision, limit=limit)

    def _parse_keys(keys: list[str]) -> list[dict]:
        parsed: list[dict] = []
        for k in keys:
            try:
                sid, rel = split_key(k)
                parsed.append({"source_id": sid, "path": rel})
            except Exception:
                parsed.append({"path": k})
        return parsed

    if as_json:
        summary = {"added": 0, "updated": 0, "deleted": 0, "events": len(events)}
        events_out = []
        for ev in events:
            d = ev.get("delta") or {}
            summary["added"] += len(d.get("added") or [])
            summary["updated"] += len(d.get("updated") or [])
            summary["deleted"] += len(d.get("deleted") or [])

            ev2 = dict(ev)
            ev2["delta_paths"] = {
                "added": _parse_keys(d.get("added") or []),
                "updated": _parse_keys(d.get("updated") or []),
                "deleted": _parse_keys(d.get("deleted") or []),
            }
            events_out.append(ev2)

        click.echo(json.dumps({"summary": summary, "events": events_out}, indent=2, ensure_ascii=True))
        return

    if not events:
        console.print("[dim]No index changes recorded.[/dim]")
        console.print("[dim]Tip: run 'colibri index' to (re)build and record a revision.[/dim]")
        return

    console.print("[bold]Index Changes[/bold]\n")
    for ev in events:
        rev = ev.get("revision")
        ts = ev.get("timestamp")
        digest = ev.get("digest")
        delta = ev.get("delta") or {}
        a = len(delta.get("added") or [])
        u = len(delta.get("updated") or [])
        d = len(delta.get("deleted") or [])
        console.print(f"[cyan]rev {rev}[/cyan] {ts}  (+{a} ~{u} -{d})")
        console.print(f"[dim]{digest}[/dim]")


@cli.command()
@click.argument("query")
@click.option("-n", "--limit", default=5, help="Number of results")
@click.option("--books-only", is_flag=True, help="Search only books")
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
def search(query: str, limit: int, books_only: bool, as_json: bool) -> None:
    """Search the indexed library."""
    from colibri.query import SearchEngine

    engine = SearchEngine()

    if books_only:
        results = engine.search_books(query, limit=limit)
    else:
        results = engine.search_library(query, limit=limit)

    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=True))
        return

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}[/bold cyan] (score: {r['score']})")
        console.print(f"[dim]{r['file']}[/dim]")
        console.print(f"[bold]{r['title']}[/bold] [{r['type']}]")

        # Show truncated text
        text = r["text"]
        if len(text) > 500:
            text = text[:500] + "..."
        console.print(text)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
def books(as_json: bool) -> None:
    """List indexed books."""
    from colibri.query import SearchEngine

    try:
        engine = SearchEngine()
        book_list = engine.list_books()

        if as_json:
            click.echo(json.dumps(book_list, indent=2, ensure_ascii=True))
            return

        if not book_list:
            console.print("[yellow]No books indexed yet[/yellow]")
            console.print("\n[dim]Import books with: colibri import <file>[/dim]")
            return

        table = Table(title="Indexed Books")
        table.add_column("Title", style="cyan")
        table.add_column("Chunks", justify="right")

        for book in book_list:
            table.add_row(book["title"], str(book["chunks"]))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Index may not exist. Run: colibri index[/dim]")


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output status as JSON")
def status(as_json: bool) -> None:
    """Check system status."""
    status_obj: dict = {
        "ollama": {
            "base_url": OLLAMA_BASE_URL,
            "running": False,
            "embedding_model": EMBEDDING_MODEL,
            "model_available": False,
            "models": [],
            "error": None,
        },
        "sources": [],
        "index": {
            "exists": False,
            "path": str(LANCEDB_DIR),
            "schema_version": None,
            "embedding_model": None,
            "revision": None,
            "digest": None,
            "updated_at": None,
            "last_indexed_at": None,
        },
        "manifest": {"exists": False, "path": None, "files_tracked": 0, "last_indexed": None},
        "data_dir": str(DATA_DIR),
    }

    # Check Ollama
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        status_obj["ollama"]["running"] = True
        status_obj["ollama"]["models"] = models
        status_obj["ollama"]["model_available"] = bool(
            EMBEDDING_MODEL in models or any(EMBEDDING_MODEL in m for m in models)
        )
    except httpx.ConnectError as e:
        status_obj["ollama"]["error"] = str(e)
    except Exception as e:
        status_obj["ollama"]["error"] = str(e)

    # Check sources
    for src in SOURCES:
        src_path = Path(src.path)
        src_entry = {
            "name": src.display_name,
            "path": src.path,
            "mode": src.mode.value,
            "doc_type": src.doc_type,
            "exists": src_path.exists(),
            "md_files": None,
        }
        if src_path.exists():
            try:
                src_entry["md_files"] = sum(1 for _ in src_path.rglob("*.md"))
            except OSError:
                src_entry["md_files"] = None
        status_obj["sources"].append(src_entry)

    # Check LanceDB index
    status_obj["index"]["exists"] = bool(LANCEDB_DIR.exists() and any(LANCEDB_DIR.iterdir()))
    meta = read_index_meta(LANCEDB_DIR)
    status_obj["index"]["schema_version"] = meta.get("schema_version")
    status_obj["index"]["embedding_model"] = meta.get("embedding_model")
    status_obj["index"]["revision"] = meta.get("revision")
    status_obj["index"]["digest"] = meta.get("digest")
    status_obj["index"]["updated_at"] = meta.get("updated_at")
    status_obj["index"]["last_indexed_at"] = meta.get("last_indexed_at")

    # Check manifest
    from colibri.manifest import Manifest, get_manifest_path

    manifest_path = get_manifest_path()
    status_obj["manifest"]["path"] = str(manifest_path)
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        status_obj["manifest"]["exists"] = True
        status_obj["manifest"]["files_tracked"] = len(manifest.files)
        status_obj["manifest"]["last_indexed"] = manifest.indexed_at or None

    if as_json:
        click.echo(json.dumps(status_obj, indent=2, ensure_ascii=True))
        return

    console.print("[bold]CoLibri Status[/bold]\n")

    # Check Ollama
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]

        if EMBEDDING_MODEL in models or any(EMBEDDING_MODEL in m for m in models):
            console.print(f"[green]Ollama running[/green] - {EMBEDDING_MODEL} available")
        else:
            console.print(f"[yellow]Ollama running[/yellow] - {EMBEDDING_MODEL} not found")
            console.print(f"[dim]  Available: {', '.join(models[:5])}...[/dim]")
            console.print(f"[dim]  Run: ollama pull {EMBEDDING_MODEL}[/dim]")

    except httpx.ConnectError:
        console.print("[red]Ollama not running[/red]")
        console.print("[dim]  Start with: ollama serve[/dim]")
    except Exception as e:
        console.print(f"[red]Ollama error:[/red] {e}")

    # Check sources
    console.print(f"\n[bold]Sources ({len(SOURCES)})[/bold]")
    for src in SOURCES:
        src_path = Path(src.path)
        if src_path.exists():
            file_count = sum(1 for _ in src_path.rglob("*.md"))
            console.print(
                f"  [green]{src.display_name}[/green] - {src_path} "
                f"({file_count} .md files, {src.mode.value})"
            )
        else:
            console.print(f"  [yellow]{src.display_name}[/yellow] - {src_path} (not found)")

    # Check LanceDB index
    if LANCEDB_DIR.exists() and any(LANCEDB_DIR.iterdir()):
        console.print(f"\n[green]Index exists[/green] - {LANCEDB_DIR}")
    else:
        console.print("\n[yellow]Index not found[/yellow]")
        console.print("[dim]  Run: colibri index[/dim]")

    # Check manifest
    from colibri.manifest import Manifest, get_manifest_path

    manifest_path = get_manifest_path()
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        console.print(f"[green]Manifest[/green] - {len(manifest.files)} files tracked")
        if manifest.indexed_at:
            console.print(f"[dim]  Last indexed: {manifest.indexed_at}[/dim]")
    else:
        console.print("[yellow]Manifest not found[/yellow] - run colibri index")

    # Show data directory
    console.print(f"\n[dim]Data directory: {DATA_DIR}[/dim]")


@cli.command()
def serve() -> None:
    """Start the MCP server (for Claude integration)."""
    from colibri.mcp_server import run_server

    console.print("[cyan]Starting MCP server...[/cyan]")
    console.print("[dim]This should be called by Claude, not directly.[/dim]")
    console.print("[dim]Add to claude_desktop_config.json to use.[/dim]\n")
    run_server()


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8420, help="Port to bind to")
def api(host: str, port: int) -> None:
    """Start the REST API server (for Copilot/HTTP integration)."""
    from colibri.api_server import run_server

    console.print("[cyan]Starting CoLibri REST API server...[/cyan]")
    console.print(f"[dim]Server: http://{host}:{port}[/dim]")
    console.print(f"[dim]OpenAPI docs: http://{host}:{port}/docs[/dim]")
    console.print(f"[dim]OpenAPI spec: http://{host}:{port}/openapi.json[/dim]\n")
    run_server(host=host, port=port)


@cli.command()
@click.option("--edit", is_flag=True, help="Open config file in editor")
@click.option("--path", is_flag=True, help="Show config file path only")
@click.option("--tui", is_flag=True, help="Open TUI for managing source profiles")
def config(edit: bool, path: bool, tui: bool) -> None:
    """Show current configuration."""
    from colibri.config import get_config_path

    if tui:
        from colibri.tui import run_tui

        run_tui()
        return

    config_path = get_config_path()

    if path:
        console.print(str(config_path))
        return

    if edit:
        import subprocess

        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(config_path)])
        return

    console.print("[bold]CoLibri Configuration[/bold]\n")
    console.print(f"[dim]Config file: {config_path}[/dim]\n")

    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Library path", str(LIBRARY_PATH))
    table.add_row("Data directory", str(DATA_DIR))
    table.add_row("Index location", str(LANCEDB_DIR))
    table.add_row("Ollama URL", OLLAMA_BASE_URL)
    table.add_row("Embedding model", EMBEDDING_MODEL)

    console.print(table)

    # Source profiles
    console.print("\n[bold]Source Profiles[/bold]\n")
    src_table = Table(show_header=True)
    src_table.add_column("Name", style="cyan")
    src_table.add_column("Path")
    src_table.add_column("Mode")
    src_table.add_column("Doc Type")
    src_table.add_column("Chunk Size")

    for src in SOURCES:
        chunk_info = str(src.chunk_size) if src.chunk_size else "default"
        src_table.add_row(
            src.display_name,
            src.path,
            src.mode.value,
            src.doc_type,
            chunk_info,
        )

    console.print(src_table)

    console.print("\n[dim]Edit config: colibri config --edit[/dim]")
    console.print("[dim]Environment variables can override config file settings.[/dim]")


@cli.command()
def setup() -> None:
    """Run the interactive setup wizard.

    This command helps you:
    - Check and install prerequisites (Ollama, embedding model)
    - Configure your library/notes path
    - Set up Claude Code integration (MCP)
    """
    from colibri.setup import run_setup

    success = run_setup()
    raise SystemExit(0 if success else 1)


@cli.command()
def doctor() -> None:
    """Check system health and diagnose issues."""
    from colibri.setup import check_health

    console.print("[bold]CoLibri Health Check[/bold]\n")

    health = check_health()

    # Python
    py = health["python"]
    if py["ok"]:
        console.print(f"[green]✓[/green] Python {py['version']}")
    else:
        console.print(f"[red]✗[/red] Python {py['version']} (requires 3.11+)")

    # Ollama
    ollama = health["ollama"]
    if not ollama["installed"]:
        console.print("[red]✗[/red] Ollama not installed")
        console.print("  [dim]Run: colibri setup[/dim]")
    elif not ollama["running"]:
        console.print("[yellow]![/yellow] Ollama installed but not running")
        console.print("  [dim]Start with: ollama serve[/dim]")
    elif not ollama["model_available"]:
        console.print("[yellow]![/yellow] Ollama running but embedding model missing")
        console.print("  [dim]Run: ollama pull nomic-embed-text[/dim]")
    else:
        console.print("[green]✓[/green] Ollama ready with embedding model")

    # Config
    cfg = health["config"]
    if cfg["exists"]:
        console.print(f"[green]✓[/green] Config file exists ({cfg['path']})")
    else:
        console.print("[yellow]![/yellow] Config file not found")
        console.print("  [dim]Run: colibri setup[/dim]")

    # Overall assessment
    console.print()
    all_ok = (
        py["ok"]
        and ollama["installed"]
        and ollama["running"]
        and ollama["model_available"]
        and cfg["exists"]
    )

    if all_ok:
        console.print("[green]All systems operational![/green]")
    else:
        console.print("[yellow]Some issues detected. Run 'colibri setup' to fix.[/yellow]")


if __name__ == "__main__":
    cli()
