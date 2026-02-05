#!/usr/bin/env python3
"""Index content sources for semantic search.

Uses direct LanceDB SDK and Ollama HTTP API — no framework dependency.

Supports four per-folder indexing modes:

- **static**: Index once, skip on subsequent runs unless forced.
- **incremental**: Track file changes via manifest, re-index only modified/new.
- **append_only**: Only index files not yet known, never re-check existing.
- **disabled**: Skip entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
from rich.console import Console
from rich.progress import track

from colibri.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    LANCEDB_DIR,
    SOURCES,
    FolderProfile,
    IndexMode,
    ensure_directories,
)
from colibri.embedding import embed_texts
from colibri.doc_catalog import (
    compute_summary as compute_catalog_summary,
    get_catalog_path,
    load_catalog,
    prune_missing_files,
    save_catalog,
    update_from_index_row,
)
from colibri.index_meta import SCHEMA_VERSION, read_index_meta, write_index_meta
from colibri.index_state import (
    append_change_event,
    compute_delta_from_signatures,
    compute_digest_from_signature,
    manifest_signature,
)
from colibri.manifest import (
    Manifest,
    get_manifest_path,
    is_namespaced_key,
    make_key,
    source_id_for_root,
    split_key,
)
from colibri.source_factory import compute_nested_exclusions, create_source_for_profile
from colibri.sources import ContentSource

# nomic-embed-text has 8192 token context, but we use chars as a safe proxy
# Token/char ratio varies (code has lower ratio), so be conservative
# 8192 tokens * 2.5 chars/token ≈ 20k chars, use 16k for safety margin
MAX_CHUNK_CHARS = 16000

console = Console()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IndexResult:
    """Summary of an indexing operation."""

    total_chunks: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    errors: int = 0

    def accumulate(self, other: IndexResult) -> None:
        """Merge another result into this one."""
        self.total_chunks += other.total_chunks
        self.files_indexed += other.files_indexed
        self.files_skipped += other.files_skipped
        self.files_deleted += other.files_deleted
        self.errors += other.errors


# ---------------------------------------------------------------------------
# Low-level helpers (unchanged)
# ---------------------------------------------------------------------------


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks on natural boundaries.

    Tries paragraph boundaries first, then sentence boundaries,
    and finally hard-breaks at chunk_size.
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary (\n\n)
        segment = text[start:end]
        para_break = segment.rfind("\n\n")
        if para_break > chunk_size // 4:
            end = start + para_break + 2  # include the \n\n
        else:
            # Try sentence boundary (. followed by space or newline)
            for sep in (". ", ".\n", "? ", "!\n", "! ", "?\n"):
                sent_break = segment.rfind(sep)
                if sent_break > chunk_size // 4:
                    end = start + sent_break + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _try_open_table(
    db: lancedb.DBConnection,
    table_name: str,
) -> lancedb.table.Table | None:
    """Try to open an existing LanceDB table, return None if missing."""
    try:
        return db.open_table(table_name)
    except Exception:
        return None


def _ensure_table(
    db: lancedb.DBConnection,
    table: lancedb.table.Table | None,
    table_name: str,
    rows: list[dict],
) -> lancedb.table.Table:
    """Return the existing table after adding *rows*, or create a new one."""
    if table is not None:
        table.add(rows)
        return table
    return db.create_table(table_name, data=rows)


# ---------------------------------------------------------------------------
# Per-file row building
# ---------------------------------------------------------------------------


def _build_rows_for_doc(
    source: ContentSource,
    doc_path: Path,
    profile: FolderProfile,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Read a single document and return its chunk rows (without vectors)."""
    source_doc = source.read_document(doc_path)

    # Frontmatter type takes precedence; profile doc_type is the fallback.
    doc_type = source_doc.doc_type if "type" in source_doc.metadata else profile.doc_type

    tags_str = ",".join(str(t) for t in source_doc.tags) if source_doc.tags else ""
    chunks = _split_text(source_doc.content, chunk_size, chunk_overlap)

    rows: list[dict] = []
    for chunk_text in chunks:
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS] + "..."
        rows.append(
            {
                "text": chunk_text,
                "source_file": str(source_doc.path),
                "title": source_doc.title,
                "doc_type": doc_type,
                "folder": source_doc.folder,
                "source_name": source_doc.source_name,
                "source_type": source_doc.source_type,
                "tags": tags_str,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Deleted-file detection
# ---------------------------------------------------------------------------


def _detect_deleted_files(
    manifest: Manifest,
    known_files: set[str],
    current_files: set[str],
    table: lancedb.table.Table | None,
) -> int:
    """Remove index chunks and manifest entries for files no longer on disk.

    Args:
        manifest: The manifest to update.
        known_files: Manifest keys that belong to this source (namespaced keys).
        current_files: Files currently on disk (namespaced keys).
        table: LanceDB table (for chunk deletion).

    Returns the number of deleted files removed.
    """
    if table is None:
        return 0

    deleted = 0

    for key in list(known_files):
        if key not in current_files:
            _, rel_path = split_key(key)
            escaped = rel_path.replace("'", "''")
            table.delete(f"source_file = '{escaped}'")
            manifest.remove_file(key)
            deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# Folder-level indexing
# ---------------------------------------------------------------------------


def _classify_files(
    source: ContentSource,
    profile: FolderProfile,
    manifest: Manifest,
    force: bool,
    source_id: str,
) -> tuple[list[Path], int]:
    """Decide which files in *profile* need (re-)indexing.

    Returns ``(files_to_index, files_skipped)``.
    """
    all_files = list(source.list_documents())

    if force:
        return all_files, 0

    to_index: list[Path] = []
    skipped = 0

    for fp in all_files:
        rel = str(fp)
        key = make_key(source_id, rel)

        if profile.mode == IndexMode.STATIC:
            if manifest.is_file_known(key):
                skipped += 1
            else:
                to_index.append(fp)

        elif profile.mode == IndexMode.INCREMENTAL:
            abs_path = source.root_path / fp
            if manifest.is_file_changed(key, abs_path):
                to_index.append(fp)
            else:
                skipped += 1

        elif profile.mode == IndexMode.APPEND_ONLY:
            if manifest.is_file_known(key):
                skipped += 1
            else:
                to_index.append(fp)

    return to_index, skipped


def _index_folder(
    source: ContentSource,
    profile: FolderProfile,
    manifest: Manifest,
    db: lancedb.DBConnection,
    table: lancedb.table.Table | None,
    table_name: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
    overwrite_first: bool,
    catalog: dict[str, dict] | None = None,
) -> tuple[IndexResult, lancedb.table.Table | None]:
    """Index a single folder according to its profile.

    Args:
        source: Content source scoped to this folder.
        profile: The folder's indexing profile.
        manifest: The current manifest (mutated in-place).
        db: LanceDB connection.
        table: Current table reference (may be None on first folder).
        table_name: LanceDB table name.
        chunk_size: Effective chunk size for this folder.
        chunk_overlap: Effective chunk overlap for this folder.
        force: Whether to ignore the folder mode.
        overwrite_first: If True and table is None, create with overwrite mode.

    Returns:
        (IndexResult, updated table reference)
    """
    display = profile.display_name
    all_files = list(source.list_documents())
    all_rel_paths = {str(p) for p in all_files}

    src_id = source_id_for_root(source.root_path)
    files_to_index, files_skipped = _classify_files(
        source, profile, manifest, force, source_id=src_id
    )

    prefix = f"{src_id}:"
    manifest_keys_for_source = {k for k in manifest.files if k.startswith(prefix)}
    current_keys_for_source = {make_key(src_id, rel) for rel in all_rel_paths}

    if not files_to_index:
        # Still detect deletions for incremental mode
        deleted = 0
        if profile.mode == IndexMode.INCREMENTAL and not force:
            deleted = _detect_deleted_files(
                manifest, manifest_keys_for_source, current_keys_for_source, table
            )
        if files_skipped or deleted:
            console.print(
                f"[dim]{display}: {files_skipped} unchanged"
                + (f", {deleted} removed" if deleted else "")
                + "[/dim]"
            )
        return IndexResult(files_skipped=files_skipped, files_deleted=deleted), table

    # Read and chunk all files for this folder
    rows: list[dict] = []
    files_indexed = 0
    errors = 0
    file_chunk_counts: dict[str, int] = {}

    for doc_path in track(files_to_index, description=f"Reading {display}..."):
        try:
            doc_rows = _build_rows_for_doc(source, doc_path, profile, chunk_size, chunk_overlap)
            rows.extend(doc_rows)
            file_chunk_counts[str(doc_path)] = len(doc_rows)
            files_indexed += 1
            if catalog is not None and doc_rows:
                abs_path = (source.root_path / doc_path).resolve()
                update_from_index_row(
                    catalog,
                    abs_path=abs_path,
                    row=doc_rows[0],
                    chunk_count=len(doc_rows),
                    indexed_at=None,
                    manifest_key=make_key(src_id, str(doc_path)),
                )
        except Exception as e:
            console.print(f"[yellow]Skipping {doc_path.name}: {e}[/yellow]")
            errors += 1

    if not rows:
        return (
            IndexResult(files_skipped=files_skipped, errors=errors),
            table,
        )

    # Embed all chunks for this folder in one batch
    console.print(f"[dim]Embedding {len(rows)} chunks from {files_indexed} files...[/dim]")
    texts = [r["text"] for r in rows]
    vectors = embed_texts(texts)
    for row, vec in zip(rows, vectors, strict=True):
        row["vector"] = vec

    # Write to LanceDB
    if overwrite_first and table is None:
        # First folder in a full rebuild — create table with overwrite
        table = db.create_table(table_name, data=rows, mode="overwrite")
    else:
        # Delete old chunks for files we're re-indexing, then add new rows
        if table is not None and not overwrite_first:
            indexed_files = {r["source_file"] for r in rows}
            for sf in indexed_files:
                escaped = sf.replace("'", "''")
                table.delete(f"source_file = '{escaped}'")
            table.add(rows)
        else:
            # Table doesn't exist yet (first incremental run)
            table = _ensure_table(db, table, table_name, rows)

    # Update manifest entries
    for doc_path in files_to_index:
        rel = str(doc_path)
        key = make_key(src_id, rel)
        abs_path = source.root_path / doc_path
        if abs_path.exists() and rel in file_chunk_counts:
            manifest.record_file(key, abs_path, file_chunk_counts[rel])

    # Detect files deleted from disk (incremental mode only)
    deleted = 0
    if profile.mode == IndexMode.INCREMENTAL and not force:
        deleted = _detect_deleted_files(
            manifest, manifest_keys_for_source, current_keys_for_source, table
        )

    console.print(
        f"[green]{display}: indexed {files_indexed} files ({len(rows)} chunks)[/green]"
    )

    return (
        IndexResult(
            total_chunks=len(rows),
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            files_deleted=deleted,
            errors=errors,
        ),
        table,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _resolve_profiles(
    folder_filter: str | None,
    force: bool,
) -> list[FolderProfile]:
    """Return the list of profiles to process based on CLI arguments."""
    if folder_filter:
        # Match against display_name (basename or explicit name)
        matches = [p for p in SOURCES if p.display_name == folder_filter]
        if not matches:
            console.print(
                f"[red]Unknown source: {folder_filter}[/red]\n"
                f"[dim]Configured: {', '.join(s.display_name for s in SOURCES)}[/dim]"
            )
            return []
        return matches

    # All active (non-disabled) profiles
    return [p for p in SOURCES if p.mode != IndexMode.DISABLED]


def index_library(
    folder_filter: str | None = None,
    force: bool = False,
    table_name: str = "chunks",
) -> IndexResult:
    """Smart incremental indexing that respects per-source modes.

    Args:
        folder_filter: If set, only index the source matching this display name.
        force: If True, re-index all files regardless of mode.
        table_name: LanceDB table name.

    Returns:
        Aggregate result across all processed sources.
    """
    ensure_directories()

    profiles = _resolve_profiles(folder_filter, force)
    if not profiles:
        return IndexResult()

    manifest_path = get_manifest_path()
    manifest = Manifest.load(manifest_path)

    # Upgrade legacy v1 manifests (flat rel paths) to v2 namespaced keys.
    if manifest.version < 2 or any((not is_namespaced_key(k)) for k in manifest.files):
        console.print("[yellow]Upgrading legacy manifest to v2 (multi-source safe).[/yellow]")
        migrated = Manifest(version=2)
        roots = [Path(p.path).resolve() for p in SOURCES]
        root_ids = {r: source_id_for_root(r) for r in roots}

        for legacy_key, entry in manifest.files.items():
            if is_namespaced_key(legacy_key):
                migrated.files[legacy_key] = entry
                continue
            matches = [r for r in roots if (r / legacy_key).exists()]
            if len(matches) == 1:
                r = matches[0]
                migrated.files[make_key(root_ids[r], legacy_key)] = entry

        manifest = migrated

    before_sig = manifest_signature(manifest)

    catalog_path = get_catalog_path(DATA_DIR)
    catalog = load_catalog(catalog_path)
    index_path = LANCEDB_DIR
    index_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(index_path))

    # Check for schema version mismatch — force full rebuild if outdated
    meta = read_index_meta(index_path)
    stored_version = meta.get("schema_version", 0)
    stored_model = str(meta.get("embedding_model") or "")
    if stored_version != SCHEMA_VERSION and not force:
        console.print(
            f"[yellow]Index schema outdated (v{stored_version} → v{SCHEMA_VERSION}). "
            f"Forcing full rebuild.[/yellow]"
        )
        force = True
        folder_filter = None
        profiles = _resolve_profiles(None, force=True)

    full_rebuild = force and not folder_filter

    # Pre-compute nested exclusions
    nested_map = compute_nested_exclusions(profiles)

    # Print header
    sources_str = ", ".join(f"{p.display_name} ({p.mode.value})" for p in profiles)
    console.print(f"[cyan]Indexing sources:[/cyan] {sources_str}")
    if full_rebuild:
        console.print("[dim]Mode: full rebuild[/dim]")
    console.print()

    if full_rebuild:
        manifest = Manifest(version=2)  # start fresh
        catalog = {}

    table = None if full_rebuild else _try_open_table(db, table_name)

    aggregate = IndexResult()

    for i, profile in enumerate(profiles):
        source: ContentSource = create_source_for_profile(profile, nested_map)
        chunk_size = profile.effective_chunk_size(CHUNK_SIZE)
        chunk_overlap = profile.effective_chunk_overlap(CHUNK_OVERLAP)

        folder_result, table = _index_folder(
            source=source,
            profile=profile,
            manifest=manifest,
            db=db,
            table=table,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=force,
            overwrite_first=(full_rebuild and i == 0),
            catalog=catalog,
        )
        aggregate.accumulate(folder_result)

    # If the catalog is missing or incomplete, populate it without embedding:
    # read each document once to extract title/doc_type/tags.
    if len(catalog) < len(manifest.files):
        for profile in profiles:
            source = create_source_for_profile(profile, nested_map)
            src_id = source_id_for_root(source.root_path)
            for doc_path in source.list_documents():
                abs_path = (source.root_path / doc_path).resolve()
                if str(abs_path) in catalog:
                    continue
                try:
                    doc = source.read_document(doc_path)
                except Exception:
                    continue
                tags_str = ",".join(str(t) for t in (doc.tags or []))
                mkey = make_key(src_id, str(doc.path))
                mf = manifest.files.get(mkey)
                update_from_index_row(
                    catalog,
                    abs_path=abs_path,
                    row={
                        "source_file": str(doc.path),
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        "folder": doc.folder,
                        "source_name": doc.source_name,
                        "source_type": doc.source_type,
                        "tags": tags_str,
                    },
                    chunk_count=mf.chunk_count if mf else 0,
                    indexed_at=mf.indexed_at if mf else None,
                    manifest_key=mkey,
                )

    # Update per-document indexed_at/chunk_count from the manifest for any
    # docs we touched this run (best-effort; catalog is purely introspection).
    for abs_path_str, entry in list(catalog.items()):
        mkey = str(entry.get("manifest_key") or "")
        if not mkey:
            continue
        mf = manifest.files.get(mkey)
        if mf is None:
            continue
        entry["indexed_at"] = mf.indexed_at
        entry["chunk_count"] = mf.chunk_count

    prune_missing_files(catalog)
    save_catalog(catalog_path, catalog)

    # If the catalog is missing/empty (e.g. deleted by user), don't regress
    # meta fields on a no-op incremental run. Prefer previous meta values.
    catalog_summary = compute_catalog_summary(catalog, top_tags=50)
    if catalog_summary.get("doc_count", 0) == 0 and meta:
        catalog_summary = {
            "doc_count": meta.get("doc_count"),
            "books_count": meta.get("books_count"),
            "doc_type_counts": meta.get("doc_type_counts"),
            "top_tags": meta.get("top_tags") or [],
        }

    after_sig = manifest_signature(manifest)
    after_digest = compute_digest_from_signature(after_sig)
    prev_digest = str(meta.get("digest") or compute_digest_from_signature(before_sig))
    prev_revision = int(meta.get("revision") or 0)

    digest_changed = after_digest != prev_digest
    schema_changed = stored_version != SCHEMA_VERSION
    model_changed = stored_model and stored_model != EMBEDDING_MODEL
    state_missing = ("digest" not in meta) or ("revision" not in meta)
    bump_revision = digest_changed or schema_changed or model_changed or state_missing
    revision = prev_revision + 1 if bump_revision else prev_revision

    if bump_revision:
        delta = compute_delta_from_signatures(before_sig, after_sig)
        append_change_event(
            data_dir=DATA_DIR,
            revision=revision,
            digest=after_digest,
            delta=delta,
            schema_version=SCHEMA_VERSION,
            embedding_model=EMBEDDING_MODEL,
        )

    # Total chunks in the index (not just "this run"): sum manifest chunk counts.
    total_chunks = sum(cc for _, cc in after_sig.values())

    manifest.save(manifest_path)
    write_index_meta(
        index_path,
        extra={
            "revision": revision,
            "digest": after_digest,
            "last_indexed_at": manifest.indexed_at,
            "file_count": len(after_sig),
            "chunk_count": total_chunks,
            "doc_count": catalog_summary.get("doc_count"),
            "books_count": catalog_summary.get("books_count"),
            "doc_type_counts": catalog_summary.get("doc_type_counts"),
            "top_tags": catalog_summary.get("top_tags"),
            "files_indexed_last_run": aggregate.files_indexed,
            "files_skipped_last_run": aggregate.files_skipped,
            "files_deleted_last_run": aggregate.files_deleted,
            "errors_last_run": aggregate.errors,
        },
    )

    # Print summary
    console.print()
    console.print(
        f"[bold green]Done:[/bold green] "
        f"{aggregate.files_indexed} files indexed "
        f"({aggregate.total_chunks} chunks)"
    )
    if aggregate.files_skipped:
        console.print(f"[dim]{aggregate.files_skipped} unchanged files skipped[/dim]")
    if aggregate.files_deleted:
        console.print(
            f"[yellow]{aggregate.files_deleted} deleted files removed from index[/yellow]"
        )
    if aggregate.errors:
        console.print(f"[red]{aggregate.errors} files had errors[/red]")

    return aggregate


def reindex_file(
    file_path: Path,
    source: ContentSource | None = None,
    table_name: str = "chunks",
) -> int:
    """Reindex a single file (for incremental updates).

    Deletes existing chunks for the file before inserting new ones.

    Args:
        file_path: Absolute path to the file.
        source: Content source owning this file. If None, searches SOURCES.
        table_name: Name of the LanceDB table.

    Returns:
        Number of chunks created.
    """
    if source is None:
        # Find which source owns this file
        nested_map = compute_nested_exclusions(SOURCES)
        for profile in SOURCES:
            src_root = Path(profile.path).resolve()
            try:
                file_path.resolve().relative_to(src_root)
                source = create_source_for_profile(profile, nested_map)
                break
            except ValueError:
                continue
        if source is None:
            msg = f"File {file_path} is not under any configured source"
            raise ValueError(msg)

    rel_path = file_path.resolve().relative_to(source.root_path)
    source_doc = source.read_document(rel_path)

    tags_str = ",".join(str(t) for t in source_doc.tags) if source_doc.tags else ""
    chunks = _split_text(source_doc.content, CHUNK_SIZE, CHUNK_OVERLAP)

    rows: list[dict] = []
    for chunk_text in chunks:
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS] + "..."
        rows.append(
            {
                "text": chunk_text,
                "source_file": str(source_doc.path),
                "title": source_doc.title,
                "doc_type": source_doc.doc_type,
                "folder": source_doc.folder,
                "source_name": source_doc.source_name,
                "source_type": source_doc.source_type,
                "tags": tags_str,
            }
        )

    if not rows:
        return 0

    texts = [r["text"] for r in rows]
    vectors = embed_texts(texts)
    for row, vec in zip(rows, vectors, strict=True):
        row["vector"] = vec

    db = lancedb.connect(str(LANCEDB_DIR))
    table = db.open_table(table_name)

    escaped_path = str(source_doc.path).replace("'", "''")
    table.delete(f"source_file = '{escaped_path}'")
    table.add(rows)

    return len(rows)
