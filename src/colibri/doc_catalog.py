"""Document catalog for fast corpus introspection.

We keep a lightweight catalog of indexed documents (one entry per file) in
``DATA_DIR/doc_catalog.json``. This lets clients discover "what is indexed"
and compute topic sketches (e.g., top tags) without scanning the vector table.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def get_catalog_path(data_dir: Path) -> Path:
    return data_dir / "doc_catalog.json"


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_catalog(path: Path, catalog: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=True), encoding="utf-8")


def prune_missing_files(catalog: dict[str, dict[str, Any]]) -> int:
    """Remove entries for files that no longer exist on disk."""
    deleted = 0
    for abs_path in list(catalog.keys()):
        try:
            if not Path(abs_path).exists():
                catalog.pop(abs_path, None)
                deleted += 1
        except OSError:
            catalog.pop(abs_path, None)
            deleted += 1
    return deleted


def _parse_tags(tags_str: str) -> list[str]:
    tags = []
    for raw in tags_str.split(","):
        t = raw.strip()
        if t:
            tags.append(t)
    return tags


def update_from_index_row(
    catalog: dict[str, dict[str, Any]],
    *,
    abs_path: Path,
    row: dict[str, Any],
    chunk_count: int,
    indexed_at: str | None,
    manifest_key: str | None = None,
) -> None:
    """Update catalog entry from an index row (one chunk).

    Uses the first chunk row of a document as its metadata source.
    """
    tags_str = str(row.get("tags") or "")
    entry: dict[str, Any] = {
        "source_file": str(row.get("source_file") or ""),
        "title": str(row.get("title") or abs_path.stem),
        "doc_type": str(row.get("doc_type") or "note"),
        "folder": str(row.get("folder") or ""),
        "source_name": str(row.get("source_name") or ""),
        "source_type": str(row.get("source_type") or ""),
        "tags": _parse_tags(tags_str),
        "chunk_count": int(chunk_count),
        "indexed_at": indexed_at,
    }
    if manifest_key:
        entry["manifest_key"] = manifest_key
    catalog[str(abs_path)] = entry


def compute_summary(
    catalog: dict[str, dict[str, Any]],
    *,
    top_tags: int = 50,
) -> dict[str, Any]:
    """Compute a lightweight corpus summary from the catalog."""
    doc_type_counts: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    for entry in catalog.values():
        dt = str(entry.get("doc_type") or "note")
        doc_type_counts[dt] += 1
        tags = entry.get("tags") or []
        if isinstance(tags, str):
            tags = _parse_tags(tags)
        for t in tags:
            if t:
                tag_counter[str(t)] += 1

    return {
        "doc_count": len(catalog),
        "doc_type_counts": dict(doc_type_counts),
        "books_count": int(doc_type_counts.get("book", 0)),
        "top_tags": [
            {"tag": tag, "document_count": count}
            for tag, count in tag_counter.most_common(max(0, top_tags))
        ],
    }
