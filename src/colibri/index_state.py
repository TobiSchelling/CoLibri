"""Index state helpers (digest, revision delta, and change journal).

CoLibri's *effective capabilities* depend on what is indexed. These helpers
provide a stable, machine-readable way to detect corpus changes between runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from colibri.manifest import Manifest


def get_changes_log_path(data_dir: Path) -> Path:
    """Return the path to the index change journal (JSONL)."""
    return data_dir / "index_changes.jsonl"


def manifest_signature(manifest: Manifest) -> dict[str, tuple[str, int]]:
    """Return a compact signature for change detection.

    Signature per file is (content_hash, chunk_count).
    """
    return {
        rel: (entry.content_hash, entry.chunk_count)
        for rel, entry in manifest.files.items()
    }


def compute_digest_from_signature(sig: dict[str, tuple[str, int]]) -> str:
    """Compute a stable digest for the indexed corpus.

    Based on the manifest's per-file content hash + chunk count. The digest
    changes when indexed content changes, independent of indexing time.
    """
    h = hashlib.sha256()
    for rel in sorted(sig.keys()):
        content_hash, chunk_count = sig[rel]
        h.update(rel.encode("utf-8", errors="strict"))
        h.update(b"\t")
        h.update(content_hash.encode("utf-8", errors="strict"))
        h.update(b"\t")
        h.update(str(chunk_count).encode("ascii", errors="strict"))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def compute_digest(manifest: Manifest) -> str:
    """Compute digest from a Manifest."""
    return compute_digest_from_signature(manifest_signature(manifest))


def compute_delta(
    before: Manifest,
    after: Manifest,
) -> dict[str, list[str]]:
    """Compute (added/updated/deleted) file lists between two manifests."""
    return compute_delta_from_signatures(
        manifest_signature(before),
        manifest_signature(after),
    )


def compute_delta_from_signatures(
    before_sig: dict[str, tuple[str, int]],
    after_sig: dict[str, tuple[str, int]],
) -> dict[str, list[str]]:
    """Compute (added/updated/deleted) file lists between two signatures."""

    before_keys = set(before_sig.keys())
    after_keys = set(after_sig.keys())

    added = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)

    updated: list[str] = []
    for k in sorted(before_keys & after_keys):
        if before_sig[k] != after_sig[k]:
            updated.append(k)

    return {"added": added, "updated": updated, "deleted": deleted}


def append_change_event(
    *,
    data_dir: Path,
    revision: int,
    digest: str,
    delta: dict[str, list[str]],
    schema_version: int,
    embedding_model: str,
) -> None:
    """Append a single change event to the JSONL journal."""
    path = get_changes_log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "revision": revision,
        "digest": digest,
        "timestamp": datetime.now(UTC).isoformat(),
        "delta": delta,
        "schema_version": schema_version,
        "embedding_model": embedding_model,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True))
        f.write("\n")


def read_change_events(
    *,
    data_dir: Path,
    since_revision: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read change events from the journal, filtered by revision."""
    path = get_changes_log_path(data_dir)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(ev.get("revision", 0)) <= since_revision:
            continue
        events.append(ev)

    events.sort(key=lambda e: int(e.get("revision", 0)))
    if limit is not None:
        events = events[-limit:]
    return events
