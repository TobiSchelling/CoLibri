"""Index metadata (schema version tracking).

Tracks the schema version of the LanceDB index so that layout changes
trigger an automatic rebuild.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from colibri.config import EMBEDDING_MODEL

# Schema version — bump when column layout changes to force rebuild
SCHEMA_VERSION = 4  # Renamed table: vault_content → chunks


def read_index_meta(data_dir: Path) -> dict[str, Any]:
    """Read index metadata (schema version, creation time)."""
    meta_path = data_dir / "index_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def write_index_meta(data_dir: Path, *, extra: dict[str, Any] | None = None) -> None:
    """Write index metadata (schema version, model, revision/digest, timestamps).

    This file is intended as a lightweight machine-readable handshake for clients
    to understand what is currently indexed.
    """
    meta_path = data_dir / "index_meta.json"
    existing = read_index_meta(data_dir)

    created_at = existing.get("created_at") or datetime.now(UTC).isoformat()
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
    }
    if extra:
        meta.update(extra)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
