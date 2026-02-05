"""Manifest-based change tracking for incremental indexing.

Tracks which files have been indexed, their modification times, and
content hashes.  The manifest is stored as JSON in the data directory
(default: ``~/.local/share/colibri/manifest.json``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re


@dataclass
class FileEntry:
    """Tracking state for a single indexed file."""

    mtime: float
    content_hash: str
    chunk_count: int
    indexed_at: str  # ISO 8601


@dataclass
class Manifest:
    """Tracks which files have been indexed and their state.

    On first run the manifest file does not exist --- ``load()`` returns
    an empty instance so every file is treated as new.
    """

    version: int = 2
    indexed_at: str = ""
    files: dict[str, FileEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, manifest_path: Path) -> Manifest:
        """Load manifest from disk, or return an empty one if missing."""
        if not manifest_path.exists():
            return cls()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {path: FileEntry(**entry) for path, entry in data.get("files", {}).items()}
        return cls(
            version=data.get("version", 1),
            indexed_at=data.get("indexed_at", ""),
            files=files,
        )

    def save(self, manifest_path: Path) -> None:
        """Persist manifest to disk."""
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        indexed_at = datetime.now(UTC).isoformat()
        # Keep in-memory state consistent with what's written to disk.
        self.indexed_at = indexed_at
        data = {
            "version": self.version,
            "indexed_at": indexed_at,
            "files": {path: asdict(entry) for path, entry in self.files.items()},
        }
        manifest_path.write_text(
            json.dumps(data, indent=2, sort_keys=False, ensure_ascii=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_file_changed(self, rel_path: str, abs_path: Path) -> bool:
        """Check if a file has changed since last indexing.

        Compares mtime first (fast).  Falls back to content hash only
        when mtime differs --- avoids hashing every file on every run.
        """
        entry = self.files.get(rel_path)
        if entry is None:
            return True  # new file
        stat = abs_path.stat()
        if stat.st_mtime != entry.mtime:
            current_hash = _compute_hash(abs_path)
            return current_hash != entry.content_hash
        return False

    def is_file_known(self, rel_path: str) -> bool:
        """Return True if the file path exists in the manifest."""
        return rel_path in self.files

    def get_folder_files(self, folder: str) -> dict[str, FileEntry]:
        """Return all manifest entries whose *relative path* starts with ``folder/``.

        Supports both legacy v1 keys (plain rel paths) and v2 namespaced keys
        (``<source_id>:<rel_path>``).
        """
        prefix = f"{folder}/"
        out: dict[str, FileEntry] = {}
        for k, entry in self.files.items():
            rel = k
            if is_namespaced_key(k):
                _, rel = split_key(k)
            if rel.startswith(prefix):
                out[k] = entry
        return out

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def record_file(
        self,
        rel_path: str,
        abs_path: Path,
        chunk_count: int,
    ) -> None:
        """Record a file as indexed (or update an existing entry)."""
        stat = abs_path.stat()
        self.files[rel_path] = FileEntry(
            mtime=stat.st_mtime,
            content_hash=_compute_hash(abs_path),
            chunk_count=chunk_count,
            indexed_at=datetime.now(UTC).isoformat(),
        )

    def remove_file(self, rel_path: str) -> None:
        """Remove a file entry from the manifest."""
        self.files.pop(rel_path, None)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _compute_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def get_manifest_path(data_dir: Path | None = None) -> Path:
    """Return the canonical manifest file location.

    Args:
        data_dir: Directory for CoLibri data files.
                  Defaults to ``DATA_DIR`` from config.
    """
    if data_dir is None:
        from colibri.config import DATA_DIR

        data_dir = DATA_DIR
    return data_dir / "manifest.json"


# ---------------------------------------------------------------------------
# Key helpers (v2): namespace keys by source_id so multiple sources are safe.
# ---------------------------------------------------------------------------


_KEY_RE = re.compile(r"^[0-9a-f]{12}:.+")


def is_namespaced_key(key: str) -> bool:
    return bool(_KEY_RE.match(key))


def source_id_for_root(root: Path) -> str:
    """Stable 12-hex source identifier for a given source root."""
    h = hashlib.sha256(str(root.resolve()).encode("utf-8", errors="strict"))
    return h.hexdigest()[:12]


def make_key(source_id: str, rel_path: str) -> str:
    return f"{source_id}:{rel_path}"


def split_key(key: str) -> tuple[str, str]:
    source_id, rel = key.split(":", 1)
    return source_id, rel
