"""Configuration for the CoLibri RAG system.

Configuration is loaded from ~/.config/colibri/config.yaml
Environment variables can override config file settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

# Config file location
CONFIG_DIR = Path.home() / ".config" / "colibri"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Per-folder source profiles
# ---------------------------------------------------------------------------


class IndexMode(Enum):
    """How a folder should be indexed."""

    STATIC = "static"
    INCREMENTAL = "incremental"
    APPEND_ONLY = "append_only"
    DISABLED = "disabled"


@dataclass(frozen=True)
class FolderProfile:
    """Per-source indexing configuration.

    Each source is identified by an absolute ``path`` to its root directory.
    """

    path: str  # Absolute path (required)
    mode: IndexMode = IndexMode.INCREMENTAL
    doc_type: str = "note"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    extensions: tuple[str, ...] = (".md",)
    name: str | None = None  # Display name (default: path basename)

    @property
    def display_name(self) -> str:
        return self.name or Path(self.path).name

    def effective_chunk_size(self, default: int) -> int:
        """Return per-folder override or the global default."""
        return self.chunk_size if self.chunk_size is not None else default

    def effective_chunk_overlap(self, default: int) -> int:
        """Return per-folder override or the global default."""
        return self.chunk_overlap if self.chunk_overlap is not None else default


def _parse_sources(raw_sources: list[dict]) -> list[FolderProfile]:
    """Convert raw YAML source dicts into FolderProfile objects."""
    profiles: list[FolderProfile] = []
    for src in raw_sources:
        raw_ext = src.get("extensions")
        extensions = tuple(raw_ext) if raw_ext else (".md",)
        profiles.append(
            FolderProfile(
                path=src["path"],
                mode=IndexMode(src.get("mode", "incremental")),
                doc_type=src.get("doc_type", "note"),
                chunk_size=src.get("chunk_size"),
                chunk_overlap=src.get("chunk_overlap"),
                extensions=extensions,
                name=src.get("name"),
            )
        )
    return profiles


def get_books_source(sources: list[FolderProfile] | None = None) -> FolderProfile | None:
    """Return the first source with doc_type='book', or None."""
    for s in (sources or SOURCES):
        if s.doc_type == "book":
            return s
    return None


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "sources": [
        {
            "path": str(Path.home() / "Documents" / "CoLibri" / "Books"),
            "mode": "static",
            "doc_type": "book",
        },
    ],
    "data": {
        "directory": None,  # None = XDG default (~/.local/share/colibri)
    },
    "index": {
        "directory": "lancedb",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
    },
    "retrieval": {
        "top_k": 10,
        "similarity_threshold": 0.3,
    },
    "chunking": {
        "chunk_size": 3000,
        "chunk_overlap": 200,
    },
    "translation": {
        "model": None,  # Auto-detect from Ollama if None
    },
}


def load_config() -> dict:
    """Load configuration from YAML file, with defaults as fallback."""
    config = _DEFAULTS.copy()

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into defaults
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values

    return config


def get_config_path() -> Path:
    """Return the config file path."""
    return CONFIG_FILE


# Load config on module import
_config = load_config()

# Per-source profiles (top-level "sources" key)
SOURCES: list[FolderProfile] = _parse_sources(
    _config.get("sources", _DEFAULTS["sources"])
)

# Derive LIBRARY_PATH: parent of book source, or parent of first source
_books_src = get_books_source(SOURCES)
_default_library = Path(_books_src.path).parent if _books_src else (
    Path(SOURCES[0].path).parent if SOURCES else Path.home() / "Documents" / "CoLibri"
)

LIBRARY_PATH = Path(
    os.environ.get("COLIBRI_LIBRARY_PATH", str(_default_library))
)

# Data directory (separate from library content)
_data_dir_override = os.environ.get("COLIBRI_DATA_DIR") or _config.get("data", {}).get("directory")
if _data_dir_override:
    DATA_DIR = Path(_data_dir_override)
else:
    _xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    DATA_DIR = Path(_xdg) / "colibri"

# LanceDB index location (relative to data directory)
LANCEDB_DIR = DATA_DIR / _config["index"]["directory"]

# Ollama settings
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", _config["ollama"]["base_url"])
EMBEDDING_MODEL = os.environ.get("COLIBRI_EMBEDDING_MODEL", _config["ollama"]["embedding_model"])

# Retrieval settings
TOP_K: int = _config["retrieval"]["top_k"]
SIMILARITY_THRESHOLD: float = _config["retrieval"]["similarity_threshold"]

# Chunking settings
CHUNK_SIZE: int = _config["chunking"]["chunk_size"]
CHUNK_OVERLAP: int = _config["chunking"]["chunk_overlap"]

# Translation settings
TRANSLATION_MODEL: str | None = _config.get("translation", {}).get("model")


def ensure_directories() -> None:
    """Ensure required directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config mutation helpers (used by TUI and Web config UIs)
# ---------------------------------------------------------------------------


def load_raw_config() -> dict:
    """Load raw YAML config dict from disk (or defaults if missing)."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_raw_config(config: dict) -> None:
    """Write a full config dict back to YAML."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_sources_raw() -> list[dict]:
    """Return the raw sources list from the config file."""
    cfg = load_raw_config()
    return cfg.get("sources", _DEFAULTS["sources"])


def set_sources_raw(sources: list[dict]) -> None:
    """Update the sources list in the config file and save.

    Merges into existing config so other sections are preserved.
    """
    cfg = load_raw_config()
    cfg["sources"] = sources
    save_raw_config(cfg)
