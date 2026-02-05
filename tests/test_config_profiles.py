"""Tests for per-folder source profiles in configuration."""

import pytest

from colibri.config import FolderProfile, IndexMode, _parse_sources


class TestIndexMode:
    """Tests for the IndexMode enum."""

    def test_values(self) -> None:
        assert IndexMode.STATIC.value == "static"
        assert IndexMode.INCREMENTAL.value == "incremental"
        assert IndexMode.APPEND_ONLY.value == "append_only"
        assert IndexMode.DISABLED.value == "disabled"

    def test_from_string(self) -> None:
        assert IndexMode("static") is IndexMode.STATIC
        assert IndexMode("incremental") is IndexMode.INCREMENTAL
        assert IndexMode("append_only") is IndexMode.APPEND_ONLY
        assert IndexMode("disabled") is IndexMode.DISABLED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            IndexMode("bogus")


class TestFolderProfile:
    """Tests for the FolderProfile dataclass."""

    def test_defaults(self) -> None:
        p = FolderProfile(path="/tmp/Notes")
        assert p.path == "/tmp/Notes"
        assert p.mode is IndexMode.INCREMENTAL
        assert p.doc_type == "note"
        assert p.chunk_size is None
        assert p.chunk_overlap is None
        assert p.extensions == (".md",)
        assert p.name is None

    def test_display_name_from_path(self) -> None:
        p = FolderProfile(path="/tmp/Notes")
        assert p.display_name == "Notes"

    def test_display_name_explicit(self) -> None:
        p = FolderProfile(path="/tmp/Notes", name="My Notes")
        assert p.display_name == "My Notes"

    def test_explicit_fields(self) -> None:
        p = FolderProfile(
            path="/tmp/Books",
            mode=IndexMode.STATIC,
            doc_type="book",
            chunk_size=2000,
            chunk_overlap=300,
            extensions=(".md", ".yaml"),
            name="Library Books",
        )
        assert p.path == "/tmp/Books"
        assert p.mode is IndexMode.STATIC
        assert p.doc_type == "book"
        assert p.chunk_size == 2000
        assert p.chunk_overlap == 300
        assert p.extensions == (".md", ".yaml")
        assert p.display_name == "Library Books"

    def test_frozen(self) -> None:
        p = FolderProfile(path="/tmp/Notes")
        with pytest.raises(AttributeError):
            p.path = "/tmp/Other"  # type: ignore[misc]

    def test_effective_chunk_size_with_override(self) -> None:
        p = FolderProfile(path="/tmp/Docs", chunk_size=1500)
        assert p.effective_chunk_size(default=3000) == 1500

    def test_effective_chunk_size_without_override(self) -> None:
        p = FolderProfile(path="/tmp/Docs")
        assert p.effective_chunk_size(default=3000) == 3000

    def test_effective_chunk_overlap_with_override(self) -> None:
        p = FolderProfile(path="/tmp/Docs", chunk_overlap=400)
        assert p.effective_chunk_overlap(default=200) == 400

    def test_effective_chunk_overlap_without_override(self) -> None:
        p = FolderProfile(path="/tmp/Docs")
        assert p.effective_chunk_overlap(default=200) == 200


class TestParseSources:
    """Tests for _parse_sources() config parsing."""

    def test_minimal_entry(self) -> None:
        profiles = _parse_sources([{"path": "/tmp/Notes"}])
        assert len(profiles) == 1
        assert profiles[0].path == "/tmp/Notes"
        assert profiles[0].mode is IndexMode.INCREMENTAL
        assert profiles[0].doc_type == "note"

    def test_full_entry(self) -> None:
        profiles = _parse_sources(
            [
                {
                    "path": "/tmp/Books",
                    "mode": "static",
                    "doc_type": "book",
                    "chunk_size": 2000,
                    "chunk_overlap": 300,
                    "extensions": [".md", ".yaml"],
                    "name": "My Books",
                }
            ]
        )
        p = profiles[0]
        assert p.path == "/tmp/Books"
        assert p.mode is IndexMode.STATIC
        assert p.doc_type == "book"
        assert p.chunk_size == 2000
        assert p.chunk_overlap == 300
        assert p.extensions == (".md", ".yaml")
        assert p.display_name == "My Books"

    def test_multiple_entries(self) -> None:
        profiles = _parse_sources(
            [
                {"path": "/tmp/Books", "mode": "static", "doc_type": "book"},
                {"path": "/tmp/Notes", "mode": "incremental"},
                {"path": "/tmp/Drafts", "mode": "disabled"},
            ]
        )
        assert len(profiles) == 3
        assert profiles[0].mode is IndexMode.STATIC
        assert profiles[1].mode is IndexMode.INCREMENTAL
        assert profiles[2].mode is IndexMode.DISABLED

    def test_all_modes(self) -> None:
        profiles = _parse_sources(
            [
                {"path": "/tmp/A", "mode": "static"},
                {"path": "/tmp/B", "mode": "incremental"},
                {"path": "/tmp/C", "mode": "append_only"},
                {"path": "/tmp/D", "mode": "disabled"},
            ]
        )
        assert [p.mode for p in profiles] == [
            IndexMode.STATIC,
            IndexMode.INCREMENTAL,
            IndexMode.APPEND_ONLY,
            IndexMode.DISABLED,
        ]

    def test_active_sources_excludes_disabled(self) -> None:
        """Verify disabled sources are excluded from active list."""
        profiles = _parse_sources(
            [
                {"path": "/tmp/Books", "mode": "static"},
                {"path": "/tmp/Notes", "mode": "incremental"},
                {"path": "/tmp/Drafts", "mode": "disabled"},
            ]
        )
        active = [s.display_name for s in profiles if s.mode != IndexMode.DISABLED]
        assert active == ["Books", "Notes"]
        assert "Drafts" not in active

    def test_empty_list(self) -> None:
        profiles = _parse_sources([])
        assert profiles == []
