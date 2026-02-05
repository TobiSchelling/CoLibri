"""Tests for the manifest-based change tracking module."""

import json
import time
from pathlib import Path

from colibri.manifest import (
    FileEntry,
    Manifest,
    _compute_hash,
    get_manifest_path,
)


class TestFileEntry:
    """Tests for the FileEntry dataclass."""

    def test_creation(self) -> None:
        entry = FileEntry(
            mtime=1706600000.0,
            content_hash="sha256:abc123",
            chunk_count=42,
            indexed_at="2026-01-30T10:00:00+00:00",
        )
        assert entry.mtime == 1706600000.0
        assert entry.content_hash == "sha256:abc123"
        assert entry.chunk_count == 42
        assert entry.indexed_at == "2026-01-30T10:00:00+00:00"


class TestComputeHash:
    """Tests for SHA-256 hashing."""

    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("hello world")
        h1 = _compute_hash(f)
        h2 = _compute_hash(f)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("content A")
        f2.write_text("content B")
        assert _compute_hash(f1) != _compute_hash(f2)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("")
        h = _compute_hash(f)
        assert h.startswith("sha256:")
        assert len(h) > len("sha256:")


class TestManifestPersistence:
    """Tests for Manifest load/save."""

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        m = Manifest.load(tmp_path / "nonexistent.json")
        assert m.version == 2
        assert m.indexed_at == ""
        assert m.files == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"

        m = Manifest()
        m.files["Books/Test.md"] = FileEntry(
            mtime=1000.0,
            content_hash="sha256:abc",
            chunk_count=5,
            indexed_at="2026-01-30T00:00:00+00:00",
        )
        m.save(manifest_path)

        loaded = Manifest.load(manifest_path)
        assert len(loaded.files) == 1
        assert "Books/Test.md" in loaded.files
        entry = loaded.files["Books/Test.md"]
        assert entry.mtime == 1000.0
        assert entry.content_hash == "sha256:abc"
        assert entry.chunk_count == 5
        assert loaded.indexed_at != ""  # set by save()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "manifest.json"
        m = Manifest()
        m.save(nested)
        assert nested.exists()

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        m = Manifest()
        m.files["Notes/test.md"] = FileEntry(
            mtime=1.0, content_hash="sha256:x", chunk_count=1, indexed_at="t"
        )
        m.save(manifest_path)
        data = json.loads(manifest_path.read_text())
        assert data["version"] == 2
        assert "Notes/test.md" in data["files"]


class TestManifestChangeDetection:
    """Tests for is_file_changed and is_file_known."""

    def test_new_file_is_changed(self) -> None:
        m = Manifest()
        assert m.is_file_changed("Books/New.md", Path("/dummy")) is True

    def test_unchanged_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("content")
        stat = f.stat()

        m = Manifest()
        m.files["test.md"] = FileEntry(
            mtime=stat.st_mtime,
            content_hash=_compute_hash(f),
            chunk_count=3,
            indexed_at="t",
        )
        assert m.is_file_changed("test.md", f) is False

    def test_modified_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("original")
        original_hash = _compute_hash(f)

        m = Manifest()
        m.files["test.md"] = FileEntry(
            mtime=f.stat().st_mtime,
            content_hash=original_hash,
            chunk_count=3,
            indexed_at="t",
        )

        # Ensure mtime changes (some filesystems have 1s resolution)
        time.sleep(0.05)
        f.write_text("modified content")

        assert m.is_file_changed("test.md", f) is True

    def test_touched_but_same_content(self, tmp_path: Path) -> None:
        """mtime changed but content is identical — should not re-index."""
        f = tmp_path / "test.md"
        f.write_text("same content")
        original_mtime = f.stat().st_mtime
        content_hash = _compute_hash(f)

        m = Manifest()
        m.files["test.md"] = FileEntry(
            mtime=original_mtime - 1.0,  # stale mtime triggers hash check
            content_hash=content_hash,
            chunk_count=3,
            indexed_at="t",
        )
        # mtime differs but hash matches
        assert m.is_file_changed("test.md", f) is False

    def test_is_file_known_true(self) -> None:
        m = Manifest()
        m.files["Books/Test.md"] = FileEntry(
            mtime=1.0, content_hash="sha256:x", chunk_count=1, indexed_at="t"
        )
        assert m.is_file_known("Books/Test.md") is True

    def test_is_file_known_false(self) -> None:
        m = Manifest()
        assert m.is_file_known("Books/Missing.md") is False


class TestManifestMutation:
    """Tests for record_file, remove_file, get_folder_files."""

    def test_record_file(self, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text("hello")

        m = Manifest()
        m.record_file("Notes/note.md", f, chunk_count=7)

        assert "Notes/note.md" in m.files
        entry = m.files["Notes/note.md"]
        assert entry.mtime == f.stat().st_mtime
        assert entry.content_hash.startswith("sha256:")
        assert entry.chunk_count == 7
        assert entry.indexed_at != ""

    def test_record_file_overwrites(self, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text("v1")

        m = Manifest()
        m.record_file("Notes/note.md", f, chunk_count=3)
        f.write_text("v2")
        m.record_file("Notes/note.md", f, chunk_count=5)

        assert m.files["Notes/note.md"].chunk_count == 5

    def test_remove_file(self) -> None:
        m = Manifest()
        m.files["Books/Test.md"] = FileEntry(
            mtime=1.0, content_hash="sha256:x", chunk_count=1, indexed_at="t"
        )
        m.remove_file("Books/Test.md")
        assert "Books/Test.md" not in m.files

    def test_remove_nonexistent_is_noop(self) -> None:
        m = Manifest()
        m.remove_file("missing.md")  # should not raise

    def test_get_folder_files(self) -> None:
        m = Manifest()
        entry_a = FileEntry(mtime=1.0, content_hash="sha256:a", chunk_count=1, indexed_at="t")
        entry_b = FileEntry(mtime=2.0, content_hash="sha256:b", chunk_count=2, indexed_at="t")
        entry_c = FileEntry(mtime=3.0, content_hash="sha256:c", chunk_count=3, indexed_at="t")
        m.files["Books/A.md"] = entry_a
        m.files["Books/B.md"] = entry_b
        m.files["Notes/C.md"] = entry_c

        books = m.get_folder_files("Books")
        assert len(books) == 2
        assert "Books/A.md" in books
        assert "Books/B.md" in books
        assert "Notes/C.md" not in books

    def test_get_folder_files_namespaced(self) -> None:
        from colibri.manifest import make_key

        m = Manifest()
        sid = "0123456789ab"
        m.files[make_key(sid, "Books/A.md")] = FileEntry(
            mtime=1.0, content_hash="sha256:a", chunk_count=1, indexed_at="t"
        )
        m.files[make_key(sid, "Notes/C.md")] = FileEntry(
            mtime=1.0, content_hash="sha256:c", chunk_count=1, indexed_at="t"
        )

        books = m.get_folder_files("Books")
        assert list(books.keys()) == [make_key(sid, "Books/A.md")]

    def test_get_folder_files_empty(self) -> None:
        m = Manifest()
        assert m.get_folder_files("Books") == {}


class TestGetManifestPath:
    """Tests for the manifest path helper."""

    def test_path(self, tmp_path: Path) -> None:
        p = get_manifest_path(tmp_path)
        assert p == tmp_path / "manifest.json"
