"""Tests for incremental per-folder indexing.

These are integration tests that exercise the full indexing pipeline with
a real LanceDB table and a temporary vault, but use a mock embedding
function to avoid requiring Ollama.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import lancedb
import pytest

from colibri.config import FolderProfile, IndexMode
from colibri.index_meta import SCHEMA_VERSION, read_index_meta, write_index_meta
from colibri.indexer import (
    IndexResult,
    _build_rows_for_doc,
    _classify_files,
    _detect_deleted_files,
    _index_folder,
    _try_open_table,
)
from colibri.manifest import Manifest, make_key, source_id_for_root
from colibri.sources import MarkdownFolderSource

# Fake 4-dim vectors (real nomic-embed-text produces 768-dim)
EMBED_DIM = 4


def _fake_embed(texts: list[str], **_kwargs: object) -> list[list[float]]:
    """Deterministic fake embedding: one vector per text."""
    return [[float(i)] * EMBED_DIM for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Create a minimal vault with Books/ and Notes/ folders."""
    books = tmp_path / "Books"
    notes = tmp_path / "Notes"
    books.mkdir()
    notes.mkdir()

    (books / "BookA.md").write_text("---\ntitle: Book A\ntype: book\n---\nContent of book A.")
    (books / "BookB.md").write_text("---\ntitle: Book B\ntype: book\n---\nContent of book B.")
    (notes / "Note1.md").write_text("---\ntitle: Note One\n---\nSome note content.")

    # Ensure .colibri dirs exist
    (tmp_path / ".colibri" / "lancedb").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def books_profile(vault: Path) -> FolderProfile:
    return FolderProfile(path=str(vault / "Books"), mode=IndexMode.STATIC, doc_type="book")


@pytest.fixture()
def notes_profile(vault: Path) -> FolderProfile:
    return FolderProfile(path=str(vault / "Notes"), mode=IndexMode.INCREMENTAL, doc_type="note")


# ---------------------------------------------------------------------------
# _classify_files tests
# ---------------------------------------------------------------------------


class TestClassifyFiles:
    def test_force_returns_all(self, vault: Path, books_profile: FolderProfile) -> None:
        source = MarkdownFolderSource(books_profile.path)
        manifest = Manifest()

        sid = source_id_for_root(Path(books_profile.path))
        to_index, skipped = _classify_files(
            source, books_profile, manifest, force=True, source_id=sid
        )

        assert len(to_index) == 2
        assert skipped == 0

    def test_static_skips_known(self, vault: Path, books_profile: FolderProfile) -> None:
        source = MarkdownFolderSource(books_profile.path)
        manifest = Manifest()
        sid = source_id_for_root(Path(books_profile.path))
        # Mark BookA as known
        manifest.record_file(
            make_key(sid, "BookA.md"),
            vault / "Books" / "BookA.md",
            chunk_count=1,
        )

        to_index, skipped = _classify_files(
            source, books_profile, manifest, force=False, source_id=sid
        )

        assert len(to_index) == 1  # only BookB
        assert skipped == 1
        assert to_index[0].name == "BookB.md"

    def test_incremental_skips_unchanged(self, vault: Path, notes_profile: FolderProfile) -> None:
        source = MarkdownFolderSource(notes_profile.path)
        manifest = Manifest()
        sid = source_id_for_root(Path(notes_profile.path))
        manifest.record_file(
            make_key(sid, "Note1.md"),
            vault / "Notes" / "Note1.md",
            chunk_count=1,
        )

        to_index, skipped = _classify_files(
            source, notes_profile, manifest, force=False, source_id=sid
        )

        assert len(to_index) == 0
        assert skipped == 1

    def test_incremental_detects_change(self, vault: Path, notes_profile: FolderProfile) -> None:
        source = MarkdownFolderSource(notes_profile.path)
        manifest = Manifest()
        sid = source_id_for_root(Path(notes_profile.path))
        manifest.record_file(
            make_key(sid, "Note1.md"),
            vault / "Notes" / "Note1.md",
            chunk_count=1,
        )

        # Modify file
        time.sleep(0.05)
        (vault / "Notes" / "Note1.md").write_text("---\ntitle: Note One\n---\nUpdated.")

        to_index, skipped = _classify_files(
            source, notes_profile, manifest, force=False, source_id=sid
        )

        assert len(to_index) == 1
        assert skipped == 0

    def test_append_only_skips_known(self, vault: Path) -> None:
        notes_path = str(vault / "Notes")
        profile = FolderProfile(path=notes_path, mode=IndexMode.APPEND_ONLY, doc_type="journal")
        source = MarkdownFolderSource(notes_path)
        manifest = Manifest()
        sid = source_id_for_root(Path(notes_path))
        manifest.record_file(
            make_key(sid, "Note1.md"),
            vault / "Notes" / "Note1.md",
            chunk_count=1,
        )

        to_index, skipped = _classify_files(source, profile, manifest, force=False, source_id=sid)

        assert len(to_index) == 0
        assert skipped == 1


# ---------------------------------------------------------------------------
# _build_rows_for_doc tests
# ---------------------------------------------------------------------------


class TestBuildRowsForDoc:
    def test_uses_frontmatter_type(self, vault: Path) -> None:
        """When frontmatter has type: book, profile doc_type is ignored."""
        books_path = str(vault / "Books")
        source = MarkdownFolderSource(books_path)
        profile = FolderProfile(path=books_path, doc_type="documentation")

        rows = _build_rows_for_doc(
            source,
            Path("BookA.md"),
            profile,
            chunk_size=3000,
            chunk_overlap=200,
        )

        assert rows[0]["doc_type"] == "book"  # from frontmatter

    def test_uses_profile_type_as_fallback(self, vault: Path) -> None:
        """When frontmatter has no type, profile doc_type is used."""
        notes_path = str(vault / "Notes")
        source = MarkdownFolderSource(notes_path)
        profile = FolderProfile(path=notes_path, doc_type="journal")

        rows = _build_rows_for_doc(
            source,
            Path("Note1.md"),
            profile,
            chunk_size=3000,
            chunk_overlap=200,
        )

        assert rows[0]["doc_type"] == "journal"  # from profile


# ---------------------------------------------------------------------------
# _detect_deleted_files tests
# ---------------------------------------------------------------------------


class TestDetectDeletedFiles:
    def test_removes_stale_entries(self, vault: Path) -> None:
        manifest = Manifest()
        sid = source_id_for_root(vault / "Books")
        manifest.record_file(make_key(sid, "BookA.md"), vault / "Books" / "BookA.md", chunk_count=1)
        manifest.record_file(make_key(sid, "Deleted.md"), vault / "Books" / "BookA.md", chunk_count=1)

        current_files = {make_key(sid, "BookA.md"), make_key(sid, "BookB.md")}

        # Create a real LanceDB table for deletion
        db = lancedb.connect(str(vault / ".colibri" / "lancedb"))
        table = db.create_table(
            "test_del",
            data=[
                {
                    "vector": [0.0] * EMBED_DIM,
                    "text": "x",
                    "source_file": "Deleted.md",
                    "title": "t",
                    "doc_type": "book",
                    "folder": "",
                    "source_name": "Books",
                    "source_type": "markdown",
                    "tags": "",
                },
            ],
        )

        # known_files = all manifest keys belonging to this source
        known_files = {make_key(sid, "BookA.md"), make_key(sid, "Deleted.md")}
        deleted = _detect_deleted_files(manifest, known_files, current_files, table)

        assert deleted == 1
        assert not manifest.is_file_known(make_key(sid, "Deleted.md"))

    def test_no_false_positives(self, vault: Path) -> None:
        manifest = Manifest()
        sid = source_id_for_root(vault / "Books")
        manifest.record_file(make_key(sid, "BookA.md"), vault / "Books" / "BookA.md", chunk_count=1)

        current_files = {make_key(sid, "BookA.md"), make_key(sid, "BookB.md")}
        known_files = {make_key(sid, "BookA.md")}
        deleted = _detect_deleted_files(manifest, known_files, current_files, None)
        assert deleted == 0

    def test_none_table_returns_zero(self) -> None:
        manifest = Manifest()
        assert _detect_deleted_files(manifest, set(), set(), None) == 0


# ---------------------------------------------------------------------------
# _index_folder integration tests
# ---------------------------------------------------------------------------


class TestIndexFolder:
    """Integration tests with real LanceDB and mock embeddings."""

    def _run_index_folder(
        self,
        vault: Path,
        profile: FolderProfile,
        manifest: Manifest,
        table: lancedb.table.Table | None = None,
        force: bool = False,
        overwrite_first: bool = False,
    ) -> tuple[IndexResult, lancedb.table.Table | None]:
        db = lancedb.connect(str(vault / ".colibri" / "lancedb"))
        source = MarkdownFolderSource(profile.path)

        with patch("colibri.indexer.embed_texts", side_effect=_fake_embed):
            return _index_folder(
                source=source,
                profile=profile,
                manifest=manifest,
                db=db,
                table=table,
                table_name="test_vault",
                chunk_size=3000,
                chunk_overlap=200,
                force=force,
                overwrite_first=overwrite_first,
            )

    def test_first_run_indexes_everything(self, vault: Path, books_profile: FolderProfile) -> None:
        manifest = Manifest()
        result, table = self._run_index_folder(vault, books_profile, manifest, overwrite_first=True)

        assert result.files_indexed == 2
        assert result.total_chunks >= 2
        assert table is not None
        assert len(manifest.files) == 2

    def test_static_skips_after_first_run(self, vault: Path, books_profile: FolderProfile) -> None:
        manifest = Manifest()

        # First run: indexes everything
        result1, table = self._run_index_folder(
            vault, books_profile, manifest, overwrite_first=True
        )
        assert result1.files_indexed == 2

        # Second run: static mode skips all
        result2, _ = self._run_index_folder(vault, books_profile, manifest, table=table)
        assert result2.files_indexed == 0
        assert result2.files_skipped == 2

    def test_incremental_skips_unchanged(self, vault: Path, notes_profile: FolderProfile) -> None:
        manifest = Manifest()

        result1, table = self._run_index_folder(
            vault, notes_profile, manifest, overwrite_first=True
        )
        assert result1.files_indexed == 1

        result2, _ = self._run_index_folder(vault, notes_profile, manifest, table=table)
        assert result2.files_indexed == 0
        assert result2.files_skipped == 1

    def test_incremental_reindexes_changed(self, vault: Path, notes_profile: FolderProfile) -> None:
        manifest = Manifest()
        result1, table = self._run_index_folder(
            vault, notes_profile, manifest, overwrite_first=True
        )
        assert result1.files_indexed == 1

        # Modify file
        time.sleep(0.05)
        (vault / "Notes" / "Note1.md").write_text("---\ntitle: Note One\n---\nUpdated content.")

        result2, _ = self._run_index_folder(vault, notes_profile, manifest, table=table)
        assert result2.files_indexed == 1
        assert result2.files_skipped == 0

    def test_force_reindexes_everything(self, vault: Path, books_profile: FolderProfile) -> None:
        manifest = Manifest()
        result1, table = self._run_index_folder(
            vault, books_profile, manifest, overwrite_first=True
        )

        result2, _ = self._run_index_folder(vault, books_profile, manifest, table=table, force=True)
        assert result2.files_indexed == 2  # re-indexed despite static mode

    def test_disabled_skips_folder(self, vault: Path) -> None:
        """Disabled folders produce no files from _classify_files."""
        books_path = str(vault / "Books")
        profile = FolderProfile(path=books_path, mode=IndexMode.DISABLED, doc_type="book")
        source = MarkdownFolderSource(books_path)
        manifest = Manifest()

        # _classify_files with disabled mode returns nothing (force=False path)
        sid = source_id_for_root(Path(books_path))
        to_index, skipped = _classify_files(source, profile, manifest, force=False, source_id=sid)
        assert len(to_index) == 0
        assert skipped == 0

    def test_deleted_files_removed(self, vault: Path, notes_profile: FolderProfile) -> None:
        manifest = Manifest()
        result1, table = self._run_index_folder(
            vault, notes_profile, manifest, overwrite_first=True
        )
        assert result1.files_indexed == 1

        # Delete the file from disk
        (vault / "Notes" / "Note1.md").unlink()

        result2, _ = self._run_index_folder(vault, notes_profile, manifest, table=table)
        assert result2.files_deleted == 1
        assert not manifest.is_file_known("Note1.md")

    def test_append_only_indexes_new_only(self, vault: Path) -> None:
        notes_path = str(vault / "Notes")
        profile = FolderProfile(path=notes_path, mode=IndexMode.APPEND_ONLY, doc_type="journal")
        manifest = Manifest()

        result1, table = self._run_index_folder(vault, profile, manifest, overwrite_first=True)
        assert result1.files_indexed == 1

        # Add a new file
        (vault / "Notes" / "Note2.md").write_text("---\ntitle: Note Two\n---\nNew note.")

        result2, _ = self._run_index_folder(vault, profile, manifest, table=table)
        assert result2.files_indexed == 1  # only the new file
        assert result2.files_skipped == 1  # existing file skipped


# ---------------------------------------------------------------------------
# IndexResult tests
# ---------------------------------------------------------------------------


class TestIndexResult:
    def test_defaults(self) -> None:
        r = IndexResult()
        assert r.total_chunks == 0
        assert r.files_indexed == 0

    def test_accumulate(self) -> None:
        a = IndexResult(total_chunks=10, files_indexed=3, files_skipped=1)
        b = IndexResult(total_chunks=5, files_indexed=2, errors=1)
        a.accumulate(b)

        assert a.total_chunks == 15
        assert a.files_indexed == 5
        assert a.files_skipped == 1
        assert a.errors == 1


# ---------------------------------------------------------------------------
# _try_open_table tests
# ---------------------------------------------------------------------------


class TestTryOpenTable:
    def test_returns_none_when_missing(self, vault: Path) -> None:
        db = lancedb.connect(str(vault / ".colibri" / "lancedb"))
        assert _try_open_table(db, "nonexistent") is None

    def test_returns_table_when_exists(self, vault: Path) -> None:
        db = lancedb.connect(str(vault / ".colibri" / "lancedb"))
        db.create_table("existing", data=[{"vector": [0.0] * EMBED_DIM, "text": "x"}])
        table = _try_open_table(db, "existing")
        assert table is not None


# ---------------------------------------------------------------------------
# Index metadata (schema versioning) tests
# ---------------------------------------------------------------------------


class TestIndexMeta:
    def test_read_missing_returns_empty(self, tmp_path: Path) -> None:
        assert read_index_meta(tmp_path) == {}

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        write_index_meta(tmp_path)
        meta = read_index_meta(tmp_path)

        assert meta["schema_version"] == SCHEMA_VERSION
        assert "created_at" in meta
        assert "embedding_model" in meta

    def test_write_creates_file(self, tmp_path: Path) -> None:
        write_index_meta(tmp_path)
        assert (tmp_path / "index_meta.json").exists()

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        # Write old meta
        meta_path = tmp_path / "index_meta.json"
        meta_path.write_text('{"schema_version": 1}')

        write_index_meta(tmp_path)
        meta = read_index_meta(tmp_path)
        assert meta["schema_version"] == SCHEMA_VERSION
