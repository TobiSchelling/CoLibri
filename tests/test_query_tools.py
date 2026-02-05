"""Tests for enriched list_books, get_book_outline, and browse_topics.

Uses a temp library with real LanceDB tables and fake embeddings,
following the fixture pattern from test_indexer_incremental.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import lancedb
import pytest

from colibri.index_meta import write_index_meta
from colibri.query import SearchEngine
from colibri.sources import MarkdownFolderSource

# Fake 4-dim vectors (consistent with test_indexer_incremental.py)
EMBED_DIM = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_book(path: Path, title: str, content: str, **extra_meta: object) -> None:
    """Write a markdown file with frontmatter."""
    lines = [f"title: {title}", "type: book"]
    for key, value in extra_meta.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    header = "\n".join(lines)
    path.write_text(f"---\n{header}\n---\n{content}")


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Create a vault with books and notes containing realistic frontmatter."""
    books = tmp_path / "Books"
    notes = tmp_path / "Notes"
    books.mkdir()
    notes.mkdir()

    _make_book(
        books / "Clean Architecture.md",
        title="Clean Architecture",
        content=(
            "# Part I: Introduction\n\n"
            "Architecture matters.\n\n"
            "## Chapter 1: Design and Architecture\n\n"
            "The goal of software architecture is to minimize human resources.\n\n"
            "## Chapter 2: A Tale of Two Values\n\n"
            "Every software system provides two values.\n\n"
            "# Part II: Programming Paradigms\n\n"
            "## Chapter 3: Paradigm Overview\n\n"
            "Three paradigms.\n\n"
            "### Structured Programming\n\nDijkstra.\n"
        ),
        author="Robert C. Martin",
        language="en",
        tags=["architecture", "design", "book"],
    )

    _make_book(
        books / "Effective Software Architectures.md",
        title="Effective Software Architectures",
        content=(
            "# Introduction\n\n"
            "This is the translated edition.\n\n"
            "## Chapter 1: Basics\n\nFundamentals of architecture.\n"
        ),
        author="Gernot Starke",
        language="de",
        tags=["architecture", "book", "translated"],
        original_title="Effektive Softwarearchitekturen",
        translated_from="de",
    )

    (notes / "Meeting Notes.md").write_text(
        "---\ntitle: Meeting Notes\ntags:\n  - meeting\n  - architecture\n---\n"
        "Discussed architecture patterns.\n"
    )

    return tmp_path


def _build_index(vault: Path) -> Path:
    """Build a LanceDB index in a .colibri directory and return the path."""
    index_dir = vault / ".colibri" / "lancedb"
    index_dir.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(index_dir))
    rows = [
        # Clean Architecture — 3 chunks
        {
            "vector": [1.0] * EMBED_DIM,
            "text": "Architecture matters.",
            "source_file": "Clean Architecture.md",
            "title": "Clean Architecture",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "architecture,design,book",
        },
        {
            "vector": [2.0] * EMBED_DIM,
            "text": "The goal of software architecture.",
            "source_file": "Clean Architecture.md",
            "title": "Clean Architecture",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "architecture,design,book",
        },
        {
            "vector": [3.0] * EMBED_DIM,
            "text": "Three paradigms.",
            "source_file": "Clean Architecture.md",
            "title": "Clean Architecture",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "architecture,design,book",
        },
        # Effective Software Architectures — 2 chunks
        {
            "vector": [4.0] * EMBED_DIM,
            "text": "This is the translated edition.",
            "source_file": "Effective Software Architectures.md",
            "title": "Effective Software Architectures",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "architecture,book,translated",
        },
        {
            "vector": [5.0] * EMBED_DIM,
            "text": "Fundamentals of architecture.",
            "source_file": "Effective Software Architectures.md",
            "title": "Effective Software Architectures",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "architecture,book,translated",
        },
        # Meeting Notes — 1 chunk (in Notes source)
        {
            "vector": [6.0] * EMBED_DIM,
            "text": "Discussed architecture patterns.",
            "source_file": "Meeting Notes.md",
            "title": "Meeting Notes",
            "doc_type": "note",
            "folder": "",
            "source_name": "Notes",
            "source_type": "markdown",
            "tags": "meeting,architecture",
        },
    ]
    db.create_table("chunks", data=rows, mode="overwrite")
    write_index_meta(index_dir)
    return index_dir


@pytest.fixture()
def engine(vault: Path):
    """Create a SearchEngine backed by fake index data.

    Patches LANCEDB_DIR and SOURCES for the full duration of the test.
    """
    index_dir = _build_index(vault)
    # Create a source that spans both Books and Notes (for get_note / get_linked_notes)
    books_source = MarkdownFolderSource(vault / "Books", name="Books")
    notes_source = MarkdownFolderSource(vault / "Notes", name="Notes")

    with patch("colibri.query.LANCEDB_DIR", index_dir):
        eng = SearchEngine(source=books_source)
        # Add notes source so get_note can resolve notes too
        eng._all_sources.append(notes_source)
        yield eng


# ---------------------------------------------------------------------------
# list_books tests
# ---------------------------------------------------------------------------


class TestListBooksEnriched:
    def test_returns_enriched_metadata(self, engine: SearchEngine) -> None:
        books = engine.list_books()
        assert len(books) == 2

        by_title = {b["title"]: b for b in books}

        clean = by_title["Clean Architecture"]
        assert clean["chunks"] == 3
        assert clean["file"] == "Clean Architecture.md"
        assert clean["author"] == "Robert C. Martin"
        assert clean["language"] == "en"
        assert "architecture" in clean["tags"]
        assert "design" in clean["tags"]

        effective = by_title["Effective Software Architectures"]
        assert effective["chunks"] == 2
        assert effective["author"] == "Gernot Starke"
        assert effective["language"] == "de"
        assert effective["original_title"] == "Effektive Softwarearchitekturen"
        assert effective["translated_from"] == "de"
        assert "translated" in effective["tags"]

    def test_handles_missing_source_file(self, engine: SearchEngine) -> None:
        """When the source .md file is deleted, index-only fields still work."""
        # Remove one book from disk
        source_path = engine.source.root_path / "Clean Architecture.md"
        source_path.unlink()

        books = engine.list_books()
        by_title = {b["title"]: b for b in books}

        # Index-only fields still present
        clean = by_title["Clean Architecture"]
        assert clean["chunks"] == 3
        assert clean["file"] == "Clean Architecture.md"
        # Frontmatter fields absent
        assert "author" not in clean

        # Other book still enriched
        effective = by_title["Effective Software Architectures"]
        assert effective["author"] == "Gernot Starke"

    def test_empty_books_folder(self, vault: Path) -> None:
        """Returns empty list when no books are indexed."""
        index_dir = vault / ".colibri" / "lancedb_empty"
        index_dir.mkdir(parents=True)

        db = lancedb.connect(str(index_dir))
        db.create_table(
            "chunks",
            data=[{
                "vector": [0.0] * EMBED_DIM,
                "text": "A note.",
                "source_file": "Note.md",
                "title": "Note",
                "doc_type": "note",
                "folder": "",
                "source_name": "Notes",
                "source_type": "markdown",
                "tags": "",
            }],
        )
        write_index_meta(index_dir)

        source = MarkdownFolderSource(vault / "Books", name="Books")
        with patch("colibri.query.LANCEDB_DIR", index_dir):
            eng = SearchEngine(source=source)

        assert eng.list_books() == []


# ---------------------------------------------------------------------------
# get_book_outline tests
# ---------------------------------------------------------------------------


class TestGetBookOutline:
    def test_returns_headings(self, engine: SearchEngine) -> None:
        outline = engine.get_book_outline("Clean Architecture.md")
        assert outline is not None
        assert len(outline) == 6

        # Check hierarchy
        assert outline[0] == {"level": 1, "text": "Part I: Introduction"}
        assert outline[1] == {"level": 2, "text": "Chapter 1: Design and Architecture"}
        assert outline[2] == {"level": 2, "text": "Chapter 2: A Tale of Two Values"}
        assert outline[3] == {"level": 1, "text": "Part II: Programming Paradigms"}
        assert outline[4] == {"level": 2, "text": "Chapter 3: Paradigm Overview"}
        assert outline[5] == {"level": 3, "text": "Structured Programming"}

    def test_nonexistent_file_returns_none(self, engine: SearchEngine) -> None:
        assert engine.get_book_outline("Nonexistent.md") is None

    def test_file_with_no_headings(self, engine: SearchEngine) -> None:
        """A file with content but no markdown headings returns empty list."""
        flat = engine.source.root_path / "Flat.md"
        flat.write_text("---\ntitle: Flat Book\ntype: book\n---\nJust paragraphs, no headings.\n")

        outline = engine.get_book_outline("Flat.md")
        assert outline == []


# ---------------------------------------------------------------------------
# browse_topics tests
# ---------------------------------------------------------------------------


class TestBrowseTopics:
    def test_counts_by_document(self, engine: SearchEngine) -> None:
        """Tags shared across documents count correctly (not by chunk)."""
        topics = engine.browse_topics()
        by_tag = {t["tag"]: t["document_count"] for t in topics}

        # "architecture" appears in all 3 documents
        assert by_tag["architecture"] == 3
        # "book" appears in 2 book documents
        assert by_tag["book"] == 2
        # "design" only in Clean Architecture
        assert by_tag["design"] == 1
        # "translated" only in Effective Software Architectures
        assert by_tag["translated"] == 1
        # "meeting" only in the note
        assert by_tag["meeting"] == 1

    def test_folder_filter(self, engine: SearchEngine) -> None:
        """Filtering by folder shows only documents from that folder.

        Note: with flat sources, the folder field is empty string for
        root-level files. The folder filter uses the ``folder`` column
        in the LanceDB table.
        """
        # All docs in this test have folder="" since files are at source root
        topics = engine.browse_topics(folder="")
        by_tag = {t["tag"]: t["document_count"] for t in topics}
        assert by_tag["architecture"] == 3

    def test_empty_tags(self, vault: Path) -> None:
        """Documents with no tags produce no topics."""
        index_dir = vault / ".colibri" / "lancedb_notags"
        index_dir.mkdir(parents=True)

        db = lancedb.connect(str(index_dir))
        db.create_table(
            "chunks",
            data=[{
                "vector": [0.0] * EMBED_DIM,
                "text": "Untagged content.",
                "source_file": "Untagged.md",
                "title": "Untagged",
                "doc_type": "book",
                "folder": "",
                "source_name": "Books",
                "source_type": "markdown",
                "tags": "",
            }],
        )
        write_index_meta(index_dir)

        source = MarkdownFolderSource(vault / "Books", name="Books")
        with patch("colibri.query.LANCEDB_DIR", index_dir):
            eng = SearchEngine(source=source)

        assert eng.browse_topics() == []

    def test_sorted_by_count_descending(self, engine: SearchEngine) -> None:
        """Results are sorted with most frequent tags first."""
        topics = engine.browse_topics()
        counts = [t["document_count"] for t in topics]
        assert counts == sorted(counts, reverse=True)
