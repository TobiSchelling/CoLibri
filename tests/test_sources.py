"""Tests for content source adapters."""

from pathlib import Path

import pytest

from colibri.sources import (
    MarkdownFolderSource,
    ObsidianSource,
    SourceDocument,
    get_source,
)


class TestSourceDocument:
    """Tests for the SourceDocument dataclass."""

    def test_minimal_document(self) -> None:
        """Test creating document with only required fields."""
        doc = SourceDocument(
            path=Path("test.md"),
            content="Some content",
            title="Test",
        )

        assert doc.path == Path("test.md")
        assert doc.content == "Some content"
        assert doc.title == "Test"
        assert doc.doc_type == "note"
        assert doc.tags == []

    def test_full_document(self) -> None:
        """Test creating document with all fields."""
        doc = SourceDocument(
            path=Path("Books/Clean Code.md"),
            content="Chapter 1...",
            title="Clean Code",
            doc_type="book",
            source_name="MyVault",
            source_type="obsidian",
            folder="Books",
            metadata={"author": "Robert Martin"},
            tags=["programming", "clean-code"],
        )

        assert doc.doc_type == "book"
        assert doc.source_type == "obsidian"
        assert doc.folder == "Books"
        assert "author" in doc.metadata
        assert "programming" in doc.tags


class TestObsidianSource:
    """Tests for ObsidianSource."""

    @pytest.fixture
    def temp_vault(self, tmp_path: Path) -> Path:
        """Create a temporary Obsidian vault structure."""
        vault = tmp_path / "TestVault"
        vault.mkdir()

        # Create .obsidian folder (should be ignored)
        (vault / ".obsidian").mkdir()
        (vault / ".obsidian" / "config.json").write_text("{}")

        # Create Books folder with frontmatter
        books = vault / "Books"
        books.mkdir()

        (books / "Clean Code.md").write_text(
            """---
title: "Clean Code"
type: book
author: "Robert C. Martin"
tags:
  - programming
  - best-practices
---

# Clean Code

A Handbook of Agile Software Craftsmanship.

See also [[Design Patterns]].
"""
        )

        (books / "Design Patterns.md").write_text(
            """---
title: Design Patterns
type: book
---

# Design Patterns

Gang of Four book.
"""
        )

        # Create Notes folder
        notes = vault / "Notes"
        notes.mkdir()

        (notes / "Daily Note.md").write_text(
            """---
title: Daily Note
type: note
---

Today's notes.
"""
        )

        return vault

    def test_source_properties(self, temp_vault: Path) -> None:
        """Test basic source properties."""
        source = ObsidianSource(temp_vault, folders=["Books"])

        assert source.name == "TestVault"
        assert source.root_path == temp_vault
        assert source.source_type == "obsidian"
        assert source.books_folder == "Books"

    def test_list_documents(self, temp_vault: Path) -> None:
        """Test listing documents in folders."""
        source = ObsidianSource(temp_vault, folders=["Books"])

        docs = list(source.list_documents())
        assert len(docs) == 2

        names = [d.name for d in docs]
        assert "Clean Code.md" in names
        assert "Design Patterns.md" in names

    def test_list_documents_multiple_folders(self, temp_vault: Path) -> None:
        """Test listing from multiple folders."""
        source = ObsidianSource(temp_vault, folders=["Books", "Notes"])

        docs = list(source.list_documents())
        assert len(docs) == 3

    def test_list_excludes_hidden(self, temp_vault: Path) -> None:
        """Test that .obsidian folder is excluded."""
        source = ObsidianSource(temp_vault, folders=["Books"])

        docs = list(source.list_documents())
        for doc in docs:
            assert not any(part.startswith(".") for part in doc.parts)

    def test_read_document(self, temp_vault: Path) -> None:
        """Test reading a document with frontmatter."""
        source = ObsidianSource(temp_vault, folders=["Books"])

        doc = source.read_document(Path("Books/Clean Code.md"))

        assert doc.title == "Clean Code"
        assert doc.doc_type == "book"
        assert doc.folder == "Books"
        assert doc.source_type == "obsidian"
        assert "programming" in doc.tags
        assert "Robert C. Martin" in doc.metadata.get("author", "")
        assert "Clean Code" in doc.content
        assert "---" not in doc.content  # Frontmatter stripped

    def test_read_document_not_found(self, temp_vault: Path) -> None:
        """Test reading non-existent document."""
        source = ObsidianSource(temp_vault)

        with pytest.raises(FileNotFoundError):
            source.read_document(Path("nonexistent.md"))

    def test_resolve_wiki_link(self, temp_vault: Path) -> None:
        """Test resolving wiki links."""
        source = ObsidianSource(temp_vault, folders=["Books"])

        # Simple link
        resolved = source.resolve_link("Design Patterns")
        assert resolved is not None
        assert resolved.name == "Design Patterns.md"

        # Link with alias
        resolved = source.resolve_link("Design Patterns|GoF Book")
        assert resolved is not None
        assert resolved.name == "Design Patterns.md"

    def test_resolve_wiki_link_not_found(self, temp_vault: Path) -> None:
        """Test resolving non-existent link."""
        source = ObsidianSource(temp_vault)

        resolved = source.resolve_link("Nonexistent Note")
        assert resolved is None

    def test_extract_wiki_links(self, temp_vault: Path) -> None:
        """Test extracting wiki links from content."""
        source = ObsidianSource(temp_vault)

        content = "See [[Note A]] and [[Note B|alias]] for details."
        links = source.extract_wiki_links(content)

        assert "Note A" in links
        assert "Note B" in links
        assert len(links) == 2



class TestMarkdownFolderSource:
    """Tests for MarkdownFolderSource."""

    @pytest.fixture
    def temp_folder(self, tmp_path: Path) -> Path:
        """Create a temporary markdown folder structure."""
        folder = tmp_path / "Notes"
        folder.mkdir()

        # File with frontmatter
        (folder / "with-frontmatter.md").write_text(
            """---
title: My Document
type: article
tags: python, testing
---

# My Document

Content here.
"""
        )

        # File without frontmatter
        (folder / "no-frontmatter.md").write_text(
            """# Plain Document

Just markdown, no frontmatter.
"""
        )

        # Nested file
        (folder / "sub").mkdir()
        (folder / "sub" / "nested.md").write_text(
            """# Nested

In a subfolder.
"""
        )

        # Hidden folder (should be ignored)
        (folder / ".hidden").mkdir()
        (folder / ".hidden" / "secret.md").write_text("Secret content")

        return folder

    def test_source_properties(self, temp_folder: Path) -> None:
        """Test basic source properties."""
        source = MarkdownFolderSource(temp_folder, name="My Notes")

        assert source.name == "My Notes"
        assert source.root_path == temp_folder
        assert source.source_type == "markdown"
        assert source.recursive is True

    def test_list_documents_recursive(self, temp_folder: Path) -> None:
        """Test listing documents recursively."""
        source = MarkdownFolderSource(temp_folder, recursive=True)

        docs = list(source.list_documents())
        assert len(docs) == 3

        names = [d.name for d in docs]
        assert "with-frontmatter.md" in names
        assert "no-frontmatter.md" in names
        assert "nested.md" in names

    def test_list_documents_non_recursive(self, temp_folder: Path) -> None:
        """Test listing documents non-recursively."""
        source = MarkdownFolderSource(temp_folder, recursive=False)

        docs = list(source.list_documents())
        assert len(docs) == 2  # Excludes nested.md

    def test_list_excludes_hidden(self, temp_folder: Path) -> None:
        """Test that hidden folders are excluded."""
        source = MarkdownFolderSource(temp_folder)

        docs = list(source.list_documents())
        for doc in docs:
            assert not any(part.startswith(".") for part in doc.parts)

    def test_read_document_with_frontmatter(self, temp_folder: Path) -> None:
        """Test reading document with frontmatter."""
        source = MarkdownFolderSource(temp_folder)

        doc = source.read_document(Path("with-frontmatter.md"))

        assert doc.title == "My Document"
        assert doc.doc_type == "article"
        assert "python" in doc.tags
        assert "---" not in doc.content

    def test_read_document_without_frontmatter(self, temp_folder: Path) -> None:
        """Test reading document without frontmatter."""
        source = MarkdownFolderSource(temp_folder)

        doc = source.read_document(Path("no-frontmatter.md"))

        # Should extract title from H1
        assert doc.title == "Plain Document"
        assert doc.doc_type == "note"  # Default

    def test_read_document_title_from_filename(self, temp_folder: Path) -> None:
        """Test title extraction from filename."""
        # Create file without H1
        (temp_folder / "my-file-name.md").write_text("No heading here.")

        source = MarkdownFolderSource(temp_folder)
        doc = source.read_document(Path("my-file-name.md"))

        assert doc.title == "My File Name"  # Converted from filename

    def test_resolve_relative_link(self, temp_folder: Path) -> None:
        """Test resolving relative links."""
        source = MarkdownFolderSource(temp_folder)

        # From root
        resolved = source.resolve_link("sub/nested.md", from_doc=Path("with-frontmatter.md"))
        assert resolved is not None
        assert resolved == Path("sub/nested.md")

    def test_resolve_link_adds_extension(self, temp_folder: Path) -> None:
        """Test that .md extension is added if needed."""
        source = MarkdownFolderSource(temp_folder)

        resolved = source.resolve_link("sub/nested", from_doc=Path("with-frontmatter.md"))
        assert resolved is not None
        assert resolved.suffix == ".md"

    def test_resolve_link_not_found(self, temp_folder: Path) -> None:
        """Test resolving non-existent link."""
        source = MarkdownFolderSource(temp_folder)

        resolved = source.resolve_link("nonexistent.md")
        assert resolved is None


class TestMarkdownFolderSourceExclusion:
    """Tests for nesting exclusion in MarkdownFolderSource."""

    def test_exclude_paths_filters_nested_source(self, tmp_path: Path) -> None:
        """Files under excluded paths are not listed."""
        root = tmp_path / "vault"
        root.mkdir()
        (root / "note.md").write_text("# Root note")

        nested = root / "Books"
        nested.mkdir()
        (nested / "book.md").write_text("# Book")

        # Source at root, excluding the Books subfolder
        source = MarkdownFolderSource(root, exclude_paths=(str(nested),))
        docs = list(source.list_documents())

        names = [d.name for d in docs]
        assert "note.md" in names
        assert "book.md" not in names

    def test_no_exclusion_returns_all(self, tmp_path: Path) -> None:
        """Without exclude_paths, nested files are included."""
        root = tmp_path / "vault"
        root.mkdir()
        (root / "note.md").write_text("# Root note")

        nested = root / "Books"
        nested.mkdir()
        (nested / "book.md").write_text("# Book")

        source = MarkdownFolderSource(root)
        docs = list(source.list_documents())
        assert len(docs) == 2


class TestGetSource:
    """Tests for the get_source factory function."""

    def test_create_obsidian_source(self, tmp_path: Path) -> None:
        """Test creating Obsidian source from config."""
        vault = tmp_path / "vault"
        vault.mkdir()

        source = get_source(
            {
                "type": "obsidian",
                "path": str(vault),
                "folders": ["Books"],
            }
        )

        assert isinstance(source, ObsidianSource)
        assert source.folders == ["Books"]

    def test_create_markdown_source(self, tmp_path: Path) -> None:
        """Test creating markdown source from config."""
        folder = tmp_path / "notes"
        folder.mkdir()

        source = get_source(
            {
                "type": "markdown",
                "path": str(folder),
                "recursive": False,
                "name": "My Notes",
            }
        )

        assert isinstance(source, MarkdownFolderSource)
        assert source.name == "My Notes"
        assert source.recursive is False

    def test_default_type_is_markdown(self, tmp_path: Path) -> None:
        """Test that default source type is markdown."""
        folder = tmp_path / "notes"
        folder.mkdir()

        source = get_source({"path": str(folder)})
        assert isinstance(source, MarkdownFolderSource)

    def test_missing_path_raises(self) -> None:
        """Test that missing path raises ValueError."""
        with pytest.raises(ValueError, match="must include 'path'"):
            get_source({"type": "obsidian"})

    def test_unknown_type_raises(self, tmp_path: Path) -> None:
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source type"):
            get_source({"type": "unknown", "path": str(tmp_path)})
