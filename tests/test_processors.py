"""Tests for the document processor architecture."""

from datetime import datetime
from pathlib import Path

from colibri.processors.base import DocumentProcessor, ExtractedContent
from colibri.processors.registry import ProcessorRegistry
from colibri.processors.utils import (
    build_frontmatter_dict,
    clean_html_whitespace,
    clean_pdf_text,
    clean_text,
    format_frontmatter,
    generate_document,
    sanitize_filename,
)


class TestExtractedContent:
    """Tests for the ExtractedContent dataclass."""

    def test_minimal_content(self) -> None:
        """Test creating content with only required fields."""
        content = ExtractedContent(
            title="Test Book",
            content="Some content here",
            source_path=Path("/path/to/book.pdf"),
            source_format="pdf",
        )

        assert content.title == "Test Book"
        assert content.content == "Some content here"
        assert content.source_format == "pdf"
        assert content.author is None
        assert content.metadata == {}
        assert isinstance(content.extracted_at, datetime)

    def test_full_content(self) -> None:
        """Test creating content with all fields."""
        content = ExtractedContent(
            title="Clean Architecture",
            content="Chapter 1...",
            source_path=Path("/books/clean-arch.epub"),
            source_format="epub",
            author="Robert C. Martin",
            publisher="Prentice Hall",
            language="en",
            isbn="978-0-13-449416-6",
            description="A guide to software architecture",
            metadata={"chapter_count": 34},
        )

        assert content.author == "Robert C. Martin"
        assert content.publisher == "Prentice Hall"
        assert content.isbn == "978-0-13-449416-6"
        assert content.metadata["chapter_count"] == 34


class TestProcessorRegistry:
    """Tests for the ProcessorRegistry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        # Save current processors
        self._saved_processors = ProcessorRegistry._processors.copy()
        ProcessorRegistry.clear()

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        ProcessorRegistry._processors = self._saved_processors

    def test_register_processor(self) -> None:
        """Test registering a processor."""

        @ProcessorRegistry.register
        class TestProcessor(DocumentProcessor):
            extensions = [".test"]

            def extract(self, path: Path) -> ExtractedContent:
                return ExtractedContent(
                    title="Test",
                    content="",
                    source_path=path,
                    source_format="test",
                )

        assert TestProcessor in ProcessorRegistry._processors

    def test_get_processor_by_path(self) -> None:
        """Test finding processor by file path."""

        @ProcessorRegistry.register
        class TestProcessor(DocumentProcessor):
            extensions = [".xyz"]

            def extract(self, path: Path) -> ExtractedContent:
                return ExtractedContent(
                    title="Test",
                    content="",
                    source_path=path,
                    source_format="xyz",
                )

        processor = ProcessorRegistry.get_processor(Path("file.xyz"))
        assert processor is not None
        assert isinstance(processor, TestProcessor)

    def test_get_processor_case_insensitive(self) -> None:
        """Test that extension matching is case-insensitive."""

        @ProcessorRegistry.register
        class TestProcessor(DocumentProcessor):
            extensions = [".abc"]

            def extract(self, path: Path) -> ExtractedContent:
                return ExtractedContent(
                    title="Test",
                    content="",
                    source_path=path,
                    source_format="abc",
                )

        # Should match regardless of case
        assert ProcessorRegistry.get_processor(Path("file.ABC")) is not None
        assert ProcessorRegistry.get_processor(Path("file.Abc")) is not None

    def test_get_processor_unknown_format(self) -> None:
        """Test that unknown formats return None."""
        processor = ProcessorRegistry.get_processor(Path("file.unknown"))
        assert processor is None

    def test_supported_extensions(self) -> None:
        """Test listing supported extensions."""

        @ProcessorRegistry.register
        class Proc1(DocumentProcessor):
            extensions = [".aaa", ".bbb"]

            def extract(self, path: Path) -> ExtractedContent:
                raise NotImplementedError

        @ProcessorRegistry.register
        class Proc2(DocumentProcessor):
            extensions = [".ccc"]

            def extract(self, path: Path) -> ExtractedContent:
                raise NotImplementedError

        extensions = ProcessorRegistry.supported_extensions()
        assert ".aaa" in extensions
        assert ".bbb" in extensions
        assert ".ccc" in extensions

    def test_list_processors(self) -> None:
        """Test listing processors with their names."""

        @ProcessorRegistry.register
        class MyProcessor(DocumentProcessor):
            extensions = [".my"]

            def extract(self, path: Path) -> ExtractedContent:
                raise NotImplementedError

        processors = ProcessorRegistry.list_processors()
        assert ("MyProcessor", [".my"]) in processors


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_basic_sanitization(self) -> None:
        """Test basic character removal."""
        assert sanitize_filename("Clean Code: A Handbook") == "Clean Code A Handbook"

    def test_preserves_alphanumeric(self) -> None:
        """Test that alphanumeric chars are preserved."""
        assert sanitize_filename("Chapter 1") == "Chapter 1"
        assert sanitize_filename("Test123") == "Test123"

    def test_preserves_hyphens_underscores(self) -> None:
        """Test that hyphens and underscores are preserved."""
        assert sanitize_filename("my-book_name") == "my-book_name"

    def test_removes_special_chars(self) -> None:
        """Test removal of special characters."""
        assert sanitize_filename("Test!@#$%Book") == "TestBook"
        assert sanitize_filename("What? Why!") == "What Why"

    def test_collapses_spaces(self) -> None:
        """Test that multiple spaces are collapsed."""
        assert sanitize_filename("Too    Many   Spaces") == "Too Many Spaces"

    def test_max_length(self) -> None:
        """Test length truncation."""
        long_title = "A" * 200
        result = sanitize_filename(long_title, max_length=50)
        assert len(result) == 50

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        assert sanitize_filename("") == ""
        assert sanitize_filename("   ") == ""


class TestCleanText:
    """Tests for text cleaning utilities."""

    def test_smart_quotes(self) -> None:
        """Test conversion of smart quotes."""
        text = "\u201cHello\u201d \u2018World\u2019"
        cleaned = clean_text(text)
        assert cleaned == "\"Hello\" 'World'"

    def test_dashes(self) -> None:
        """Test conversion of dashes."""
        text = "en\u2013dash and em\u2014dash"
        cleaned = clean_text(text)
        assert cleaned == "en-dash and em--dash"

    def test_ellipsis(self) -> None:
        """Test conversion of ellipsis."""
        assert clean_text("Wait\u2026") == "Wait..."

    def test_nbsp(self) -> None:
        """Test conversion of non-breaking space."""
        assert clean_text("Hello\u00a0World") == "Hello World"

    def test_control_characters(self) -> None:
        """Test removal of control characters."""
        text = "Hello\x07World\x08Test"  # Bell and backspace
        cleaned = clean_text(text)
        assert cleaned == "HelloWorldTest"

    def test_preserves_newlines_tabs(self) -> None:
        """Test that newlines and tabs are preserved."""
        text = "Line1\nLine2\tTabbed"
        assert clean_text(text) == text


class TestCleanPdfText:
    """Tests for PDF-specific text cleaning."""

    def test_replacement_characters(self) -> None:
        """Test removal of replacement characters."""
        text = "Chapter 1\ufffd\ufffd\ufffd\ufffd5"
        cleaned = clean_pdf_text(text)
        assert "\ufffd" not in cleaned

    def test_toc_dots(self) -> None:
        """Test cleanup of TOC dot sequences."""
        text = "Chapter 1.......................5"
        cleaned = clean_pdf_text(text)
        assert "....." not in cleaned
        assert "..." in cleaned  # Should be collapsed to ellipsis

    def test_excessive_spaces(self) -> None:
        """Test cleanup of excessive spaces."""
        text = "Too     many     spaces"
        cleaned = clean_pdf_text(text)
        assert "     " not in cleaned

    def test_excessive_newlines(self) -> None:
        """Test cleanup of excessive newlines."""
        text = "Para 1\n\n\n\n\n\nPara 2"
        cleaned = clean_pdf_text(text)
        assert "\n\n\n\n" not in cleaned


class TestCleanHtmlWhitespace:
    """Tests for HTML whitespace cleaning."""

    def test_collapses_empty_lines(self) -> None:
        """Test that multiple empty lines are collapsed."""
        text = "Line 1\n\n\n\nLine 2"
        cleaned = clean_html_whitespace(text)
        assert cleaned == "Line 1\n\nLine 2"

    def test_preserves_single_blank_line(self) -> None:
        """Test that single blank lines are preserved."""
        text = "Para 1\n\nPara 2"
        cleaned = clean_html_whitespace(text)
        assert cleaned == "Para 1\n\nPara 2"

    def test_strips_edges(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        text = "\n\nContent\n\n"
        cleaned = clean_html_whitespace(text)
        assert cleaned == "Content"


class TestFormatFrontmatter:
    """Tests for frontmatter formatting."""

    def test_basic_frontmatter(self) -> None:
        """Test basic frontmatter generation."""
        metadata = {"title": "Test Book", "type": "book"}
        result = format_frontmatter(metadata)

        assert result.startswith("---\n")
        assert result.endswith("---\n")
        assert "title: Test Book" in result
        assert "type: book" in result

    def test_quotes_special_chars(self) -> None:
        """Test that special characters are properly quoted."""
        metadata = {"title": 'Book: "The Guide"'}
        result = format_frontmatter(metadata)
        # YAML should properly escape the quotes
        assert "---" in result

    def test_preserves_order(self) -> None:
        """Test that field order is preserved."""
        metadata = {"z_field": 1, "a_field": 2, "m_field": 3}
        result = format_frontmatter(metadata)
        # With sort_keys=False, order should be preserved
        z_pos = result.find("z_field")
        a_pos = result.find("a_field")
        m_pos = result.find("m_field")
        assert z_pos < a_pos < m_pos


class TestBuildFrontmatterDict:
    """Tests for frontmatter dictionary building."""

    def test_minimal_content(self) -> None:
        """Test with minimal content fields."""
        content = ExtractedContent(
            title="Test",
            content="",
            source_path=Path("test.pdf"),
            source_format="pdf",
        )
        fm = build_frontmatter_dict(content)

        assert fm["title"] == "Test"
        assert fm["type"] == "book"
        assert fm["source_pdf"] == "test.pdf"
        assert "imported" in fm
        assert "book" in fm["tags"]
        assert "pdf" in fm["tags"]

    def test_with_author(self) -> None:
        """Test that author is included when present."""
        content = ExtractedContent(
            title="Test",
            content="",
            source_path=Path("test.epub"),
            source_format="epub",
            author="John Doe",
        )
        fm = build_frontmatter_dict(content)
        assert fm["author"] == "John Doe"

    def test_extra_tags(self) -> None:
        """Test adding extra tags."""
        content = ExtractedContent(
            title="Test",
            content="",
            source_path=Path("test.pdf"),
            source_format="pdf",
        )
        fm = build_frontmatter_dict(content, extra_tags=["programming", "architecture"])
        assert "programming" in fm["tags"]
        assert "architecture" in fm["tags"]


class TestGenerateObsidianDocument:
    """Tests for full document generation."""

    def test_basic_document(self) -> None:
        """Test basic document structure."""
        content = ExtractedContent(
            title="My Book",
            content="Chapter 1 content here.",
            source_path=Path("book.pdf"),
            source_format="pdf",
        )
        doc = generate_document(content)

        assert doc.startswith("---\n")
        assert "title: My Book" in doc
        assert "# My Book" in doc
        assert "Chapter 1 content here." in doc
        assert "> [!info] Source" in doc

    def test_without_callout(self) -> None:
        """Test document without source callout."""
        content = ExtractedContent(
            title="My Book",
            content="Content",
            source_path=Path("book.pdf"),
            source_format="pdf",
        )
        doc = generate_document(content, include_source_callout=False)

        assert "> [!info]" not in doc
        assert "Content" in doc


class TestRealProcessors:
    """Integration tests with actual processors (if available)."""

    def test_pdf_processor_registered(self) -> None:
        """Test that PDF processor is registered."""
        # Import to trigger registration
        from colibri.processors import ProcessorRegistry

        processor = ProcessorRegistry.get_processor(Path("test.pdf"))
        assert processor is not None
        assert processor.name == "PDF"

    def test_epub_processor_registered(self) -> None:
        """Test that EPUB processor is registered."""
        from colibri.processors import ProcessorRegistry

        processor = ProcessorRegistry.get_processor(Path("test.epub"))
        assert processor is not None
        assert processor.name == "EPUB"
