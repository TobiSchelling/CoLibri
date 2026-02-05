"""EPUB document processor.

Extracts content from EPUB files using ebooklib for parsing
and markdownify for HTML-to-Markdown conversion.
"""

from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from markdownify import markdownify as md

from colibri.processors.base import DocumentProcessor, ExtractedContent
from colibri.processors.registry import ProcessorRegistry
from colibri.processors.utils import clean_html_whitespace, clean_text, sanitize_filename


@ProcessorRegistry.register
class EPUBProcessor(DocumentProcessor):
    """Extract content from EPUB documents.

    EPUB files are essentially zipped HTML documents with metadata.
    This processor:
    - Extracts Dublin Core metadata (title, author, publisher, etc.)
    - Converts each chapter's HTML to Markdown
    - Joins chapters with horizontal rules
    - Cleans up whitespace and formatting
    """

    extensions = [".epub"]

    def extract(self, path: Path) -> ExtractedContent:
        """Extract content from an EPUB file.

        Args:
            path: Path to the EPUB file

        Returns:
            ExtractedContent with markdown text and bibliographic metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the EPUB cannot be processed
        """
        if not path.exists():
            raise FileNotFoundError(f"EPUB file not found: {path}")

        try:
            book = epub.read_epub(str(path), options={"ignore_ncx": True})
        except Exception as e:
            raise ValueError(f"Failed to read EPUB: {e}") from e

        # Extract metadata
        title = self._get_metadata(book, "title") or sanitize_filename(path.stem) or "Untitled"
        author = self._get_metadata(book, "creator")
        publisher = self._get_metadata(book, "publisher")
        language = self._get_metadata(book, "language")
        isbn = self._extract_isbn(book)
        description = self._get_metadata(book, "description")

        # Process chapters
        chapters = self._extract_chapters(book)
        content = "\n\n---\n\n".join(chapters)

        return ExtractedContent(
            title=title,
            content=content,
            source_path=path,
            source_format="epub",
            author=author,
            publisher=publisher,
            language=language,
            isbn=isbn,
            description=description,
            metadata={
                "processor": "ebooklib+markdownify",
                "chapter_count": len(chapters),
            },
        )

    def _get_metadata(self, book: epub.EpubBook, field: str) -> str | None:
        """Extract a Dublin Core metadata field.

        Args:
            book: The parsed EPUB book
            field: Dublin Core field name (e.g., "title", "creator")

        Returns:
            The field value, or None if not present
        """
        values = book.get_metadata("DC", field)
        if values:
            return values[0][0]
        return None

    def _extract_isbn(self, book: epub.EpubBook) -> str | None:
        """Extract ISBN from identifiers if present.

        EPUB files may have multiple identifiers; this looks
        for one that contains "isbn".

        Args:
            book: The parsed EPUB book

        Returns:
            ISBN string, or None if not found
        """
        identifiers = book.get_metadata("DC", "identifier") or []
        for ident in identifiers:
            if "isbn" in str(ident).lower():
                return ident[0]
        return None

    def _extract_chapters(self, book: epub.EpubBook) -> list[str]:
        """Extract and convert all document items to Markdown.

        Args:
            book: The parsed EPUB book

        Returns:
            List of chapter contents as Markdown strings
        """
        chapters: list[str] = []

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            # Parse HTML content
            try:
                html_content = item.get_content().decode("utf-8")
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                html_content = item.get_content().decode("latin-1")

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove non-content elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get body content or full document
            body = soup.find("body")
            content_html = str(body) if body else str(soup)

            # Convert to Markdown
            markdown_content = md(
                content_html,
                heading_style="ATX",  # Use # style headings
                bullets="-",  # Use - for unordered lists
                strip=["a"],  # Remove links but keep text
            )

            # Clean up
            markdown_content = clean_text(markdown_content)
            markdown_content = clean_html_whitespace(markdown_content)

            if markdown_content:
                chapters.append(markdown_content)

        return chapters
