"""PDF document processor.

Extracts content from PDF files using pymupdf4llm, which provides
high-quality Markdown conversion with layout preservation.
"""

from pathlib import Path

import pymupdf4llm

from colibri.processors.base import DocumentProcessor, ExtractedContent
from colibri.processors.registry import ProcessorRegistry
from colibri.processors.utils import clean_pdf_text, sanitize_filename


@ProcessorRegistry.register
class PDFProcessor(DocumentProcessor):
    """Extract content from PDF documents.

    Uses pymupdf4llm for extraction, which handles:
    - Multi-column layouts
    - Tables (converted to Markdown tables)
    - Headers and formatting
    - Code blocks (when detectable)

    Note: Image extraction is disabled by default to keep
    the output focused on text content.
    """

    extensions = [".pdf"]

    def extract(self, path: Path) -> ExtractedContent:
        """Extract content from a PDF file.

        Args:
            path: Path to the PDF file

        Returns:
            ExtractedContent with cleaned markdown text

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the PDF cannot be processed
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Extract PDF to markdown
        try:
            raw_markdown = pymupdf4llm.to_markdown(
                str(path),
                page_chunks=False,  # Single document, not per-page
                write_images=False,  # Skip image extraction
                show_progress=False,  # We handle progress externally
            )
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {e}") from e

        # Clean up the extracted text
        content = clean_pdf_text(raw_markdown)

        # Derive title from filename
        title = sanitize_filename(path.stem)
        if not title:
            title = "Untitled PDF"

        return ExtractedContent(
            title=title,
            content=content,
            source_path=path,
            source_format="pdf",
            metadata={
                "processor": "pymupdf4llm",
            },
        )
