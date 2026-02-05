"""Base classes for document processors.

This module defines the abstract interface that all document processors must implement,
plus the normalized data structure for extracted content.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ExtractedContent:
    """Normalized extraction result from any document format.

    This dataclass represents the common output of all processors,
    regardless of source format. It separates content from metadata
    and provides optional fields for rich bibliographic information.

    Attributes:
        title: Document title (from metadata or filename)
        content: Main text content in Markdown format
        source_path: Original file path
        source_format: Format identifier (e.g., "pdf", "epub")
        metadata: Additional format-specific metadata
        extracted_at: Timestamp of extraction
        author: Author name(s) if available
        publisher: Publisher if available
        language: Language code if available
        isbn: ISBN if available
        description: Description/summary if available
    """

    title: str
    content: str
    source_path: Path
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    extracted_at: datetime = field(default_factory=datetime.now)

    # Optional bibliographic fields (commonly available in EPUBs)
    author: str | None = None
    publisher: str | None = None
    language: str | None = None
    isbn: str | None = None
    description: str | None = None


class DocumentProcessor(ABC):
    """Abstract base class for document format processors.

    To implement a new processor:
    1. Subclass DocumentProcessor
    2. Set the `extensions` class variable
    3. Implement the `extract()` method
    4. Register with @ProcessorRegistry.register decorator

    Example:
        @ProcessorRegistry.register
        class MyFormatProcessor(DocumentProcessor):
            extensions = ['.xyz', '.abc']

            def extract(self, path: Path) -> ExtractedContent:
                # ... extraction logic ...
                return ExtractedContent(...)
    """

    # Subclasses must define supported file extensions (lowercase, with dot)
    extensions: ClassVar[list[str]] = []

    @abstractmethod
    def extract(self, path: Path) -> ExtractedContent:
        """Extract content from a document.

        Args:
            path: Path to the document file

        Returns:
            ExtractedContent with title, content, and metadata

        Raises:
            ValueError: If the file cannot be processed
            FileNotFoundError: If the file doesn't exist
        """
        ...

    def can_handle(self, path: Path) -> bool:
        """Check if this processor can handle the given file.

        Default implementation checks file extension against `extensions`.
        Override for more sophisticated detection (e.g., magic bytes).

        Args:
            path: Path to check

        Returns:
            True if this processor can handle the file
        """
        return path.suffix.lower() in self.extensions

    @property
    def name(self) -> str:
        """Human-readable processor name."""
        return self.__class__.__name__.replace("Processor", "")
