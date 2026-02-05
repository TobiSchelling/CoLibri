"""Base classes for content sources.

This module defines the abstract interface that all content sources must implement,
plus the normalized document structure for indexed content.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceDocument:
    """Normalized document representation from any source.

    This dataclass represents a document ready for indexing, with
    consistent metadata regardless of the source type.

    Attributes:
        path: Path to the document (relative to source root)
        content: Document text content (without frontmatter)
        title: Document title
        doc_type: Document type (e.g., "book", "note")
        source_name: Name of the content source
        source_type: Type of source ("markdown", "obsidian", etc.)
        folder: Folder within the source (for filtering)
        metadata: Additional metadata from frontmatter or source
        tags: List of tags if available
    """

    path: Path
    content: str
    title: str
    doc_type: str = "note"
    source_name: str = ""
    source_type: str = ""
    folder: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class ContentSource(ABC):
    """Abstract base class for content sources.

    A content source represents a collection of markdown documents
    that can be indexed and searched. Different implementations handle
    different folder structures and conventions.

    To implement a new source:
    1. Subclass ContentSource
    2. Implement all abstract methods
    3. Set source_type class variable

    Example:
        class MySource(ContentSource):
            source_type = "my_source"

            def list_documents(self, folders=None):
                # ... yield document paths ...

            def read_document(self, path):
                # ... return SourceDocument ...
    """

    # Subclasses should set this
    source_type: str = "unknown"

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this source."""
        ...

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """Base path for this content source."""
        ...

    @abstractmethod
    def list_documents(self, folders: list[str] | None = None) -> Iterator[Path]:
        """Yield paths to indexable documents.

        Args:
            folders: Optional list of folder names to restrict to.
                     If None, all configured folders are included.

        Yields:
            Path objects relative to source root
        """
        ...

    @abstractmethod
    def read_document(self, path: Path) -> SourceDocument:
        """Read and parse a document.

        Args:
            path: Path to the document (relative to source root)

        Returns:
            SourceDocument with content and metadata

        Raises:
            FileNotFoundError: If document doesn't exist
            ValueError: If document can't be parsed
        """
        ...

    @abstractmethod
    def resolve_link(self, link: str, from_doc: Path | None = None) -> Path | None:
        """Resolve an internal link to a document path.

        Different sources have different link conventions:
        - Some sources use [[wiki links]]
        - Plain markdown uses relative paths

        Args:
            link: The link text to resolve
            from_doc: Optional source document (for relative links)

        Returns:
            Resolved path relative to source root, or None if not found
        """
        ...

