"""Obsidian-compatible content source.

Handles Obsidian-specific conventions:
- YAML frontmatter for metadata
- [[wiki links]] for internal links
- .obsidian/ folder exclusion
- Standard folder organization (Books/, Notes/, etc.)
"""

import re
from collections.abc import Iterator
from pathlib import Path

import frontmatter

from colibri.sources.base import ContentSource, SourceDocument


class ObsidianSource(ContentSource):
    """Content source for Obsidian-compatible markdown folders.

    Supports Obsidian conventions:
    - YAML frontmatter for document metadata
    - [[wiki links]] for linking between notes
    - .obsidian/ configuration folder (excluded from indexing)
    - Tag support via frontmatter or inline #tags

    Example:
        source = ObsidianSource(
            root_path=Path("~/Documents/CoLibri"),
            folders=["Books", "Notes"],
            books_folder="Books"
        )

        for doc_path in source.list_documents():
            doc = source.read_document(doc_path)
            print(f"{doc.title}: {len(doc.content)} chars")
    """

    source_type = "obsidian"

    def __init__(
        self,
        root_path: str | Path | None = None,
        folders: list[str] | None = None,
        books_folder: str = "Books",
    ):
        """Initialize Obsidian-compatible content source.

        Args:
            root_path: Path to the content root directory.
            folders: List of folders to index (default: [books_folder]).
            books_folder: Name of the books folder (default: "Books").
        """
        if root_path is None:
            raise TypeError("root_path is required")
        self._root_path = Path(root_path).expanduser().resolve()
        self._books_folder = books_folder
        self._folders = folders or [books_folder]
        self._link_index: dict[str, Path] | None = None

    @property
    def name(self) -> str:
        """Source name (directory name)."""
        return self._root_path.name

    @property
    def root_path(self) -> Path:
        """Content root path."""
        return self._root_path

    @property
    def books_folder(self) -> str:
        """Name of the books folder."""
        return self._books_folder

    @property
    def folders(self) -> list[str]:
        """List of indexed folders."""
        return self._folders

    def list_documents(self, folders: list[str] | None = None) -> Iterator[Path]:
        """Yield paths to all markdown files in specified folders.

        Args:
            folders: Folders to scan (defaults to configured folders)

        Yields:
            Paths relative to source root
        """
        folders = folders or self._folders

        for folder in folders:
            folder_path = self._root_path / folder
            if not folder_path.exists():
                continue

            for md_file in folder_path.rglob("*.md"):
                # Skip hidden files and .obsidian directory
                if any(part.startswith(".") for part in md_file.parts):
                    continue

                yield md_file.relative_to(self._root_path)

    def read_document(self, path: Path) -> SourceDocument:
        """Read and parse an Obsidian markdown file.

        Extracts YAML frontmatter metadata and document content.

        Args:
            path: Path relative to source root

        Returns:
            SourceDocument with frontmatter metadata extracted

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file can't be parsed
        """
        full_path = self._root_path / path

        if not full_path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        try:
            post = frontmatter.load(full_path)
        except Exception as e:
            raise ValueError(f"Failed to parse {path}: {e}") from e

        # Determine folder (first path component)
        folder = path.parts[0] if len(path.parts) > 1 else ""

        # Extract tags from frontmatter
        tags = post.metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        return SourceDocument(
            path=path,
            content=post.content,
            title=post.metadata.get("title", path.stem),
            doc_type=post.metadata.get("type", "note"),
            source_name=self.name,
            source_type=self.source_type,
            folder=folder,
            metadata=dict(post.metadata),
            tags=tags,
        )

    def resolve_link(self, link: str, from_doc: Path | None = None) -> Path | None:
        """Resolve an Obsidian wiki link to a document path.

        Handles:
        - [[Note Name]] -> finds Note Name.md anywhere in source
        - [[Note Name|Alias]] -> extracts Note Name, ignores alias
        - [[folder/Note Name]] -> respects path hint

        Args:
            link: Wiki link text (without brackets)
            from_doc: Source document (unused for wiki links)

        Returns:
            Path relative to source root, or None if not found
        """
        # Strip alias if present: [[link|alias]] -> link
        if "|" in link:
            link = link.split("|")[0]

        # Build link index on first use
        if self._link_index is None:
            self._build_link_index()

        # Normalize link for lookup
        link_lower = link.lower().strip()

        # Try exact match first
        if link_lower in self._link_index:
            return self._link_index[link_lower]

        # Try with .md extension
        if not link_lower.endswith(".md"):
            link_with_ext = f"{link_lower}.md"
            if link_with_ext in self._link_index:
                return self._link_index[link_with_ext]

        return None

    def _build_link_index(self) -> None:
        """Build index of note names to paths for link resolution."""
        self._link_index = {}

        for md_file in self._root_path.rglob("*.md"):
            # Skip hidden files
            if any(part.startswith(".") for part in md_file.relative_to(self._root_path).parts):
                continue

            rel_path = md_file.relative_to(self._root_path)

            # Index by filename without extension (how Obsidian links work)
            name_lower = md_file.stem.lower()
            if name_lower not in self._link_index:
                self._link_index[name_lower] = rel_path

            # Also index by full relative path
            path_lower = str(rel_path).lower()
            self._link_index[path_lower] = rel_path

    def extract_wiki_links(self, content: str) -> list[str]:
        """Extract all wiki links from content.

        Args:
            content: Markdown content

        Returns:
            List of link targets (note names, without brackets)
        """
        # Match [[link]] and [[link|alias]]
        pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        return re.findall(pattern, content)
