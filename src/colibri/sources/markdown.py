"""Plain markdown folder content source.

Handles simple markdown folders for plain markdown folders.
Supports optional YAML frontmatter but doesn't require it.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from colibri.sources.base import ContentSource, SourceDocument


class MarkdownFolderSource(ContentSource):
    """Content source for plain markdown folders.

    A general-purpose source for indexing a folder of markdown files.

    Features:
    - Optional YAML frontmatter (graceful fallback if missing)
    - Relative path links (standard markdown style)
    - Recursive or flat folder scanning
    - Title extraction from filename if not in frontmatter

    Example:
        source = MarkdownFolderSource(
            folder_path=Path("~/Documents/Notes"),
            recursive=True,
            name="My Notes"
        )

        for doc_path in source.list_documents():
            doc = source.read_document(doc_path)
            print(f"{doc.title}")
    """

    source_type = "markdown"

    def __init__(
        self,
        folder_path: str | Path,
        recursive: bool = True,
        name: str | None = None,
        extensions: tuple[str, ...] = (".md",),
        exclude_paths: tuple[str, ...] = (),
    ):
        """Initialize markdown folder source.

        Args:
            folder_path: Path to the markdown folder
            recursive: Whether to scan subfolders (default: True)
            name: Display name for this source (default: folder name)
            extensions: File extensions to index (default: (".md",))
            exclude_paths: Absolute paths to exclude (nested source roots)
        """
        self._folder_path = Path(folder_path).expanduser().resolve()
        self._recursive = recursive
        self._name = name or self._folder_path.name
        self._extensions = extensions
        self._exclude_paths = tuple(
            str(Path(p).resolve()) for p in exclude_paths
        )

    @property
    def name(self) -> str:
        """Source display name."""
        return self._name

    @property
    def root_path(self) -> Path:
        """Folder root path."""
        return self._folder_path

    @property
    def recursive(self) -> bool:
        """Whether subfolders are scanned."""
        return self._recursive

    def list_documents(self, folders: list[str] | None = None) -> Iterator[Path]:
        """Yield paths to all matching files.

        Args:
            folders: Optional subfolder filter (scans these subfolders only)

        Yields:
            Paths relative to folder root
        """
        seen: set[Path] = set()

        def _glob_base(base: Path) -> Iterator[Path]:
            for ext in self._extensions:
                pattern = f"**/*{ext}" if self._recursive else f"*{ext}"
                for fp in base.glob(pattern):
                    rel = fp.relative_to(self._folder_path)
                    if rel in seen:
                        continue
                    if any(part.startswith(".") for part in rel.parts):
                        continue
                    # Skip files under excluded (nested) source paths
                    if self._exclude_paths:
                        abs_str = str(fp.resolve())
                        if any(abs_str.startswith(ex + "/") or abs_str == ex
                               for ex in self._exclude_paths):
                            continue
                    seen.add(rel)
                    yield rel

        if folders:
            for folder in folders:
                folder_path = self._folder_path / folder
                if folder_path.exists():
                    yield from _glob_base(folder_path)
        else:
            yield from _glob_base(self._folder_path)

    def read_document(self, path: Path) -> SourceDocument:
        """Read and parse a document file.

        Handles markdown (.md) with optional YAML frontmatter, and
        YAML/YML files (e.g. OpenAPI specs) as plain text.

        Args:
            path: Path relative to folder root

        Returns:
            SourceDocument with available metadata

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self._folder_path / path

        if not full_path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = full_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return self._read_yaml(path, full_path)

        return self._read_markdown(path, full_path)

    def _read_markdown(self, path: Path, full_path: Path) -> SourceDocument:
        """Read a markdown file with optional frontmatter."""
        content = full_path.read_text(encoding="utf-8")

        # Try to parse frontmatter
        metadata: dict = {}
        doc_content = content

        if content.startswith("---"):
            try:
                import frontmatter

                post = frontmatter.loads(content)
                metadata = dict(post.metadata)
                doc_content = post.content
            except Exception:
                pass

        # Extract title: frontmatter > first H1 > filename
        title = metadata.get("title")
        if not title:
            title = self._extract_first_heading(doc_content)
        if not title:
            title = self._title_from_filename(path.stem)

        folder = path.parts[0] if len(path.parts) > 1 else ""

        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        return SourceDocument(
            path=path,
            content=doc_content,
            title=title,
            doc_type=metadata.get("type", "note"),
            source_name=self.name,
            source_type=self.source_type,
            folder=folder,
            metadata=metadata,
            tags=tags,
        )

    def _read_yaml(self, path: Path, full_path: Path) -> SourceDocument:
        """Read a YAML file (e.g. OpenAPI spec) as a document."""
        content = full_path.read_text(encoding="utf-8")

        # Try to extract title from OpenAPI info.title
        title = self._extract_yaml_title(content)
        if not title:
            title = self._title_from_filename(path.stem)

        folder = path.parts[0] if len(path.parts) > 1 else ""

        return SourceDocument(
            path=path,
            content=content,
            title=title,
            doc_type="note",
            source_name=self.name,
            source_type=self.source_type,
            folder=folder,
            metadata={},
            tags=[],
        )

    @staticmethod
    def _extract_yaml_title(content: str) -> str | None:
        """Extract title from OpenAPI info block or top-level title key."""
        # Match `title:` inside an `info:` block (OpenAPI convention)
        info_pat = r"^info:\s*\n(?:[ \t]+\S.*\n)*?[ \t]+title:\s*(.+)"
        match = re.search(info_pat, content, re.MULTILINE)
        if match:
            return match.group(1).strip().strip("'\"")
        # Fallback: top-level `title:` key
        match = re.search(r"^title:\s*(.+)", content, re.MULTILINE)
        if match:
            return match.group(1).strip().strip("'\"")
        return None

    def resolve_link(self, link: str, from_doc: Path | None = None) -> Path | None:
        """Resolve a relative markdown link to a document path.

        Handles standard markdown links:
        - ./sibling.md
        - ../parent/doc.md
        - subfolder/doc.md

        Args:
            link: Relative path to resolve
            from_doc: Document containing the link (for relative resolution)

        Returns:
            Path relative to folder root, or None if not found
        """
        # Clean up link (remove any URL encoding, anchors)
        clean_link = link.split("#")[0]  # Remove anchor
        clean_link = clean_link.split("?")[0]  # Remove query

        if not clean_link:
            return None

        # Resolve relative to source document or root
        base_dir = (self._folder_path / from_doc).parent if from_doc else self._folder_path
        resolved = (base_dir / clean_link).resolve()

        # Security: ensure resolved path is within folder
        try:
            resolved.relative_to(self._folder_path)
        except ValueError:
            return None

        # Check if file exists
        if not resolved.exists():
            # Try adding .md extension
            if not clean_link.endswith(".md"):
                resolved_with_ext = resolved.with_suffix(".md")
                if resolved_with_ext.exists():
                    return resolved_with_ext.relative_to(self._folder_path)
            return None

        return resolved.relative_to(self._folder_path)

    def _extract_first_heading(self, content: str) -> str | None:
        """Extract the first H1 heading from content."""
        # Match # Heading at start of line
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _title_from_filename(self, stem: str) -> str:
        """Convert filename stem to readable title."""
        # Replace common separators with spaces
        title = stem.replace("-", " ").replace("_", " ")
        # Title case
        return title.title()
