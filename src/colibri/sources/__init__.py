"""Content source adapters for different markdown folder types.

This package provides a unified interface for reading documents from
different sources (plain markdown folders, Obsidian-compatible folders, etc.).

Usage:
    from colibri.sources import get_source, ObsidianSource, MarkdownFolderSource

    # Get source from config
    source = get_source(config_dict)

    # Or create directly
    source = ObsidianSource(root_path=Path("~/Documents/CoLibri"))

    # List and read documents
    for doc in source.list_documents():
        document = source.read_document(doc)
"""

from colibri.sources.base import ContentSource, SourceDocument
from colibri.sources.markdown import MarkdownFolderSource
from colibri.sources.obsidian import ObsidianSource

__all__ = [
    "ContentSource",
    "SourceDocument",
    "ObsidianSource",
    "MarkdownFolderSource",
    "get_source",
]


def get_source(source_config: dict) -> ContentSource:
    """Create a ContentSource from a configuration dictionary.

    Args:
        source_config: Dict with 'type' and 'path' keys, plus type-specific options

    Returns:
        Configured ContentSource instance

    Raises:
        ValueError: If source type is unknown

    Example:
        source = get_source({
            "type": "obsidian",
            "path": "~/Documents/CoLibri",
            "folders": ["Books", "Notes"]
        })
    """
    source_type = source_config.get("type", "markdown")
    path = source_config.get("path")

    if not path:
        raise ValueError("Source config must include 'path'")

    if source_type == "obsidian":
        return ObsidianSource(
            root_path=path,
            folders=source_config.get("folders"),
            books_folder=source_config.get("books_folder", "Books"),
        )
    elif source_type == "markdown":
        return MarkdownFolderSource(
            folder_path=path,
            recursive=source_config.get("recursive", True),
            name=source_config.get("name"),
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")
