"""Query engine for semantic search over content sources.

Uses direct LanceDB SDK and Ollama HTTP API — no framework dependency.
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path

import lancedb

from colibri.config import (
    EMBEDDING_MODEL,
    LANCEDB_DIR,
    OLLAMA_BASE_URL,
    SIMILARITY_THRESHOLD,
    SOURCES,
    TOP_K,
)
from colibri.embedding import embed_texts
from colibri.index_meta import SCHEMA_VERSION, read_index_meta
from colibri.source_factory import compute_nested_exclusions, create_source_for_profile
from colibri.sources import ContentSource, SourceDocument

logger = logging.getLogger(__name__)


class SearchEngine:
    """Search engine for content sources with semantic retrieval."""

    def __init__(
        self,
        table_name: str = "chunks",
        source: ContentSource | None = None,
    ):
        """Initialize the search engine.

        Args:
            table_name: LanceDB table name
            source: Content source (if provided, used as primary source)
        """
        # Build all sources from SOURCES config
        nested_map = compute_nested_exclusions(SOURCES)
        self._all_sources: list[ContentSource] = []

        if source is not None:
            self._all_sources.append(source)

        for profile in SOURCES:
            src = create_source_for_profile(profile, nested_map)
            # Avoid adding duplicates when a custom source was passed
            if source is not None and src.root_path == source.root_path:
                continue
            self._all_sources.append(src)

        self.source = source or (self._all_sources[0] if self._all_sources else None)
        self.index_path = LANCEDB_DIR
        self.table_name = table_name

        # Check schema version before opening the table
        meta = read_index_meta(self.index_path)
        stored_version = meta.get("schema_version", 0)
        if stored_version != SCHEMA_VERSION:
            msg = (
                f"Index schema outdated (v{stored_version}, need v{SCHEMA_VERSION}). "
                f"Run `colibri index --force` to rebuild."
            )
            raise RuntimeError(msg)

        # Connect to LanceDB
        self._db = lancedb.connect(str(self.index_path))
        self._table = self._db.open_table(table_name)

    def search(
        self,
        query: str,
        folder: str | None = None,
        doc_type: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search library content with optional folder or doc_type filter.

        Args:
            query: The search query
            folder: Optional folder name to filter results (e.g., "Books")
            doc_type: Optional document type filter (e.g., "book")
            limit: Maximum number of results to return

        Returns:
            List of search results with text, metadata, and scores
        """
        # Embed the query
        query_vector = embed_texts([query], model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)[0]

        # Build search query
        search_query = self._table.search(query_vector).limit(TOP_K)

        # Apply folder pre-filter (use startswith to handle nested folder paths)
        if folder:
            escaped_folder = folder.replace("'", "''")
            search_query = search_query.where(
                f"folder = '{escaped_folder}' OR source_file LIKE '{escaped_folder}/%'"
            )

        # Apply doc_type filter
        if doc_type:
            escaped_type = doc_type.replace("'", "''")
            search_query = search_query.where(f"doc_type = '{escaped_type}'")

        raw_results = search_query.to_list()

        results = []
        for row in raw_results:
            # Convert L2 distance to similarity score (0-1 range)
            distance = row.get("_distance", 0.0)
            score = math.exp(-distance)

            if score < SIMILARITY_THRESHOLD:
                continue

            source_file = row.get("source_file", "")
            results.append(
                {
                    "text": row.get("text", ""),
                    "file": source_file,
                    "title": row.get("title", Path(source_file).stem),
                    "type": row.get("doc_type", "note"),
                    "folder": row.get("folder", ""),
                    "score": round(score, 4),
                }
            )

            if len(results) >= limit:
                break

        return results

    def search_books(self, query: str, limit: int = 5) -> list[dict]:
        """Search only documents with doc_type='book'."""
        return self.search(query, doc_type="book", limit=limit)

    def search_library(self, query: str, limit: int = 5) -> list[dict]:
        """Search the entire library."""
        return self.search(query, limit=limit)

    def _resolve_document(self, note_path: str) -> SourceDocument | None:
        """Try to read a document from any known source.

        Args:
            note_path: Path relative to a source root

        Returns:
            SourceDocument if found in any source, None otherwise
        """
        path = Path(note_path)
        for source in self._all_sources:
            try:
                return source.read_document(path)
            except (FileNotFoundError, ValueError):
                continue
        return None

    def get_note(self, note_path: str) -> dict | None:
        """Get a specific note by its path.

        Searches across all configured sources (primary library and
        external sources).

        Args:
            note_path: Path relative to source root (e.g., "Books/Clean Architecture.md")

        Returns:
            Dict with note content and metadata, or None if not found
        """
        doc = self._resolve_document(note_path)
        if doc is None:
            return None
        return {
            "path": note_path,
            "title": doc.title,
            "metadata": doc.metadata,
            "content": doc.content,
        }

    def get_linked_notes(self, note_path: str) -> list[str]:
        """Extract internal links from a note.

        Supports both [[wiki links]] and standard [text](path.md) links.

        Args:
            note_path: Path relative to source root

        Returns:
            List of linked note names/paths
        """
        doc = self._resolve_document(note_path)
        if doc is None:
            return []

        # Extract wiki links: [[link]] and [[link|alias]]
        wiki_links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", doc.content)
        # Extract markdown links to .md files: [text](path.md)
        md_links = re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", doc.content)
        return wiki_links + [path for _, path in md_links]

    def list_books(self) -> list[dict]:
        """List all indexed books with metadata from frontmatter.

        Returns title, chunk count, and — when the source file is available —
        author, language, tags, and translation info.
        """
        try:
            df = self._table.to_pandas()

            books_df = df[df["doc_type"] == "book"]
            if books_df.empty:
                return []

            book_stats = books_df.groupby("title").agg(
                chunks=("title", "size"),
                source_file=("source_file", "first"),
            ).reset_index()

            results: list[dict] = []
            for _, row in book_stats.iterrows():
                entry: dict = {
                    "title": row["title"],
                    "chunks": int(row["chunks"]),
                    "file": row["source_file"],
                }

                # Enrich with frontmatter metadata when available
                try:
                    doc = self._resolve_document(row["source_file"])
                    if doc is None:
                        raise FileNotFoundError(row["source_file"])
                    meta = doc.metadata
                    if meta.get("author"):
                        entry["author"] = meta["author"]
                    if meta.get("language"):
                        entry["language"] = meta["language"]
                    if doc.tags:
                        entry["tags"] = doc.tags
                    if meta.get("original_title"):
                        entry["original_title"] = meta["original_title"]
                    if meta.get("translated_from"):
                        entry["translated_from"] = meta["translated_from"]
                except (FileNotFoundError, ValueError):
                    logger.debug("Source file unavailable for %s", row["title"])

                results.append(entry)

            return results

        except Exception as e:
            logger.warning("list_books failed: %s", e)
            return []

    def get_book_outline(self, file_path: str) -> list[dict] | None:
        """Get the heading structure (table of contents) from a document.

        Searches across all configured sources.

        Args:
            file_path: Path relative to source root (e.g. "Books/Clean Architecture.md")

        Returns:
            List of {level, text} dicts for each heading, or None if file not found.
        """
        doc = self._resolve_document(file_path)
        if doc is None:
            return None

        headings: list[dict] = []
        for match in re.finditer(r"^(#{1,3})\s+(.+)$", doc.content, re.MULTILINE):
            headings.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
            })
        return headings

    def browse_topics(self, folder: str | None = None) -> list[dict]:
        """List all topics (tags) with document counts.

        Deduplicates by source file so a multi-chunk document
        only counts once per tag.

        Args:
            folder: Optional folder filter (e.g. "Books")

        Returns:
            Sorted list of {tag, document_count} dicts.
        """
        try:
            df = self._table.to_pandas()

            if folder:
                df = df[df["folder"] == folder]

            # Deduplicate: one row per unique source_file
            unique_docs = df.drop_duplicates(subset=["source_file"])[
                ["source_file", "tags"]
            ]

            tag_counter: Counter[str] = Counter()
            for tags_str in unique_docs["tags"]:
                if not tags_str:
                    continue
                for tag in str(tags_str).split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counter[tag] += 1

            return [
                {"tag": tag, "document_count": count}
                for tag, count in tag_counter.most_common()
            ]

        except Exception as e:
            logger.warning("browse_topics failed: %s", e)
            return []


# Singleton instance
_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    """Get or create the search engine singleton."""
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


