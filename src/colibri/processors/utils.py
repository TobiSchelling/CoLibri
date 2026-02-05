"""Shared utilities for document processors.

This module contains functions used across multiple processors:
- Text cleaning and normalization
- Filename sanitization
- Frontmatter generation
- Markdown document formatting
"""

import re
from pathlib import Path
from typing import Any

import yaml

from colibri.processors.base import ExtractedContent


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Convert a title to a safe filename.

    Removes characters that are problematic in filenames across
    different operating systems.

    Args:
        title: The title to convert
        max_length: Maximum filename length (default: 100)

    Returns:
        A sanitized filename (without extension)

    Example:
        >>> sanitize_filename("Clean Code: A Handbook")
        'Clean Code A Handbook'
    """
    # Keep only alphanumeric, spaces, hyphens, underscores
    safe = re.sub(r"[^\w\s\-]", "", title)
    # Collapse multiple spaces
    safe = re.sub(r"\s+", " ", safe).strip()
    # Truncate if needed
    return safe[:max_length]


def clean_text(text: str) -> str:
    """Normalize unicode and clean up text.

    Performs common text cleanup operations:
    - Converts smart quotes to straight quotes
    - Normalizes dashes and ellipses
    - Removes control characters
    - Cleans up excessive whitespace

    Args:
        text: The text to clean

    Returns:
        Cleaned text
    """
    # Unicode replacements for common typography
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote (apostrophe)
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove control characters (except newline, tab, carriage return)
    # Includes bell (\x07), backspace (\x08), and other control chars
    text = re.sub(r"[\x00-\x06\x07\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text


def clean_pdf_text(text: str) -> str:
    """Additional cleanup specific to PDF extraction.

    Handles PDF-specific artifacts like:
    - Replacement characters from encoding issues
    - TOC leader dots
    - Excessive whitespace from layout extraction

    Args:
        text: Text extracted from PDF

    Returns:
        Cleaned text
    """
    # First apply general cleaning
    text = clean_text(text)

    # Clean up sequences of replacement characters (often from garbled TOC dots)
    # 3+ replacement chars → remove (likely TOC leader dots)
    text = re.sub(r"\ufffd{3,}", "", text)
    # Single/double replacement chars → remove (likely broken quotes)
    text = re.sub(r"\ufffd{1,2}", "", text)

    # Clean up TOC dot leaders (e.g., "Chapter 1.......................5")
    text = re.sub(r"\.{3,}", "...", text)

    # Clean up excessive whitespace
    text = re.sub(r" {3,}", "  ", text)  # Max 2 consecutive spaces
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # Max 3 consecutive newlines

    return text


def clean_html_whitespace(text: str) -> str:
    """Clean up whitespace from HTML-to-Markdown conversion.

    Collapses multiple empty lines while preserving paragraph breaks.

    Args:
        text: Text converted from HTML

    Returns:
        Text with normalized whitespace
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    prev_empty = False

    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        cleaned_lines.append(line)
        prev_empty = is_empty

    return "\n".join(cleaned_lines).strip()


def format_frontmatter(metadata: dict[str, Any]) -> str:
    """Generate YAML frontmatter block.

    Args:
        metadata: Dictionary of frontmatter fields

    Returns:
        YAML frontmatter as string (including --- delimiters)

    Example:
        >>> format_frontmatter({"title": "My Book", "type": "book"})
        '---\\ntitle: My Book\\ntype: book\\n---\\n'
    """
    # Use yaml.dump for proper escaping, but with cleaner formatting
    yaml_content = yaml.dump(
        metadata,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return f"---\n{yaml_content}---\n"


def build_frontmatter_dict(
    content: ExtractedContent,
    extra_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build frontmatter dictionary from ExtractedContent.

    Creates a structured frontmatter dict with standard fields
    plus any available bibliographic metadata.

    Args:
        content: The extracted content
        extra_tags: Additional tags to include

    Returns:
        Dictionary ready for format_frontmatter()
    """
    frontmatter: dict[str, Any] = {
        "title": content.title,
        "type": "book",
        f"source_{content.source_format}": content.source_path.name,
        "imported": content.extracted_at.isoformat(),
    }

    # Add optional bibliographic fields if present
    if content.author:
        frontmatter["author"] = content.author
    if content.publisher:
        frontmatter["publisher"] = content.publisher
    if content.language:
        frontmatter["language"] = content.language
    if content.isbn:
        frontmatter["isbn"] = content.isbn

    # Translation metadata
    if content.metadata.get("original_title"):
        frontmatter["original_title"] = content.metadata["original_title"]
    if content.metadata.get("translated_from"):
        frontmatter["translated_from"] = content.metadata["translated_from"]
    if content.metadata.get("translation_model"):
        frontmatter["translation_model"] = content.metadata["translation_model"]

    # Build tags list
    tags = ["book", "imported", content.source_format]
    if content.metadata.get("translated_from"):
        tags.append("translated")
    if extra_tags:
        tags.extend(extra_tags)
    frontmatter["tags"] = tags

    return frontmatter


def generate_document(
    content: ExtractedContent,
    include_source_callout: bool = True,
    extra_tags: list[str] | None = None,
) -> str:
    """Format ExtractedContent as a markdown document with frontmatter.

    Creates a complete document with:
    - YAML frontmatter
    - H1 title
    - Optional source callout
    - Main content

    Args:
        content: The extracted content
        include_source_callout: Whether to add the source info callout
        extra_tags: Additional tags for frontmatter

    Returns:
        Complete markdown document as string
    """
    # Build frontmatter
    frontmatter_dict = build_frontmatter_dict(content, extra_tags)
    frontmatter = format_frontmatter(frontmatter_dict)

    # Build document parts
    parts = [frontmatter, f"# {content.title}", ""]

    if include_source_callout:
        date_str = content.extracted_at.strftime("%Y-%m-%d")
        callout = f"> [!info] Source\n> Imported from `{content.source_path.name}` on {date_str}"
        parts.append(callout)
        parts.append("")

    parts.append(content.content)

    return "\n".join(parts)


def write_to_library(
    content: ExtractedContent,
    output_dir: Path,
    include_source_callout: bool = True,
) -> Path:
    """Write extracted content to the library as a markdown file.

    Args:
        content: The extracted content
        output_dir: Directory to write to (e.g., library/Books)
        include_source_callout: Whether to include source info

    Returns:
        Path to the created file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate document
    document = generate_document(content, include_source_callout)

    # Create safe filename
    safe_title = sanitize_filename(content.title)
    output_path = output_dir / f"{safe_title}.md"

    # Write file
    output_path.write_text(document, encoding="utf-8")

    return output_path
