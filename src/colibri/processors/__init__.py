"""Document processors for importing various formats into the knowledge base.

This package provides a modular architecture for document format handling.
Each processor extracts content from a specific format and produces a
normalized ExtractedContent object.

Usage:
    from colibri.processors import ProcessorRegistry, ExtractedContent

    # Auto-detect and process
    processor = ProcessorRegistry.get_processor(Path("book.pdf"))
    if processor:
        content = processor.extract(path)

    # List supported formats
    formats = ProcessorRegistry.supported_extensions()
"""

# Import processors to trigger registration (side-effect imports)
from colibri.processors import epub as epub  # noqa: F401
from colibri.processors import pdf as pdf  # noqa: F401
from colibri.processors.base import DocumentProcessor, ExtractedContent
from colibri.processors.registry import ProcessorRegistry

__all__ = [
    "DocumentProcessor",
    "ExtractedContent",
    "ProcessorRegistry",
]
