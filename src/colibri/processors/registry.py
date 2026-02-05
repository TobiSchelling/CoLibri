"""Processor registry for auto-discovery and format routing.

The registry maintains a list of available processors and provides
methods to find the appropriate processor for a given file.
"""

from pathlib import Path
from typing import TypeVar

from colibri.processors.base import DocumentProcessor

T = TypeVar("T", bound=type[DocumentProcessor])


class ProcessorRegistry:
    """Central registry for document processors.

    Processors register themselves using the @register decorator.
    The registry provides lookup by file path or extension.

    Example:
        @ProcessorRegistry.register
        class PDFProcessor(DocumentProcessor):
            extensions = ['.pdf']
            ...

        # Later, find the right processor
        processor = ProcessorRegistry.get_processor(Path("book.pdf"))
        if processor:
            content = processor.extract(path)
    """

    _processors: list[type[DocumentProcessor]] = []

    @classmethod
    def register(cls, processor_cls: T) -> T:
        """Register a processor class.

        Can be used as a decorator:
            @ProcessorRegistry.register
            class MyProcessor(DocumentProcessor):
                ...

        Args:
            processor_cls: The processor class to register

        Returns:
            The same class (allows decorator usage)
        """
        if processor_cls not in cls._processors:
            cls._processors.append(processor_cls)
        return processor_cls

    @classmethod
    def get_processor(cls, path: Path) -> DocumentProcessor | None:
        """Find a processor that can handle the given file.

        Iterates through registered processors and returns the first
        one that can handle the file (based on extension or other criteria).

        Args:
            path: Path to the file

        Returns:
            An instantiated processor, or None if no processor found
        """
        for processor_cls in cls._processors:
            processor = processor_cls()
            if processor.can_handle(path):
                return processor
        return None

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """List all supported file extensions.

        Returns:
            List of extensions (e.g., ['.pdf', '.epub'])
        """
        extensions: list[str] = []
        for processor_cls in cls._processors:
            extensions.extend(processor_cls.extensions)
        return sorted(set(extensions))

    @classmethod
    def list_processors(cls) -> list[tuple[str, list[str]]]:
        """List all registered processors with their extensions.

        Returns:
            List of (processor_name, extensions) tuples
        """
        return [(p.__name__, p.extensions) for p in cls._processors]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered processors (useful for testing)."""
        cls._processors = []
