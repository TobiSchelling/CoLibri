#!/usr/bin/env python3
"""MCP Server for CoLibri semantic search."""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from colibri.query import get_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("colibri-mcp")

# Create MCP server
server = Server("colibri")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to Claude."""
    return [
        Tool(
            name="search_library",
            description=(
                "Search across all indexed content sources including technical books, "
                "architecture documents, notes, and other reference materials. "
                "Performs semantic search to find relevant passages. "
                "Use for broad knowledge queries across all indexed content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language description of what to search for. "
                            "Be specific. Examples: 'microservice testing strategies', "
                            "'clean architecture principles', 'API design patterns'"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5, max: 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_books",
            description=(
                "Search only imported technical books and reference materials. "
                "Filters to documents with type 'book'. Use when specifically "
                "looking for authoritative book content rather than notes or "
                "project documentation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the books",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5, max: 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_note",
            description=(
                "Retrieve a specific document by its path. "
                "Returns full content with metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to source root, e.g., "
                            "'Books/Clean Architecture.md' or 'Notes/Testing.md'"
                        ),
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="get_linked_notes",
            description=(
                "Get all documents linked from a specific document via "
                "[[wiki links]] or [markdown](links). "
                "Use to explore connections between concepts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="list_books",
            description=(
                "List all indexed books with metadata. Returns title, author, "
                "language, tags, chunk count, and translation info. Use to "
                "discover what books are available before searching."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_book_outline",
            description=(
                "Get the table of contents (headings) from a book. "
                "Use after list_books to understand a book's structure "
                "before searching for specific topics. Returns heading hierarchy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": (
                            "Path to the book file, as returned by list_books. "
                            "Example: 'Books/Clean Architecture.md'"
                        ),
                    },
                },
                "required": ["file"],
            },
        ),
        Tool(
            name="browse_topics",
            description=(
                "List all topics (tags) with document counts. Use to "
                "discover available topics before searching. Optionally "
                "filter by folder."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": (
                            "Optional folder to filter by (e.g. 'Books', 'Notes'). "
                            "Omit to include all folders."
                        ),
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool invocations from Claude."""
    try:
        engine = get_engine()

        if name == "search_library":
            results = engine.search_library(
                arguments["query"],
                limit=min(arguments.get("limit", 5), 10),
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "query": arguments["query"],
                            "total_results": len(results),
                            "results": results,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "search_books":
            results = engine.search_books(
                arguments["query"],
                limit=min(arguments.get("limit", 5), 10),
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "query": arguments["query"],
                            "total_results": len(results),
                            "results": results,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "get_note":
            note = engine.get_note(arguments["path"])
            if note:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(note, indent=2),
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "Note not found"}),
                )
            ]

        elif name == "get_linked_notes":
            links = engine.get_linked_notes(arguments["path"])
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"path": arguments["path"], "links": links}),
                )
            ]

        elif name == "list_books":
            books = engine.list_books()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_books": len(books),
                            "books": books,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "get_book_outline":
            outline = engine.get_book_outline(arguments["file"])
            if outline is None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "File not found"}),
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "file": arguments["file"],
                            "headings": outline,
                            "total_headings": len(outline),
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "browse_topics":
            topics = engine.browse_topics(folder=arguments.get("folder"))
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_topics": len(topics),
                            "topics": topics,
                        },
                        indent=2,
                    ),
                )
            ]

        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )
            ]

    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": str(e)}),
            )
        ]


async def main() -> None:
    """Run the MCP server."""
    logger.info("Starting CoLibri MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_server() -> None:
    """Entry point for running the server."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
