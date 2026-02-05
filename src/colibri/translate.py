"""Translation module for converting non-English content to English.

Supports two backends:
- **ollama**: Local Ollama generation model (default)
- **claude**: Claude Code CLI (``claude -p``) using your existing authentication

This ensures embeddings (nomic-embed-text) and search queries work well
against English text.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from colibri.config import OLLAMA_BASE_URL

if TYPE_CHECKING:
    from colibri.processors.base import ExtractedContent

logger = logging.getLogger(__name__)
console = Console()

# Ranked by quality for translation tasks.
# Prefix-matched against Ollama model names (e.g. "qwen3" matches "qwen3:14b-q4_K_M").
PREFERRED_MODELS = [
    "qwen3",
    "qwen2.5",
    "gemma3",
    "llama3.3",
    "llama3.1",
    "phi4",
    "mistral",
    "gemma2",
    "llama3",
]

# Maximum characters per translation chunk.
# Larger chunks give better context but cost more generation time.
MAX_CHUNK_SIZE = 4000

# Claude Haiku handles large contexts efficiently, so use bigger chunks
# to reduce the number of subprocess calls (~4x fewer round trips).
MAX_CHUNK_SIZE_CLAUDE = 16000

# Ollama generation timeout (translation is slow).
TRANSLATE_TIMEOUT = 300.0

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the following text to English. "
    "Rules:\n"
    "- Preserve ALL markdown formatting exactly (headings, lists, bold, italic, "
    "code blocks, links, etc.)\n"
    "- Translate only the human-readable text content\n"
    "- Do NOT add any commentary, notes, or explanations\n"
    "- Do NOT wrap the output in code fences or quotes\n"
    "- Output ONLY the translated text, nothing else"
)

TITLE_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the following title to English. "
    "Output ONLY the translated title, nothing else. No quotes, no explanation."
)

_LANGUAGE_NAMES: dict[str, str] = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
}


def _resolve_language_name(code: str | None) -> str:
    """Map an ISO 639-1 language code to a human-readable name.

    Falls back to a generic description for unknown codes.
    """
    if not code:
        return "the source language"
    return _LANGUAGE_NAMES.get(code.lower().split("-")[0], "the source language")


def detect_model(
    base_url: str = OLLAMA_BASE_URL,
    preferred_model: str | None = None,
) -> str:
    """Find the best available Ollama generation model for translation.

    Queries the Ollama API for installed models and selects the best match
    from the preference list. Prefers the largest variant when multiple
    sizes of the same family are installed.

    Args:
        base_url: Ollama API base URL.
        preferred_model: Explicit model override (tried first).

    Returns:
        Full model name string (e.g. "qwen3:14b-q4_K_M").

    Raises:
        RuntimeError: If Ollama is unreachable or no suitable model is found.
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
    except httpx.ConnectError:
        msg = (
            "Cannot connect to Ollama. Is it running?\n"
            "  Start with: ollama serve"
        )
        raise RuntimeError(msg) from None
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc

    available = [m["name"] for m in resp.json().get("models", [])]

    if not available:
        msg = (
            "No Ollama models installed.\n"
            f"  Try: ollama pull {PREFERRED_MODELS[0]}"
        )
        raise RuntimeError(msg)

    # If user specified a model, try exact match or prefix match
    if preferred_model:
        exact = [m for m in available if m == preferred_model]
        if exact:
            return exact[0]
        prefix = [m for m in available if m.startswith(preferred_model)]
        if prefix:
            return _pick_largest(prefix)

    # Walk the preference list and find the first match
    for pref in PREFERRED_MODELS:
        matches = [m for m in available if m.split(":")[0] == pref or m.startswith(f"{pref}:")]
        if matches:
            return _pick_largest(matches)

    msg = (
        "No suitable translation model found.\n"
        f"  Available: {', '.join(available[:5])}\n"
        f"  Try: ollama pull {PREFERRED_MODELS[0]}"
    )
    raise RuntimeError(msg)


def _pick_largest(models: list[str]) -> str:
    """Among model variants, prefer the largest by name heuristic.

    Ollama tags often include size hints like "14b", "8b", etc.
    We sort descending by the first number found in the tag portion.
    """
    import re

    def _size_key(name: str) -> float:
        tag = name.split(":")[-1] if ":" in name else ""
        # Extract first number (e.g. "14" from "14b-q4_K_M")
        match = re.search(r"(\d+\.?\d*)", tag)
        return float(match.group(1)) if match else 0.0

    result: str = sorted(models, key=_size_key, reverse=True)[0]
    return result


def _split_for_translation(
    content: str,
    source_format: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[str]:
    """Split content into translation-sized chunks.

    For both EPUBs and PDFs, tries chapter boundaries first (``---`` separators).
    Falls back to paragraph grouping for content without separators.

    Args:
        content: Full document content.
        source_format: "epub", "pdf", etc.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks, each at most *max_chunk_size* characters.
    """
    separator = "\n\n---\n\n"
    chunks = (
        content.split(separator)
        if separator in content
        else _group_paragraphs(content, max_chunk_size)
    )

    # Sub-split any oversized chunks
    result: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= max_chunk_size:
            result.append(chunk)
        else:
            result.extend(_subsplit_chunk(chunk, max_chunk_size))

    return result


def _group_paragraphs(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Group paragraphs into chunks up to *max_chunk_size* characters."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # +2 for the paragraph separator we'll rejoin with
        addition = len(para) + (2 if current else 0)
        if current_len + addition > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += addition

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _subsplit_chunk(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split an oversized chunk on paragraph boundaries.

    Falls back to hard character-boundary splits when the text
    has no paragraph breaks (e.g. a single long paragraph).
    """
    paragraphs = text.split("\n\n")

    # If there are no paragraph breaks, hard-split on character boundaries
    if len(paragraphs) <= 1:
        return _hard_split(text, max_chunk_size)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        addition = len(para) + (2 if current else 0)
        if current_len + addition > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += addition

    if current:
        chunks.append("\n\n".join(current))

    # Any resulting chunk that's still too large gets hard-split
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            result.append(chunk)
        else:
            result.extend(_hard_split(chunk, max_chunk_size))

    return result


def _hard_split(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text at *max_chunk_size* character boundaries."""
    chunks: list[str] = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i : i + max_chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def _translate_chunk(
    text: str,
    model: str,
    source_language: str,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Translate a single text chunk via Ollama chat API.

    Args:
        text: Text to translate.
        model: Ollama model name.
        source_language: Human-readable source language name.
        base_url: Ollama API base URL.

    Returns:
        Translated text.

    Raises:
        httpx.HTTPError: If the API call fails.
    """
    system_msg = f"The source language is {source_language}. {SYSTEM_PROMPT}"

    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8192},
        },
        timeout=TRANSLATE_TIMEOUT,
    )
    resp.raise_for_status()
    result: str = resp.json()["message"]["content"]
    return result


def _translate_title(
    title: str,
    model: str,
    source_language: str,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Translate a document title via Ollama chat API."""
    system_msg = f"The source language is {source_language}. {TITLE_SYSTEM_PROMPT}"

    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": title},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    result: str = resp.json()["message"]["content"]
    return result.strip()


# ---------------------------------------------------------------------------
# Claude backend (uses ``claude -p`` CLI)
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "haiku"


def _check_claude_cli() -> str:
    """Verify the ``claude`` CLI is available.

    Returns:
        Path to the claude binary.

    Raises:
        RuntimeError: If the CLI is not found.
    """
    path = shutil.which("claude")
    if not path:
        msg = (
            "claude CLI not found on PATH.\n"
            "  Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
        )
        raise RuntimeError(msg)
    return path


def _run_claude(prompt: str, model: str = CLAUDE_MODEL) -> str:
    """Run a single ``claude -p`` invocation and return the output.

    Args:
        prompt: The full prompt (system instructions + text).
        model: Claude model alias (e.g. "haiku", "sonnet").

    Returns:
        Claude's text response.

    Raises:
        subprocess.CalledProcessError: If the CLI exits non-zero.
    """
    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model", model,
            "--allowedTools", "",
            "--no-session-persistence",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout.strip()


def _translate_chunk_claude(
    text: str,
    source_language: str,
    model: str = CLAUDE_MODEL,
) -> str:
    """Translate a single text chunk via the Claude CLI."""
    prompt = (
        f"The source language is {source_language}. {SYSTEM_PROMPT}\n\n"
        f"{text}"
    )
    return _run_claude(prompt, model=model)


def _translate_title_claude(
    title: str,
    source_language: str,
    model: str = CLAUDE_MODEL,
) -> str:
    """Translate a document title via the Claude CLI."""
    prompt = (
        f"The source language is {source_language}. {TITLE_SYSTEM_PROMPT}\n\n"
        f"{title}"
    )
    return _run_claude(prompt, model=model)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def translate_content(
    content: ExtractedContent,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    backend: str = "ollama",
) -> ExtractedContent:
    """Translate extracted content to English.

    This is the main public API. It splits the content into chunks,
    translates each with a progress bar, and returns a new
    ExtractedContent with English text.

    Args:
        content: The extracted (non-English) document content.
        model: Model name/alias. Auto-detects if None.
        base_url: Ollama API base URL (ignored for claude backend).
        backend: Translation backend — "ollama" or "claude".

    Returns:
        New ExtractedContent with translated text and updated metadata.

    Raises:
        RuntimeError: If the backend is unavailable.
    """
    use_claude = backend == "claude"
    source_language = _resolve_language_name(content.language)

    if use_claude:
        _check_claude_cli()
        resolved_model = model or CLAUDE_MODEL
        console.print(
            f"[cyan]Translating from {source_language} "
            f"using Claude ({resolved_model})[/cyan]"
        )
    else:
        resolved_model = model or detect_model(base_url=base_url)
        console.print(
            f"[cyan]Translating from {source_language} "
            f"using {resolved_model}[/cyan]"
        )

    # Split content into chunks (larger for Claude — fewer round trips)
    chunk_size = MAX_CHUNK_SIZE_CLAUDE if use_claude else MAX_CHUNK_SIZE
    chunks = _split_for_translation(content.content, content.source_format, chunk_size)
    translated_chunks: list[str] = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Translating 0/{len(chunks)}...", total=len(chunks)
        )

        for i, chunk in enumerate(chunks):
            progress.update(
                task, description=f"Translating {i + 1}/{len(chunks)}..."
            )
            try:
                if use_claude:
                    translated = _translate_chunk_claude(
                        chunk, source_language, model=resolved_model
                    )
                else:
                    translated = _translate_chunk(
                        chunk, resolved_model, source_language, base_url
                    )
                translated_chunks.append(translated)
            except Exception:
                logger.warning("Failed to translate chunk %d, keeping original", i + 1)
                translated_chunks.append(chunk)
                failed += 1
            progress.advance(task)

    # Rejoin with the same separator used in splitting
    separator = "\n\n---\n\n"
    if separator in content.content:
        translated_content = separator.join(translated_chunks)
    else:
        translated_content = "\n\n".join(translated_chunks)

    # Translate title
    translated_title = content.title
    try:
        if use_claude:
            translated_title = _translate_title_claude(
                content.title, source_language, model=resolved_model
            )
        else:
            translated_title = _translate_title(
                content.title, resolved_model, source_language, base_url
            )
    except Exception:
        logger.warning("Failed to translate title, keeping original")

    # Translate description if present
    translated_description = content.description
    if content.description:
        try:
            if use_claude:
                translated_description = _translate_title_claude(
                    content.description, source_language, model=resolved_model
                )
            else:
                translated_description = _translate_title(
                    content.description, resolved_model, source_language, base_url
                )
        except Exception:
            logger.warning("Failed to translate description, keeping original")

    # Report summary
    total = len(chunks)
    if failed:
        console.print(
            f"[yellow]Translated {total - failed}/{total} sections "
            f"({failed} failed, kept original)[/yellow]"
        )
    else:
        console.print(f"[green]Translated {total}/{total} sections[/green]")

    # Build new metadata
    new_metadata = dict(content.metadata)
    new_metadata["original_title"] = content.title
    new_metadata["translated_from"] = content.language or "unknown"
    display_model = f"claude:{resolved_model}" if use_claude else resolved_model
    new_metadata["translation_model"] = display_model

    return dataclasses.replace(
        content,
        title=translated_title,
        content=translated_content,
        description=translated_description,
        metadata=new_metadata,
    )
