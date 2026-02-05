"""Tests for the translation module."""

import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from colibri.processors.base import ExtractedContent
from colibri.translate import (
    MAX_CHUNK_SIZE,
    _check_claude_cli,
    _group_paragraphs,
    _pick_largest,
    _resolve_language_name,
    _run_claude,
    _split_for_translation,
    _translate_chunk,
    _translate_chunk_claude,
    _translate_title,
    _translate_title_claude,
    detect_model,
    translate_content,
)

# ---------------------------------------------------------------------------
# _resolve_language_name
# ---------------------------------------------------------------------------


class TestResolveLanguageName:
    def test_known_code(self) -> None:
        assert _resolve_language_name("de") == "German"

    def test_known_code_uppercase(self) -> None:
        assert _resolve_language_name("DE") == "German"

    def test_code_with_region(self) -> None:
        assert _resolve_language_name("de-AT") == "German"

    def test_unknown_code(self) -> None:
        assert _resolve_language_name("xx") == "the source language"

    def test_none(self) -> None:
        assert _resolve_language_name(None) == "the source language"

    def test_empty(self) -> None:
        assert _resolve_language_name("") == "the source language"


# ---------------------------------------------------------------------------
# _split_for_translation
# ---------------------------------------------------------------------------


class TestSplitForTranslation:
    def test_chapter_splitting(self) -> None:
        """EPUB-style content with --- separators."""
        chapters = ["Chapter one content.", "Chapter two content.", "Chapter three."]
        content = "\n\n---\n\n".join(chapters)
        result = _split_for_translation(content, "epub")
        assert result == chapters

    def test_paragraph_grouping(self) -> None:
        """Content without separators groups into chunks."""
        # Each paragraph ~500 chars, 20 paragraphs = ~10k chars → multiple chunks
        paras = [f"Paragraph {i}. " * 30 for i in range(20)]
        content = "\n\n".join(paras)
        assert len(content) > MAX_CHUNK_SIZE
        result = _split_for_translation(content, "pdf")
        # Should produce multiple chunks, each under MAX_CHUNK_SIZE
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK_SIZE

    def test_single_small_chunk(self) -> None:
        """Short content stays as a single chunk."""
        result = _split_for_translation("Hello world.", "pdf")
        assert result == ["Hello world."]

    def test_oversized_chapter_gets_subsplit(self) -> None:
        """A chapter exceeding MAX_CHUNK_SIZE is sub-split."""
        big_chapter = ("A long paragraph. " * 300).strip()
        assert len(big_chapter) > MAX_CHUNK_SIZE
        content = f"Small intro.\n\n---\n\n{big_chapter}\n\n---\n\nSmall outro."
        result = _split_for_translation(content, "epub")
        # At least 3 chunks (intro, 1+ from big chapter, outro)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK_SIZE

    def test_empty_chunks_removed(self) -> None:
        """Empty sections between separators are dropped."""
        content = "Content.\n\n---\n\n\n\n---\n\nMore content."
        result = _split_for_translation(content, "epub")
        assert result == ["Content.", "More content."]

    def test_empty_content(self) -> None:
        """Empty input produces empty output."""
        assert _split_for_translation("", "pdf") == []


class TestGroupParagraphs:
    def test_groups_small_paragraphs(self) -> None:
        paras = ["Short para."] * 5
        content = "\n\n".join(paras)
        result = _group_paragraphs(content)
        # All fit in one chunk
        assert len(result) == 1
        assert "Short para." in result[0]

    def test_splits_at_limit(self) -> None:
        # Each paragraph ~100 chars, MAX_CHUNK_SIZE=4000 → ~40 per chunk
        paras = [f"Paragraph number {i:03d}. " * 5 for i in range(100)]
        result = _group_paragraphs("\n\n".join(paras))
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK_SIZE


# ---------------------------------------------------------------------------
# _pick_largest
# ---------------------------------------------------------------------------


class TestPickLargest:
    def test_prefers_larger_variant(self) -> None:
        models = ["qwen3:8b-q4_K_M", "qwen3:14b-q4_K_M", "qwen3:1.5b"]
        assert _pick_largest(models) == "qwen3:14b-q4_K_M"

    def test_single_model(self) -> None:
        assert _pick_largest(["mistral:latest"]) == "mistral:latest"

    def test_no_size_in_tag(self) -> None:
        """Models without size numbers still work."""
        models = ["mistral:latest", "mistral:instruct"]
        result = _pick_largest(models)
        assert result in models


# ---------------------------------------------------------------------------
# detect_model
# ---------------------------------------------------------------------------


def _mock_tags_response(models: list[str]) -> MagicMock:
    """Create a mock response for GET /api/tags."""
    mock = MagicMock()
    mock.json.return_value = {
        "models": [{"name": m} for m in models]
    }
    mock.raise_for_status = MagicMock()
    return mock


class TestDetectModel:
    @patch("colibri.translate.httpx.get")
    def test_preference_ranking(self, mock_get: MagicMock) -> None:
        """Selects the highest-ranked available model."""
        mock_get.return_value = _mock_tags_response(
            ["llama3:8b", "qwen2.5:14b", "mistral:7b"]
        )
        result = detect_model(base_url="http://test:11434")
        assert result == "qwen2.5:14b"

    @patch("colibri.translate.httpx.get")
    def test_explicit_override(self, mock_get: MagicMock) -> None:
        """Preferred model takes priority over ranking."""
        mock_get.return_value = _mock_tags_response(
            ["qwen3:14b", "mistral:7b"]
        )
        result = detect_model(
            base_url="http://test:11434",
            preferred_model="mistral:7b",
        )
        assert result == "mistral:7b"

    @patch("colibri.translate.httpx.get")
    def test_preferred_model_prefix_match(self, mock_get: MagicMock) -> None:
        """Preferred model does prefix matching."""
        mock_get.return_value = _mock_tags_response(
            ["qwen3:14b-q4_K_M", "qwen3:8b"]
        )
        result = detect_model(
            base_url="http://test:11434",
            preferred_model="qwen3",
        )
        # Should pick the largest variant
        assert result == "qwen3:14b-q4_K_M"

    @patch("colibri.translate.httpx.get")
    def test_largest_variant_selection(self, mock_get: MagicMock) -> None:
        """Among multiple variants, prefers the largest."""
        mock_get.return_value = _mock_tags_response(
            ["qwen3:1.5b", "qwen3:8b", "qwen3:14b"]
        )
        result = detect_model(base_url="http://test:11434")
        assert result == "qwen3:14b"

    @patch("colibri.translate.httpx.get")
    def test_no_models_error(self, mock_get: MagicMock) -> None:
        """Raises RuntimeError when no models are installed."""
        mock_get.return_value = _mock_tags_response([])
        with pytest.raises(RuntimeError, match="No Ollama models installed"):
            detect_model(base_url="http://test:11434")

    @patch("colibri.translate.httpx.get")
    def test_no_suitable_model_error(self, mock_get: MagicMock) -> None:
        """Raises RuntimeError when no preferred model family matches."""
        mock_get.return_value = _mock_tags_response(
            ["nomic-embed-text:latest", "some-obscure-model:1b"]
        )
        with pytest.raises(RuntimeError, match="No suitable translation model"):
            detect_model(base_url="http://test:11434")

    @patch("colibri.translate.httpx.get")
    def test_ollama_down_error(self, mock_get: MagicMock) -> None:
        """Raises RuntimeError when Ollama is unreachable."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            detect_model(base_url="http://test:11434")


# ---------------------------------------------------------------------------
# _translate_chunk / _translate_title
# ---------------------------------------------------------------------------


def _mock_chat_response(text: str) -> MagicMock:
    """Create a mock response for POST /api/chat."""
    mock = MagicMock()
    mock.json.return_value = {"message": {"content": text}}
    mock.raise_for_status = MagicMock()
    return mock


class TestTranslateChunk:
    @patch("colibri.translate.httpx.post")
    def test_basic_translation(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_chat_response("Hello world")
        result = _translate_chunk(
            "Hallo Welt", "qwen3:14b", "German", "http://test:11434"
        )
        assert result == "Hello world"

        # Verify the API call structure
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://test:11434/api/chat"
        payload = call_args[1]["json"]
        assert payload["model"] == "qwen3:14b"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert "German" in payload["messages"][0]["content"]


class TestTranslateTitle:
    @patch("colibri.translate.httpx.post")
    def test_title_translation(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_chat_response("Clean Architecture")
        result = _translate_title(
            "Saubere Architektur", "qwen3:14b", "German", "http://test:11434"
        )
        assert result == "Clean Architecture"


# ---------------------------------------------------------------------------
# translate_content (full pipeline)
# ---------------------------------------------------------------------------


def _make_content(
    content: str = "Hallo Welt",
    language: str = "de",
    source_format: str = "epub",
    title: str = "Mein Buch",
    description: str | None = "Eine Beschreibung",
) -> ExtractedContent:
    """Helper to create test ExtractedContent."""
    return ExtractedContent(
        title=title,
        content=content,
        source_path=Path("/test/book.epub"),
        source_format=source_format,
        language=language,
        description=description,
        extracted_at=datetime(2025, 1, 1),
    )


class TestTranslateContent:
    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_full_pipeline(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Full translation pipeline with model detection."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])
        mock_post.return_value = _mock_chat_response("Translated text")

        content = _make_content()
        result = translate_content(content, base_url="http://test:11434")

        assert result.content == "Translated text"
        assert result.title == "Translated text"
        assert result.description == "Translated text"
        assert result.language == "de"  # preserved
        assert result.metadata["translated_from"] == "de"
        assert result.metadata["translation_model"] == "qwen3:14b"

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_explicit_model(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Explicit model skips auto-detection."""
        # GET is still called by detect_model when model is not None
        # but we pass explicit model, so detect_model is skipped
        mock_post.return_value = _mock_chat_response("English text")

        content = _make_content()
        result = translate_content(
            content, model="mistral:7b", base_url="http://test:11434"
        )

        assert result.metadata["translation_model"] == "mistral:7b"
        # httpx.get should NOT have been called (no auto-detection)
        mock_get.assert_not_called()

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_partial_failure_keeps_original(
        self, mock_get: MagicMock, mock_post: MagicMock
    ) -> None:
        """When a chunk fails, the original text is kept."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])

        chapters = ["Kapitel eins", "Kapitel zwei", "Kapitel drei"]
        content = _make_content(content="\n\n---\n\n".join(chapters))

        # First chunk succeeds, second fails, third succeeds
        mock_post.side_effect = [
            _mock_chat_response("Chapter one"),    # chunk 1
            httpx.HTTPError("timeout"),             # chunk 2 fails
            _mock_chat_response("Chapter three"),   # chunk 3
            _mock_chat_response("My Book"),         # title
            _mock_chat_response("A description"),   # description
        ]

        result = translate_content(content, base_url="http://test:11434")

        parts = result.content.split("\n\n---\n\n")
        assert parts[0] == "Chapter one"
        assert parts[1] == "Kapitel zwei"  # kept original
        assert parts[2] == "Chapter three"

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_no_description(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Content without description skips description translation."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])
        mock_post.return_value = _mock_chat_response("Translated")

        content = _make_content(description=None)
        result = translate_content(content, base_url="http://test:11434")

        assert result.description is None
        # Should have called POST twice: 1 chunk + 1 title (no description)
        assert mock_post.call_count == 2

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_metadata_preserved(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Existing metadata is preserved alongside new translation fields."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])
        mock_post.return_value = _mock_chat_response("Translated")

        content = _make_content()
        content.metadata["chapter_count"] = 10

        result = translate_content(content, base_url="http://test:11434")

        assert result.metadata["chapter_count"] == 10
        assert result.metadata["translated_from"] == "de"
        assert result.metadata["translation_model"] == "qwen3:14b"

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_unknown_language(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Unknown language code records 'unknown' in metadata when None."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])
        mock_post.return_value = _mock_chat_response("Translated")

        content = _make_content(language=None)
        result = translate_content(content, base_url="http://test:11434")

        assert result.metadata["translated_from"] == "unknown"

    @patch("colibri.translate.httpx.post")
    @patch("colibri.translate.httpx.get")
    def test_title_failure_keeps_original(
        self, mock_get: MagicMock, mock_post: MagicMock
    ) -> None:
        """If title translation fails, the original title is kept."""
        mock_get.return_value = _mock_tags_response(["qwen3:14b"])

        mock_post.side_effect = [
            _mock_chat_response("Translated content"),  # chunk
            httpx.HTTPError("timeout"),                  # title fails
            _mock_chat_response("A description"),        # description
        ]

        content = _make_content()
        result = translate_content(content, base_url="http://test:11434")

        assert result.title == "Mein Buch"  # original preserved
        assert result.content == "Translated content"


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------


class TestCheckClaudeCli:
    @patch("colibri.translate.shutil.which", return_value="/usr/local/bin/claude")
    def test_found(self, mock_which: MagicMock) -> None:
        assert _check_claude_cli() == "/usr/local/bin/claude"

    @patch("colibri.translate.shutil.which", return_value=None)
    def test_not_found(self, mock_which: MagicMock) -> None:
        with pytest.raises(RuntimeError, match="claude CLI not found"):
            _check_claude_cli()


class TestRunClaude:
    @patch("colibri.translate.subprocess.run")
    def test_basic_call(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Hello world\n", stderr=""
        )
        result = _run_claude("Translate: Hallo Welt")
        assert result == "Hello world"

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd

    @patch("colibri.translate.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "claude")
        with pytest.raises(subprocess.CalledProcessError):
            _run_claude("test prompt")


class TestTranslateChunkClaude:
    @patch("colibri.translate._run_claude", return_value="Hello world")
    def test_chunk_translation(self, mock_run: MagicMock) -> None:
        result = _translate_chunk_claude("Hallo Welt", "German")
        assert result == "Hello world"
        # Verify prompt includes language and system instructions
        prompt = mock_run.call_args[0][0]
        assert "German" in prompt


class TestTranslateTitleClaude:
    @patch("colibri.translate._run_claude", return_value="Clean Architecture")
    def test_title_translation(self, mock_run: MagicMock) -> None:
        result = _translate_title_claude("Saubere Architektur", "German")
        assert result == "Clean Architecture"


class TestTranslateContentClaude:
    @patch("colibri.translate._check_claude_cli", return_value="/usr/local/bin/claude")
    @patch("colibri.translate._run_claude", return_value="Translated text")
    def test_full_pipeline(self, mock_run: MagicMock, mock_check: MagicMock) -> None:
        """Full pipeline with claude backend."""
        content = _make_content()
        result = translate_content(content, backend="claude")

        assert result.content == "Translated text"
        assert result.title == "Translated text"
        assert result.metadata["translated_from"] == "de"
        assert result.metadata["translation_model"] == "claude:haiku"

    @patch("colibri.translate._check_claude_cli", return_value="/usr/local/bin/claude")
    @patch("colibri.translate._run_claude")
    def test_partial_failure(self, mock_run: MagicMock, mock_check: MagicMock) -> None:
        """Chunk failure keeps original text."""
        chapters = ["Kapitel eins", "Kapitel zwei"]
        content = _make_content(content="\n\n---\n\n".join(chapters))

        mock_run.side_effect = [
            "Chapter one",                               # chunk 1
            subprocess.CalledProcessError(1, "claude"),  # chunk 2 fails
            "My Book",                                   # title
            "A description",                             # description
        ]

        result = translate_content(content, backend="claude")

        parts = result.content.split("\n\n---\n\n")
        assert parts[0] == "Chapter one"
        assert parts[1] == "Kapitel zwei"  # original kept

    @patch("colibri.translate._check_claude_cli", return_value="/usr/local/bin/claude")
    @patch("colibri.translate._run_claude", return_value="Translated")
    def test_explicit_model(self, mock_run: MagicMock, mock_check: MagicMock) -> None:
        """Explicit model passed to claude backend."""
        content = _make_content()
        result = translate_content(content, model="sonnet", backend="claude")

        assert result.metadata["translation_model"] == "claude:sonnet"
        # Verify model was passed to _run_claude
        assert any("sonnet" in str(call) for call in mock_run.call_args_list)
