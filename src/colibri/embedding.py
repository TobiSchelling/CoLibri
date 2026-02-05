"""Ollama embedding API client.

Wraps the Ollama /api/embed endpoint with batching support.
"""

from __future__ import annotations

import httpx

from colibri.config import EMBEDDING_MODEL, OLLAMA_BASE_URL

# Maximum texts per Ollama embedding batch request
EMBED_BATCH_SIZE = 32


def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[list[float]]:
    """Embed a list of texts using Ollama's /api/embed endpoint.

    Batches requests to stay within limits.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        all_embeddings.extend(data["embeddings"])

    return all_embeddings
