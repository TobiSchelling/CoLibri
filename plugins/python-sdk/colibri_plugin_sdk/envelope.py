"""Helpers for emitting CoLibri document envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib


def sha256_content_hash(markdown: str) -> str:
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_envelope(
    *,
    plugin_id: str,
    connector_instance: str,
    external_id: str,
    doc_id: str,
    title: str,
    markdown: str,
    doc_type: str = "note",
    classification: str = "internal",
    uri: str | None = None,
    tags: list[str] | None = None,
    acl_tags: list[str] | None = None,
    deleted: bool = False,
) -> dict:
    source = {
        "plugin_id": plugin_id,
        "connector_instance": connector_instance,
        "external_id": external_id,
    }
    if uri:
        source["uri"] = uri

    metadata = {
        "doc_type": doc_type,
        "classification": classification,
    }
    if tags:
        metadata["tags"] = tags
    if acl_tags:
        metadata["acl_tags"] = acl_tags

    return {
        "schema_version": 1,
        "source": source,
        "document": {
            "doc_id": doc_id,
            "title": title,
            "markdown": markdown,
            "content_hash": sha256_content_hash(markdown),
            "source_updated_at": utc_now_iso(),
            "deleted": deleted,
        },
        "metadata": metadata,
    }
