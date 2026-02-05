from __future__ import annotations

from pathlib import Path

from colibri.doc_catalog import compute_summary, prune_missing_files, update_from_index_row


def test_update_from_index_row_and_summary(tmp_path: Path) -> None:
    p1 = tmp_path / "A.md"
    p2 = tmp_path / "B.md"
    p1.write_text("x")
    p2.write_text("y")

    catalog: dict[str, dict] = {}

    update_from_index_row(
        catalog,
        abs_path=p1,
        row={
            "source_file": "A.md",
            "title": "A",
            "doc_type": "book",
            "folder": "",
            "source_name": "Books",
            "source_type": "markdown",
            "tags": "a,b",
        },
        chunk_count=3,
        indexed_at="2026-01-01T00:00:00Z",
    )
    update_from_index_row(
        catalog,
        abs_path=p2,
        row={
            "source_file": "B.md",
            "title": "B",
            "doc_type": "note",
            "folder": "",
            "source_name": "Notes",
            "source_type": "markdown",
            "tags": "b,c",
        },
        chunk_count=1,
        indexed_at="2026-01-01T00:00:00Z",
    )

    summary = compute_summary(catalog, top_tags=10)
    assert summary["doc_count"] == 2
    assert summary["books_count"] == 1
    assert summary["doc_type_counts"]["book"] == 1
    assert summary["doc_type_counts"]["note"] == 1
    tags = {t["tag"]: t["document_count"] for t in summary["top_tags"]}
    assert tags["b"] == 2


def test_prune_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "E.md"
    missing = tmp_path / "M.md"
    existing.write_text("x")

    catalog: dict[str, dict] = {
        str(existing): {"title": "E"},
        str(missing): {"title": "M"},
    }
    deleted = prune_missing_files(catalog)
    assert deleted == 1
    assert str(existing) in catalog
    assert str(missing) not in catalog

