from __future__ import annotations

from pathlib import Path

from colibri.index_state import (
    append_change_event,
    compute_delta_from_signatures,
    compute_digest_from_signature,
    read_change_events,
)


def test_digest_stable_for_same_signature() -> None:
    sig = {
        "A.md": ("sha256:aaa", 2),
        "B.md": ("sha256:bbb", 1),
    }
    d1 = compute_digest_from_signature(sig)
    d2 = compute_digest_from_signature(sig)
    assert d1 == d2


def test_digest_changes_when_any_file_changes() -> None:
    sig1 = {"A.md": ("sha256:aaa", 2)}
    sig2 = {"A.md": ("sha256:ccc", 2)}
    assert compute_digest_from_signature(sig1) != compute_digest_from_signature(sig2)


def test_delta_added_updated_deleted() -> None:
    before = {
        "A.md": ("sha256:aaa", 1),
        "B.md": ("sha256:bbb", 1),
    }
    after = {
        "A.md": ("sha256:aaa", 1),  # unchanged
        "B.md": ("sha256:ccc", 1),  # updated
        "C.md": ("sha256:ddd", 2),  # added
    }
    delta = compute_delta_from_signatures(before, after)
    assert delta["added"] == ["C.md"]
    assert delta["updated"] == ["B.md"]
    assert delta["deleted"] == []

    delta2 = compute_delta_from_signatures(after, before)
    assert delta2["deleted"] == ["C.md"]


def test_changes_journal_roundtrip(tmp_path: Path) -> None:
    append_change_event(
        data_dir=tmp_path,
        revision=1,
        digest="sha256:x",
        delta={"added": ["A.md"], "updated": [], "deleted": []},
        schema_version=4,
        embedding_model="nomic-embed-text",
    )
    append_change_event(
        data_dir=tmp_path,
        revision=2,
        digest="sha256:y",
        delta={"added": [], "updated": ["A.md"], "deleted": []},
        schema_version=4,
        embedding_model="nomic-embed-text",
    )

    events = read_change_events(data_dir=tmp_path, since_revision=0)
    assert [e["revision"] for e in events] == [1, 2]

    events2 = read_change_events(data_dir=tmp_path, since_revision=1)
    assert [e["revision"] for e in events2] == [2]

