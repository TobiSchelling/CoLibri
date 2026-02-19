#!/usr/bin/env python3
"""Filesystem markdown plugin example.

Reads plugin config from stdin JSON:
{
  "config": {
    "root_path": "/path/to/markdown",
    "classification": "internal"
  }
}

Writes one JSON envelope per markdown file to stdout (JSONL).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Works when executed from this folder in the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-sdk"))
from colibri_plugin_sdk import build_envelope  # noqa: E402


def load_request() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {"config": {}}
    return json.loads(raw)


def discover_markdown(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def main() -> int:
    req = load_request()
    cfg = req.get("config", {})
    cursor = req.get("cursor") or {}
    root_path = cfg.get("root_path")
    classification = cfg.get("classification", "internal")

    if not root_path:
        print("Missing config.root_path", file=sys.stderr)
        return 2

    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        print(f"root_path does not exist: {root}", file=sys.stderr)
        return 2

    last_scan_at = cursor.get("last_scan_at")
    last_scan_dt = None
    if isinstance(last_scan_at, str):
        try:
            last_scan_dt = datetime.fromisoformat(last_scan_at.replace("Z", "+00:00"))
        except ValueError:
            last_scan_dt = None

    for md_file in discover_markdown(root):
        if last_scan_dt is not None:
            mtime = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc)
            if mtime <= last_scan_dt:
                continue
        rel = md_file.relative_to(root).as_posix()
        markdown = md_file.read_text(encoding="utf-8", errors="replace")
        title = md_file.stem.replace("_", " ").replace("-", " ").strip() or rel
        envelope = build_envelope(
            plugin_id="filesystem_markdown",
            connector_instance=str(root),
            external_id=rel,
            doc_id=f"filesystem_markdown:{rel}",
            title=title,
            markdown=markdown,
            doc_type="note",
            classification=classification,
            uri=str(md_file),
        )
        print(json.dumps(envelope, ensure_ascii=False))

    # Emit cursor control line as final checkpoint for next incremental sync.
    print(
        json.dumps(
            {
                "type": "cursor",
                "cursor": {"last_scan_at": datetime.now(timezone.utc).isoformat()},
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
