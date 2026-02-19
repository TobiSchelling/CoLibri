#!/usr/bin/env python3
"""Zephyr Scale ingestion plugin (wrapper around zephyr-export).

Strategy:
- Call `zephyr-export export ... --no-interactive --output <tmp>`
- Read generated Markdown files and emit CoLibri document envelopes (JSONL)
- Track known test case keys in cursor to emit tombstones for removed test cases
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

# Works when executed from this folder in the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-sdk"))
from colibri_plugin_sdk import build_envelope  # noqa: E402


PLUGIN_ID = "zephyr_scale"
DEFAULT_API_BASE_URL = "https://api.zephyrscale.smartbear.com/v2/"


@dataclass(frozen=True)
class Cursor:
    scope: str
    known_keys: list[str]
    last_scan_at: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_request() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {"config": {}, "cursor": None}
    return json.loads(raw)


def stable_scope(project_key: str, folder: str | None) -> str:
    return f"{project_key}::{folder or ''}"


def which(tool: str) -> str | None:
    p = subprocess.run(
        ["which", tool], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tool if p.returncode == 0 else None


def run_cmd(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_secs: int = 3600,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout_secs,
    )


_FRONTMATTER_RE = re.compile(r"^---\s*$")
_KEY_RE = re.compile(r"^key:\s*(.+)\s*$")
_NAME_RE = re.compile(r"^name:\s*(.+)\s*$")
_LAST_MOD_RE = re.compile(r"^last_modified_on:\s*(.+)\s*$")


def _strip_yaml_scalar(value: str) -> str:
    v = value.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1].strip()
    return v


def parse_frontmatter_fields(markdown: str) -> tuple[str | None, str | None, str | None]:
    """Return (key, name, last_modified_on) from YAML frontmatter if present."""
    lines = markdown.splitlines()
    if not lines or not _FRONTMATTER_RE.match(lines[0]):
        return None, None, None

    key = None
    name = None
    last_modified = None

    # Scan until second '---'
    for line in lines[1:400]:
        if _FRONTMATTER_RE.match(line):
            break
        if key is None:
            m = _KEY_RE.match(line)
            if m:
                key = _strip_yaml_scalar(m.group(1))
                continue
        if name is None:
            m = _NAME_RE.match(line)
            if m:
                name = _strip_yaml_scalar(m.group(1))
                continue
        if last_modified is None:
            m = _LAST_MOD_RE.match(line)
            if m:
                last_modified = _strip_yaml_scalar(m.group(1))
                continue

    return key, name, last_modified


def parse_test_case_key_from_filename(path: Path) -> str | None:
    # exporter uses: "{KEY}_{sanitized_name}.md"
    stem = path.stem
    if "_" not in stem:
        return None
    return stem.split("_", 1)[0].strip() or None


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_rfc3339(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    # Minimal normalization for the common "...Z" form.
    if v.endswith("Z") and "+" not in v and "-" in v[:10]:
        return v
    try:
        parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def export_with_zephyr_export(
    *,
    zephyr_export_cmd: str,
    project_key: str,
    folder: str | None,
    output_dir: Path,
    api_base_url: str,
    token_env: str,
    token: str | None,
    include_steps: bool,
    include_custom_fields: bool,
) -> None:
    cmd = zephyr_export_cmd
    if Path(cmd).is_absolute():
        if not Path(cmd).exists():
            raise RuntimeError(f"zephyr_export_cmd not found: {cmd}")
    else:
        if which(cmd) is None:
            raise RuntimeError(
                f"zephyr-export not found on PATH (set zephyr_export_cmd). Missing: {cmd}"
            )

    args = [
        cmd,
        "export",
        "--project",
        project_key,
        "--no-interactive",
        "--output",
        str(output_dir),
    ]
    if folder:
        args.extend(["--folder", folder])
    if not include_steps:
        args.append("--no-steps")
    if not include_custom_fields:
        args.append("--no-custom-fields")

    env = os.environ.copy()
    env["ZEPHYR_API_BASE_URL"] = api_base_url
    if token:
        env[token_env] = token

    if token_env not in env or not env.get(token_env):
        raise RuntimeError(
            f"Missing Zephyr API token. Set env var {token_env} or pass config.token."
        )

    p = run_cmd(args, env=env, timeout_secs=7200)
    if p.returncode != 0:
        raise RuntimeError(
            "zephyr-export failed.\n"
            f"cmd={' '.join(args)}\n"
            f"stdout={p.stdout.strip()}\n"
            f"stderr={p.stderr.strip()}"
        )


def main() -> None:
    request = load_request()
    config = request.get("config") or {}
    prev_cursor = request.get("cursor")

    project_key = str(config.get("project_key") or "").strip()
    if not project_key:
        print("config.project_key is required", file=sys.stderr)
        raise SystemExit(2)

    folder = config.get("folder")
    folder = str(folder).strip() if folder is not None and str(folder).strip() else None
    classification = str(config.get("classification") or "internal")
    doc_type = str(config.get("doc_type") or "test_case")
    api_base_url = str(config.get("api_base_url") or DEFAULT_API_BASE_URL).strip()
    token_env = str(config.get("token_env") or "ZEPHYR_API_TOKEN").strip()
    token = config.get("token")
    token = str(token).strip() if token is not None and str(token).strip() else None
    zephyr_export_cmd = str(config.get("zephyr_export_cmd") or "zephyr-export").strip()
    include_steps = bool(config.get("include_steps", True))
    include_custom_fields = bool(config.get("include_custom_fields", True))
    mode = str(config.get("mode") or "incremental")
    emit_tombstones = bool(config.get("emit_tombstones", True))

    connector_instance = config.get("connector_instance")
    connector_instance = (
        str(connector_instance).strip()
        if connector_instance is not None and str(connector_instance).strip()
        else f"zephyrscale:{project_key}"
    )

    scope = stable_scope(project_key, folder)
    previous = None
    if isinstance(prev_cursor, dict) and prev_cursor.get("scope") == scope:
        known = prev_cursor.get("known_keys")
        if isinstance(known, list) and all(isinstance(x, str) for x in known):
            previous = Cursor(
                scope=scope,
                known_keys=[str(x) for x in known],
                last_scan_at=str(prev_cursor.get("last_scan_at") or ""),
            )

    with tempfile.TemporaryDirectory(prefix="colibri-zephyr-export-") as tmp:
        tmp_dir = Path(tmp)
        export_with_zephyr_export(
            zephyr_export_cmd=zephyr_export_cmd,
            project_key=project_key,
            folder=folder,
            output_dir=tmp_dir,
            api_base_url=api_base_url,
            token_env=token_env,
            token=token,
            include_steps=include_steps,
            include_custom_fields=include_custom_fields,
        )

        project_dir = tmp_dir / project_key
        if not project_dir.exists():
            raise RuntimeError(f"zephyr-export did not produce project folder: {project_dir}")

        md_files = sorted(
            p for p in project_dir.rglob("*.md") if p.is_file() and not p.name.startswith(".")
        )

        current_keys: list[str] = []
        envelopes: list[dict] = []

        for path in md_files:
            key_from_name = parse_test_case_key_from_filename(path)
            markdown = path.read_text(encoding="utf-8", errors="replace")
            key_from_fm, name_from_fm, last_modified = parse_frontmatter_fields(markdown)

            key = key_from_fm or key_from_name
            if not key:
                # Skip unknown / index markdown files (if present)
                continue

            current_keys.append(key)

            title = name_from_fm or key
            updated_at = as_rfc3339(last_modified) or now_utc_iso()

            doc_id = f"{PLUGIN_ID}:{project_key}:{key}"
            external_id = key
            uri = None

            tags = ["zephyr", project_key]
            if folder:
                tags.append("folder_scoped")

            envelopes.append(
                build_envelope(
                    plugin_id=PLUGIN_ID,
                    connector_instance=connector_instance,
                    external_id=external_id,
                    doc_id=doc_id,
                    title=title,
                    markdown=markdown,
                    doc_type=doc_type,
                    classification=classification,
                    uri=uri,
                    tags=tags,
                    deleted=False,
                    source_updated_at=updated_at,
                )
            )

        current_key_set = set(current_keys)

        # Tombstones only make sense for full exports (no folder filter).
        tombstones_allowed = (
            folder is None and mode == "incremental" and emit_tombstones and previous is not None
        )
        if tombstones_allowed:
            removed = sorted(set(previous.known_keys) - current_key_set)
            for key in removed:
                doc_id = f"{PLUGIN_ID}:{project_key}:{key}"
                envelopes.append(
                    build_envelope(
                        plugin_id=PLUGIN_ID,
                        connector_instance=connector_instance,
                        external_id=key,
                        doc_id=doc_id,
                        title=key,
                        markdown="",
                        doc_type=doc_type,
                        classification=classification,
                        tags=["zephyr", project_key, "tombstone"],
                        deleted=True,
                        source_updated_at=now_utc_iso(),
                    )
                )

        for env in envelopes:
            sys.stdout.write(json.dumps(env, ensure_ascii=False) + "\n")

        next_cursor = {
            "scope": scope,
            "known_keys": sorted(current_key_set),
            "last_scan_at": now_utc_iso(),
            "summary": {
                "project_key": project_key,
                "folder": folder,
                "case_count": len(current_key_set),
                "tombstones_emitted": int(
                    tombstones_allowed and len(envelopes) > len(current_key_set)
                ),
                "run_hash": sha256_hex(scope)[:12],
            },
        }
        sys.stdout.write(json.dumps({"type": "cursor", "cursor": next_cursor}) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"zephyr_scale plugin failed: {exc}", file=sys.stderr)
        raise

