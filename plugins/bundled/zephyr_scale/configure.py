#!/usr/bin/env python3
"""Interactive configuration for the Zephyr Scale plugin.

Protocol: receives a JSON config file path as argv[1].
Reads current config, runs interactive prompts, writes updated config back.

Exit codes:
  0  success (config file updated)
  1  user cancelled (config file unchanged)
  2+ error
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def save_config_file(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_interactive(current_config: dict) -> dict | None:
    """Run interactive prompts. Returns updated config or None if cancelled."""
    print("\n  Zephyr Scale — Plugin Configuration")
    print("  " + "=" * 38 + "\n")

    # Token env var
    token_env = current_config.get("token_env", "ZEPHYR_API_TOKEN")
    token = os.environ.get(token_env, "")
    if token:
        print(f"  API Token ({token_env}): ***set***")
    else:
        print(f"  WARNING: {token_env} is not set in environment.")
        print(f"  Set it before running sync: export {token_env}=...")
    print()

    # Project key
    current_project = current_config.get("project_key", "")
    prompt = f"  Project key [{current_project}]: " if current_project else "  Project key: "
    try:
        project_key = input(prompt).strip() or current_project
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return None

    if not project_key:
        print("  Error: project_key is required.")
        return None

    # Folder (optional)
    current_folder = current_config.get("folder", "")
    prompt = f"  Folder (optional) [{current_folder}]: " if current_folder else "  Folder (optional): "
    try:
        folder = input(prompt).strip() or current_folder
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return None

    # Build updated config
    config = dict(current_config)
    config["project_key"] = project_key
    config["token_env"] = token_env
    if folder:
        config["folder"] = folder
    elif "folder" in config:
        del config["folder"]

    # Set defaults for fields not yet configured
    config.setdefault("classification", "internal")
    config.setdefault("doc_type", "test_case")

    print(f"\n  Configuration:")
    print(f"    project_key: {config['project_key']}")
    print(f"    folder: {config.get('folder', '(all)')}")
    print(f"    token_env: {config['token_env']}")
    print(f"    classification: {config['classification']}")
    print()

    return config


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: configure.py <config_file_path>", file=sys.stderr)
        raise SystemExit(2)

    config_path = Path(sys.argv[1])
    current_config = load_config_file(config_path)

    result = run_interactive(current_config)

    if result is None:
        raise SystemExit(1)

    save_config_file(config_path, result)


if __name__ == "__main__":
    main()
