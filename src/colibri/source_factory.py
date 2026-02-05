"""Nesting-aware source creation from folder profiles.

Creates ``MarkdownFolderSource`` instances while correctly excluding
nested folder profiles from their parent.
"""

from __future__ import annotations

from pathlib import Path

from colibri.config import FolderProfile
from colibri.sources import MarkdownFolderSource


def compute_nested_exclusions(
    profiles: list[FolderProfile],
) -> dict[str, tuple[str, ...]]:
    """For each profile, find other profiles whose path is a strict sub-path.

    Returns a mapping from profile path to tuple of nested paths to exclude.
    """
    resolved = [(p, str(Path(p.path).resolve())) for p in profiles]
    result: dict[str, tuple[str, ...]] = {}

    for parent_profile, parent_resolved in resolved:
        nested: list[str] = []
        for _child_profile, child_resolved in resolved:
            if child_resolved == parent_resolved:
                continue
            if child_resolved.startswith(parent_resolved + "/"):
                nested.append(child_resolved)
        if nested:
            result[parent_profile.path] = tuple(sorted(nested))

    return result


def create_source_for_profile(
    profile: FolderProfile,
    nested_map: dict[str, tuple[str, ...]],
) -> MarkdownFolderSource:
    """Create a MarkdownFolderSource for a profile, with nesting exclusions."""
    exclude = nested_map.get(profile.path, ())
    return MarkdownFolderSource(
        folder_path=profile.path,
        name=profile.display_name,
        extensions=profile.extensions,
        exclude_paths=exclude,
    )
