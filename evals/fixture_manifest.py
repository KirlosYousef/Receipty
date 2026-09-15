from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    """Fixture manifest and evaluation dataset are inconsistent."""


def load_fixture_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_fixture_manifest(
    manifest: dict[str, Any],
    *,
    label_files: set[str],
    fixture_files: set[str],
) -> None:
    sources = manifest.get("sources")
    entries = manifest.get("fixtures")
    if not isinstance(sources, dict) or not isinstance(entries, dict):
        raise ManifestValidationError("manifest requires sources and fixtures objects")

    missing_labels = sorted(label_files - entries.keys())
    orphaned_entries = sorted(set(entries) - label_files)
    missing_files = sorted(label_files - fixture_files)
    orphaned_files = sorted(fixture_files - label_files)
    if missing_labels or orphaned_entries or missing_files or orphaned_files:
        parts = []
        if missing_labels:
            parts.append(f"missing labels: {', '.join(missing_labels)}")
        if orphaned_entries:
            parts.append(f"orphaned entries: {', '.join(orphaned_entries)}")
        if missing_files:
            parts.append(f"missing files: {', '.join(missing_files)}")
        if orphaned_files:
            parts.append(f"orphaned files: {', '.join(orphaned_files)}")
        raise ManifestValidationError("; ".join(parts))

    unknown_sources = sorted(
        filename
        for filename, entry in entries.items()
        if not isinstance(entry, dict) or entry.get("source") not in sources
    )
    if unknown_sources:
        raise ManifestValidationError(
            f"unknown source entries: {', '.join(unknown_sources)}"
        )
