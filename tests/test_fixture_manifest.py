from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fixture_manifest import (
    ManifestValidationError,
    load_fixture_manifest,
    validate_fixture_manifest,
)

FIXTURES = Path("evals/fixtures")
LABELS = Path("evals/labels.jsonl")
MANIFEST = Path("evals/fixture_manifest.json")


def test_fixture_manifest_matches_labels_and_fixture_files() -> None:
    manifest = load_fixture_manifest(MANIFEST)
    labels = {
        json.loads(line)["file"] for line in LABELS.read_text().splitlines() if line
    }
    fixtures = {
        path.name
        for path in FIXTURES.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }

    validate_fixture_manifest(manifest, label_files=labels, fixture_files=fixtures)


def test_manifest_records_verified_and_unverified_provenance() -> None:
    manifest = load_fixture_manifest(MANIFEST)

    assert manifest["sources"]["express_expense_srd"]["license"] == "MIT"
    assert manifest["sources"]["express_expense_srd"]["license_status"] == "verified"
    assert manifest["fixtures"]["1000-receipt.jpg"]["source"] == "express_expense_srd"
    assert (
        manifest["sources"]["unverified_non_receipt"]["license_status"] == "unverified"
    )


def test_manifest_rejects_missing_and_orphaned_entries() -> None:
    manifest = {
        "sources": {"source": {"license": "MIT", "url": "https://example.com"}},
        "fixtures": {"present.jpg": {"source": "source"}},
    }

    with pytest.raises(ManifestValidationError, match="missing labels: missing.jpg"):
        validate_fixture_manifest(
            manifest,
            label_files={"present.jpg", "missing.jpg"},
            fixture_files={"present.jpg", "orphan.jpg"},
        )


def test_corrected_merchants_match_printed_fixture_names() -> None:
    labels = {
        json.loads(line)["file"]: json.loads(line)
        for line in LABELS.read_text().splitlines()
        if line
    }

    assert labels["1015-receipt.jpg"]["merchant"] == "HAMMOCKS TRADING COMPANY"
    assert labels["1016-receipt.jpg"]["merchant"] == "Chef Wang"
    assert labels["1017-receipt.jpg"]["merchant"] == "UMIX"
