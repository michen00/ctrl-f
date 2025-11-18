"""Unit tests for storage module."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ctrlf.app.storage as storage_module
from ctrlf.app.models import PersistedRecord, SourceRef
from ctrlf.app.storage import export_record, save_record


class TestSaveRecord:
    """Test saving records to TinyDB."""

    def test_save_record_creates_record_id(self) -> None:
        """Test that saving a record returns a record ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            record = PersistedRecord(
                record_id="test-record-1",
                resolved={"name": ["Alice"], "email": ["alice@example.com"]},
                provenance={
                    "name": [
                        SourceRef(
                            doc_id="doc1",
                            path="doc1.txt",
                            location="page 1",
                            snippet="Name: Alice",
                        )
                    ],
                    "email": [
                        SourceRef(
                            doc_id="doc1",
                            path="doc1.txt",
                            location="page 1",
                            snippet="Email: alice@example.com",
                        )
                    ],
                },
                audit={
                    "run_id": "run1",
                    "app_version": "0.0.0",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "user": None,
                    "config": {},
                },
            )

            # Temporarily override storage path for testing
            original_get_path = storage_module.get_storage_path
            storage_module.get_storage_path = lambda: storage_path
            try:
                record_id = save_record(record)
            finally:
                storage_module.get_storage_path = original_get_path
            assert record_id == "test-record-1"

    def test_save_record_validation_failure(self) -> None:
        """Test that ValidationError is raised for invalid records."""
        # Invalid record (empty record_id) - Pydantic validates at construction time
        with pytest.raises(ValueError, match=r".*"):
            PersistedRecord(
                record_id="",
                resolved={},
                provenance={},
                audit={},
            )


class TestExportRecord:
    """Test exporting records."""

    def test_export_record_returns_dict(self) -> None:
        """Test that export returns JSON-serializable dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            record = PersistedRecord(
                record_id="test-record-1",
                resolved={"name": ["Alice"]},
                provenance={
                    "name": [
                        SourceRef(
                            doc_id="doc1",
                            path="doc1.txt",
                            location="page 1",
                            snippet="Name: Alice",
                        )
                    ]
                },
                audit={
                    "run_id": "run1",
                    "app_version": "0.0.0",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "user": None,
                    "config": {},
                },
            )

            original_get_path = storage_module.get_storage_path
            storage_module.get_storage_path = lambda: storage_path
            try:
                save_record(record)
                exported = export_record("test-record-1")
            finally:
                storage_module.get_storage_path = original_get_path

            assert isinstance(exported, dict)
            assert exported["record_id"] == "test-record-1"

    def test_export_nonexistent_record(self) -> None:
        """Test that KeyError is raised for nonexistent records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            original_get_path = storage_module.get_storage_path
            storage_module.get_storage_path = lambda: storage_path
            try:
                with pytest.raises(KeyError):
                    export_record("nonexistent")
            finally:
                storage_module.get_storage_path = original_get_path
