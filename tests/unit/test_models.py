"""Unit tests for Pydantic data models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctrlf.app.models import (
    Candidate,
    ExtractionResult,
    FieldResult,
    SourceRef,
)


class TestSourceRef:
    """Tests for SourceRef model."""

    def test_valid_source_ref(self) -> None:
        """Test creating a valid SourceRef."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 3, line 120",
            snippet="This is a context snippet around the extracted value.",
            metadata={"mtime": "2024-01-01T00:00:00Z"},
        )
        assert source.doc_id == "doc1"
        assert source.path == "/path/to/file.pdf"
        assert source.location == "page 3, line 120"
        assert len(source.snippet) >= 10

    def test_source_ref_empty_doc_id_fails(self) -> None:
        """Test that empty doc_id fails validation."""
        with pytest.raises(ValueError, match=r".*"):
            SourceRef(
                doc_id="",
                path="/path/to/file.pdf",
                location="page 1",
                snippet="Context snippet",
            )

    def test_source_ref_short_snippet_fails(self) -> None:
        """Test that snippet too short fails validation."""
        with pytest.raises(ValueError, match=r".*"):
            SourceRef(
                doc_id="doc1",
                path="/path/to/file.pdf",
                location="page 1",
                snippet="short",
            )


class TestCandidate:
    """Tests for Candidate model."""

    def test_valid_candidate(self) -> None:
        """Test creating a valid Candidate."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        candidate = Candidate(
            value="test@example.com",
            normalized="test@example.com",
            confidence=0.95,
            sources=[source],
        )
        assert candidate.value == "test@example.com"
        assert candidate.confidence == 0.95
        assert len(candidate.sources) == 1

    def test_candidate_empty_sources_fails(self) -> None:
        """Test that empty sources list fails validation (zero fabrication)."""
        with pytest.raises(ValueError, match=r".*"):
            Candidate(
                value="test",
                confidence=0.5,
                sources=[],
            )

    def test_candidate_confidence_out_of_range_fails(self) -> None:
        """Test that confidence outside [0.0, 1.0] fails validation."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        with pytest.raises(ValueError, match=r".*"):
            Candidate(
                value="test",
                confidence=1.5,
                sources=[source],
            )


class TestFieldResult:
    """Tests for FieldResult model."""

    def test_valid_field_result(self) -> None:
        """Test creating a valid FieldResult."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        candidate = Candidate(
            value="test@example.com",
            confidence=0.95,
            sources=[source],
        )
        field_result = FieldResult(
            field_name="email",
            candidates=[candidate],
            consensus=candidate,
        )
        assert field_result.field_name == "email"
        assert len(field_result.candidates) == 1
        assert field_result.consensus == candidate

    def test_field_result_consensus_not_in_candidates_fails(self) -> None:
        """Test that consensus must be one of the candidates."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        candidate1 = Candidate(
            value="test1@example.com",
            confidence=0.95,
            sources=[source],
        )
        candidate2 = Candidate(
            value="test2@example.com",
            confidence=0.90,
            sources=[source],
        )
        with pytest.raises(ValueError, match=r".*"):
            FieldResult(
                field_name="email",
                candidates=[candidate1],
                consensus=candidate2,  # Not in candidates list
            )


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_valid_extraction_result(self) -> None:
        """Test creating a valid ExtractionResult."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        candidate = Candidate(
            value="test@example.com",
            confidence=0.95,
            sources=[source],
        )
        field_result = FieldResult(
            field_name="email",
            candidates=[candidate],
        )
        timestamp = datetime.now(UTC).isoformat()
        extraction_result = ExtractionResult(
            results=[field_result],
            schema_version="v1.0",
            run_id="run123",
            created_at=timestamp,
        )
        assert len(extraction_result.results) == 1
        assert extraction_result.run_id == "run123"

    def test_extraction_result_invalid_timestamp_fails(self) -> None:
        """Test that invalid ISO timestamp fails validation."""
        source = SourceRef(
            doc_id="doc1",
            path="/path/to/file.pdf",
            location="page 1",
            snippet="Context snippet for the candidate value.",
        )
        candidate = Candidate(
            value="test@example.com",
            confidence=0.95,
            sources=[source],
        )
        field_result = FieldResult(
            field_name="email",
            candidates=[candidate],
        )
        with pytest.raises(ValueError, match=r".*"):
            ExtractionResult(
                results=[field_result],
                schema_version="v1.0",
                run_id="run123",
                created_at="invalid-timestamp",
            )
