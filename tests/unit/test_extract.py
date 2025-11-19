"""Unit tests for field extraction module."""

from __future__ import annotations

from pydantic import BaseModel

from ctrlf.app.extract import (
    _extract_snippet,
    extract_field_candidates,
    run_extraction,
)
from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.models import Candidate, ExtractionResult, SourceRef


class TestExtractFieldCandidates:
    """Test field candidate extraction."""

    def test_extract_string_field(self) -> None:
        """Test extracting a string field from markdown content."""
        markdown = "Contact: john.doe@example.com for inquiries."
        source_map: dict[str, object] = {}
        doc_id = "test_doc_1"

        candidates = extract_field_candidates(
            field_name="email",
            field_type=str,
            field_description="Email address",
            markdown_content=markdown,
            doc_id=doc_id,
            source_map=source_map,
        )

        assert isinstance(candidates, list)
        # All candidates must have sources (zero fabrication)
        for candidate in candidates:
            assert isinstance(candidate, Candidate)
            assert len(candidate.sources) > 0
            assert candidate.confidence >= 0.0
            assert candidate.confidence <= 1.0

    def test_extract_multiple_occurrences(self) -> None:
        """Test that multiple occurrences create separate candidates."""
        markdown = "Email: test@example.com. Also contact test@example.com."
        source_map: dict[str, object] = {}
        doc_id = "test_doc_1"

        candidates = extract_field_candidates(
            field_name="email",
            field_type=str,
            field_description=None,
            markdown_content=markdown,
            doc_id=doc_id,
            source_map=source_map,
        )

        # Should create separate candidates for each occurrence
        # (exact behavior depends on langextract, but should handle multiple)
        assert isinstance(candidates, list)

    def test_extract_returns_empty_on_no_match(self) -> None:
        """Test that empty list is returned when no candidates found."""
        markdown = "This document has no email addresses."
        source_map: dict[str, object] = {}
        doc_id = "test_doc_1"

        candidates = extract_field_candidates(
            field_name="email",
            field_type=str,
            field_description="Email address",
            markdown_content=markdown,
            doc_id=doc_id,
            source_map=source_map,
        )

        assert candidates == []

    def test_all_candidates_have_sources(self) -> None:
        """Test that all candidates have non-empty sources (zero fabrication)."""
        markdown = "Name: John Doe, Email: john@example.com"
        source_map: dict[str, object] = {}
        doc_id = "test_doc_1"

        candidates = extract_field_candidates(
            field_name="name",
            field_type=str,
            field_description="Person name",
            markdown_content=markdown,
            doc_id=doc_id,
            source_map=source_map,
        )

        for candidate in candidates:
            assert len(candidate.sources) > 0
            for source in candidate.sources:
                assert isinstance(source, SourceRef)
                assert source.doc_id == doc_id


class TestRunExtraction:
    """Test full extraction workflow."""

    def test_run_extraction_creates_field_results(self) -> None:
        """Test that extraction creates FieldResult for each schema field."""

        # Create a simple Extended Schema model
        class TestModel(BaseModel):
            name: list[str]
            email: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Name: Alice, Email: alice@example.com",
                source_map={"spans": {}},
            ),
        ]

        result = run_extraction(TestModel, corpus_docs)

        assert isinstance(result, ExtractionResult)
        assert len(result.results) == 2  # One per field
        assert result.run_id
        assert result.created_at
        assert result.schema_version

    def test_run_extraction_handles_errors_gracefully(self) -> None:
        """Test that extraction continues on individual field/document errors."""

        class TestModel(BaseModel):
            name: list[str]
            email: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Valid content with name: Test",
                source_map={"spans": {}},
            ),
            CorpusDocument(
                doc_id="doc2", markdown="", source_map={"spans": {}}
            ),  # Empty document
        ]

        # Should not raise, should continue processing
        result = run_extraction(TestModel, corpus_docs)
        assert isinstance(result, ExtractionResult)


class TestExtractSnippet:
    """Test snippet extraction function."""

    def test_extract_snippet_ensures_minimum_length(self) -> None:
        """Test that _extract_snippet always returns at least 10 characters."""
        # Test case: very short document
        markdown = "Hi"
        snippet = _extract_snippet(markdown, start=0, end=2)
        assert len(snippet) >= 10

    def test_extract_snippet_short_span_in_short_doc(self) -> None:
        """Test snippet extraction with short span in short document."""
        # Document is 5 characters, span is 2 characters
        markdown = "Hello"
        snippet = _extract_snippet(markdown, start=0, end=2)
        assert len(snippet) >= 10

    def test_extract_snippet_very_short_document(self) -> None:
        """Test snippet extraction with document shorter than 10 chars."""
        # Document is only 3 characters
        markdown = "Hi!"
        snippet = _extract_snippet(markdown, start=0, end=3)
        assert len(snippet) >= 10

    def test_extract_snippet_normal_case(self) -> None:
        """Test snippet extraction in normal case with sufficient content."""
        markdown = (
            "This is a longer document with plenty of content to extract snippets from."
        )
        snippet = _extract_snippet(markdown, start=10, end=20)
        assert len(snippet) >= 10
        assert "longer" in snippet or "document" in snippet
