"""Unit tests for field extraction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langextract.data import AnnotatedDocument, Extraction
from pydantic import BaseModel

from ctrlf.app.extract import (
    _extract_snippet,
    extract_field_candidates,
    run_extraction,
)
from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.models import (
    Candidate,
    ExtractionResult,
    PrePromptInteraction,
    SourceRef,
)


class TestExtractFieldCandidates:
    """Test field candidate extraction."""

    @patch("ctrlf.app.extract.extract")
    def test_extract_string_field(self, mock_extract: MagicMock) -> None:
        """Test extracting a string field from markdown content."""
        # Mock extract return value
        # Use a Mock object instead of Extraction to allow arbitrary attributes
        extraction = MagicMock(spec=Extraction)
        extraction.extraction_class = "email"
        extraction.extraction_text = "john.doe@example.com"
        extraction.char_start = 9
        extraction.char_end = 29
        extraction.confidence = 0.95

        mock_doc = AnnotatedDocument(
            text="Contact: john.doe@example.com for inquiries.",
            extractions=[extraction],
        )
        mock_extract.return_value = mock_doc

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
            assert candidate.value == "john.doe@example.com"

    @patch("ctrlf.app.extract.extract")
    def test_extract_multiple_occurrences(self, mock_extract: MagicMock) -> None:
        """Test that multiple occurrences create separate candidates."""
        # Mock extract return value
        e1 = MagicMock(spec=Extraction)
        e1.extraction_class = "email"
        e1.extraction_text = "test@example.com"
        e1.char_start = 7
        e1.char_end = 23

        e2 = MagicMock(spec=Extraction)
        e2.extraction_class = "email"
        e2.extraction_text = "test@example.com"
        e2.char_start = 38
        e2.char_end = 54

        mock_doc = AnnotatedDocument(
            text="Email: test@example.com. Also contact test@example.com.",
            extractions=[e1, e2],
        )
        mock_extract.return_value = mock_doc

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
        assert isinstance(candidates, list)
        assert len(candidates) == 2
        assert candidates[0].value == "test@example.com"
        assert candidates[1].value == "test@example.com"

    @patch("ctrlf.app.extract.extract")
    def test_extract_returns_empty_on_no_match(self, mock_extract: MagicMock) -> None:
        """Test that empty list is returned when no candidates found."""
        # Mock extract return value
        mock_doc = AnnotatedDocument(
            text="This document has no email addresses.", extractions=[]
        )
        mock_extract.return_value = mock_doc

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

    @patch("ctrlf.app.extract.extract")
    def test_all_candidates_have_sources(self, mock_extract: MagicMock) -> None:
        """Test that all candidates have non-empty sources (zero fabrication)."""
        # Mock extract return value
        e1 = MagicMock(spec=Extraction)
        e1.extraction_class = "name"
        e1.extraction_text = "John Doe"
        e1.char_start = 6
        e1.char_end = 14

        mock_doc = AnnotatedDocument(
            text="Name: John Doe, Email: john@example.com",
            extractions=[e1],
        )
        mock_extract.return_value = mock_doc

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


@patch("ctrlf.app.extract.generate_synthetic_example")
@patch("ctrlf.app.extract.generate_example_extractions")
@patch("ctrlf.app.extract.extract")
class TestRunExtraction:
    """Test full extraction workflow."""

    def test_run_extraction_creates_field_results(
        self,
        mock_extract: MagicMock,
        mock_gen_extractions: MagicMock,
        mock_gen_example: MagicMock,
    ) -> None:
        """Test that extraction creates FieldResult for each schema field."""
        mock_gen_example.return_value = (
            "Example text",
            PrePromptInteraction(
                step_name="test",
                prompt="test",
                completion="test",
                model="test",
            ),
        )
        mock_gen_extractions.return_value = (
            [],
            PrePromptInteraction(
                step_name="test",
                prompt="test",
                completion="test",
                model="test",
            ),
        )

        # Mock extraction results
        # We need to return a document with extractions for both fields
        e1 = MagicMock(spec=Extraction)
        e1.extraction_class = "name"
        e1.extraction_text = "Alice"
        e1.char_start = 6
        e1.char_end = 11

        e2 = MagicMock(spec=Extraction)
        e2.extraction_class = "email"
        e2.extraction_text = "alice@example.com"
        e2.char_start = 20
        e2.char_end = 37

        mock_doc = AnnotatedDocument(
            text="Name: Alice, Email: alice@example.com",
            extractions=[e1, e2],
        )
        mock_extract.return_value = mock_doc

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

        result, _instrumentation = run_extraction(TestModel, corpus_docs)

        assert isinstance(result, ExtractionResult)
        assert len(result.results) == 2  # One per field
        assert result.run_id
        assert result.created_at
        assert result.schema_version

        # Verify our mocks were called
        mock_gen_example.assert_called_once()
        mock_gen_extractions.assert_called_once()
        mock_extract.assert_called()

    def test_run_extraction_handles_errors_gracefully(
        self,
        mock_extract: MagicMock,
        mock_gen_extractions: MagicMock,
        mock_gen_example: MagicMock,
    ) -> None:
        """Test that extraction continues on individual field/document errors."""
        mock_gen_example.return_value = (
            "Example text",
            PrePromptInteraction(
                step_name="test",
                prompt="test",
                completion="test",
                model="test",
            ),
        )
        mock_gen_extractions.return_value = (
            [],
            PrePromptInteraction(
                step_name="test",
                prompt="test",
                completion="test",
                model="test",
            ),
        )

        # Mock extraction to fail for first doc, succeed for second (empty)
        mock_doc_success = AnnotatedDocument(text="", extractions=[])

        # Side effect: first call raises Exception, second returns empty doc
        mock_extract.side_effect = [Exception("Extraction failed"), mock_doc_success]

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
        result, _instrumentation = run_extraction(TestModel, corpus_docs)
        assert isinstance(result, ExtractionResult)
        assert len(result.results) == 2  # Results for fields exist even if empty


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
