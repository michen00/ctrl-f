"""Unit tests for field extraction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from ctrlf.app.extract import (
    MIN_SNIPPET_LENGTH,
    _extract_snippet,
    run_extraction,
)
from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.models import ExtractionResult


@patch("ctrlf.app.structured_extract._call_structured_extraction_api")
class TestRunExtraction:
    """Test full extraction workflow."""

    def test_run_extraction_creates_field_results(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test that extraction creates FieldResult for each schema field."""
        # Mock PydanticAI API call to return structured data matching the schema
        # The API returns a dict matching the Extended Schema model
        mock_api_call.return_value = {
            "name": ["Alice"],
            "email": ["alice@example.com"],
        }

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

        # Verify API was called with correct parameters
        mock_api_call.assert_called_once()
        call_args = mock_api_call.call_args
        assert call_args[0][0] == "Name: Alice, Email: alice@example.com"  # text
        assert call_args[0][1] == TestModel  # schema_model
        assert call_args.kwargs.get("provider") == "ollama"  # default provider

    def test_run_extraction_handles_errors_gracefully(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test that extraction continues on individual field/document errors."""
        # Mock API call to fail for first doc, succeed for second (empty)
        mock_api_call.side_effect = [
            RuntimeError("Extraction failed"),
            {"name": [], "email": []},  # Empty extraction for second doc
        ]

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
        assert mock_api_call.call_count == 2  # Called for both documents


class TestExtractSnippet:
    """Test snippet extraction function."""

    def test_extract_snippet_ensures_minimum_length(self) -> None:
        """Test that _extract_snippet always returns the minimum length."""
        # Test case: very short document
        markdown = "Hi"
        snippet = _extract_snippet(markdown, start=0, end=2)
        assert len(snippet) >= MIN_SNIPPET_LENGTH

    def test_extract_snippet_short_span_in_short_doc(self) -> None:
        """Test snippet extraction with short span in short document."""
        # Document is 5 characters, span is 2 characters
        markdown = "Hello"
        snippet = _extract_snippet(markdown, start=0, end=2)
        assert len(snippet) >= MIN_SNIPPET_LENGTH

    def test_extract_snippet_very_short_document(self) -> None:
        """Test snippet extraction with document shorter than the minimum length."""
        # Document is only 3 characters
        markdown = "Hi!"
        snippet = _extract_snippet(markdown, start=0, end=3)
        assert len(snippet) >= MIN_SNIPPET_LENGTH

    def test_extract_snippet_normal_case(self) -> None:
        """Test snippet extraction in normal case with sufficient content."""
        markdown = (
            "This is a longer document with plenty of content to extract snippets from."
        )
        snippet = _extract_snippet(markdown, start=10, end=20)
        assert len(snippet) >= MIN_SNIPPET_LENGTH
        assert "longer" in snippet or "document" in snippet
