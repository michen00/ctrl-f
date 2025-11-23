"""Unit tests for field extraction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langextract.data import AnnotatedDocument, Extraction
from pydantic import BaseModel

from ctrlf.app.extract import (
    MIN_SNIPPET_LENGTH,
    _extract_snippet,
    run_extraction,
)
from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.models import ExtractionResult


@patch("ctrlf.app.extract.extract")
class TestRunExtraction:
    """Test full extraction workflow."""

    def test_run_extraction_creates_field_results(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test that extraction creates FieldResult for each schema field."""
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

        # Verify extract was called with Ollama parameters
        mock_extract.assert_called()
        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs.get("model_id") == "gemma2:2b"
        assert call_kwargs.get("model_url") == "http://localhost:11434"
        assert call_kwargs.get("use_schema_constraints") is False
        assert call_kwargs.get("fence_output") is False

    def test_run_extraction_handles_errors_gracefully(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test that extraction continues on individual field/document errors."""
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
