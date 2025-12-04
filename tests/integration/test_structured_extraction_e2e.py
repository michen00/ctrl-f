"""Integration tests for structured extraction end-to-end workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.structured_extract import (
    JSONLLine,
    run_structured_extraction,
    visualize_extractions,
    write_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path


@patch("ctrlf.app.structured_extract._call_structured_extraction_api")
class TestStructuredExtractionE2E:
    """Test complete structured extraction workflow (T024)."""

    def test_ollama_extraction_workflow(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test complete workflow with Ollama provider."""
        # Mock API response matching Extended Schema format
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo"],
            "emotion": ["longing", "love"],
            "relationship": ["sister"],
        }

        # Create Extended Schema model
        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]
            relationship: list[str]

        # Create test corpus
        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed longingly at the stars. Romeo loved her.",
                source_map={"spans": {}},
            ),
        ]

        # Run structured extraction
        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
            model="llama3",
            fuzzy_threshold=80,
        )

        # Verify results
        assert len(jsonl_lines) == 1
        assert isinstance(jsonl_lines[0], JSONLLine)
        assert jsonl_lines[0].document_id == "doc1"
        assert len(jsonl_lines[0].extractions) > 0

        # Verify API was called
        mock_api_call.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"})
    @patch("ctrlf.app.structured_extract.Agent")
    def test_openai_extraction_workflow(
        self,
        mock_agent_class: MagicMock,
    ) -> None:
        """Test complete workflow with OpenAI provider."""
        # Mock PydanticAI Agent
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {
            "character": ["Lady Juliet"],
            "emotion": ["longing"],
        }
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed longingly",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="openai",
            model="gpt-4o",
        )

        assert len(jsonl_lines) == 1
        assert jsonl_lines[0].document_id == "doc1"

    @patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test_google_key_12345678901234567890"},
    )
    @patch("ctrlf.app.structured_extract.Agent")
    def test_gemini_extraction_workflow(
        self,
        mock_agent_class: MagicMock,
    ) -> None:
        """Test complete workflow with Gemini provider."""
        # Mock PydanticAI Agent
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {
            "character": ["Lady Juliet"],
        }
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="gemini",
            model="gemini-2.0-flash-exp",
        )

        assert len(jsonl_lines) == 1
        assert jsonl_lines[0].document_id == "doc1"

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_multi_provider_workflow(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test workflow with multiple providers (T034)."""
        # Mock API responses
        mock_api_call.side_effect = [
            {"character": ["Lady Juliet"]},  # Ollama
            {"character": ["Romeo"]},  # OpenAI
            {"character": ["Mercutio"]},  # Gemini
        ]

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet",
                source_map={"spans": {}},
            ),
            CorpusDocument(
                doc_id="doc2",
                markdown="Romeo",
                source_map={"spans": {}},
            ),
            CorpusDocument(
                doc_id="doc3",
                markdown="Mercutio",
                source_map={"spans": {}},
            ),
        ]

        # Test with Ollama (default)
        jsonl_lines_ollama = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )
        assert len(jsonl_lines_ollama) == 3

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_jsonl_write_and_read(
        self,
        mock_api_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test JSONL file write and read workflow."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
        }

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        # Write JSONL file
        output_path = tmp_path / "test_output.jsonl"
        write_jsonl(jsonl_lines, output_path)

        # Verify file exists and is readable
        assert output_path.exists()
        with output_path.open() as f:
            lines = f.readlines()
            assert len(lines) == 1

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    @patch("ctrlf.app.structured_extract.langextract")
    def test_visualization_workflow(
        self,
        mock_langextract: MagicMock,
        mock_api_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test visualization workflow (T044)."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
        }

        # Mock langextract.visualize()
        mock_langextract.visualize.return_value = (
            "<html><body>Visualization</body></html>"
        )

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        # Write JSONL file
        jsonl_path = tmp_path / "test.jsonl"
        write_jsonl(jsonl_lines, jsonl_path)

        # Visualize
        html_path = tmp_path / "visualization.html"
        html_content = visualize_extractions(
            jsonl_path,
            output_html_path=html_path,
        )

        assert html_content == "<html><body>Visualization</body></html>"
        assert html_path.exists()

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_error_handling_continues_processing(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test that errors in one document don't stop processing others."""
        # Mock API to fail for first doc, succeed for second
        mock_api_call.side_effect = [
            RuntimeError("API error"),
            {"character": ["Romeo"]},
        ]

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet",
                source_map={"spans": {}},
            ),
            CorpusDocument(
                doc_id="doc2",
                markdown="Romeo",
                source_map={"spans": {}},
            ),
        ]

        # Should not raise, should continue processing
        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        # Should have results for both documents (one may be empty)
        assert len(jsonl_lines) == 2
        assert jsonl_lines[0].document_id == "doc1"
        assert jsonl_lines[1].document_id == "doc2"
