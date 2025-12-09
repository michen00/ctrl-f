"""Integration and E2E tests for structured extraction workflow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from ctrlf.app.aggregate import aggregate_field_results
from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import CorpusDocument
from ctrlf.app.models import Candidate, SourceRef
from ctrlf.app.schema_io import convert_json_schema_to_pydantic
from ctrlf.app.structured_extract import (
    ExtractionRecord,
    JSONLLine,
    _extraction_record_to_candidate,
    run_structured_extraction,
    visualize_extractions,
    write_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# EXECUTABLE DOCUMENTATION: COMPLETE WORKFLOWS
# These tests demonstrate complete workflows and serve as tutorials.
# ============================================================================


class TestCompleteExtractionWorkflow:
    """Executable documentation: Complete extraction workflow from schema to results.

    This demonstrates the full pipeline:
    1. Define schema (JSON Schema or Pydantic model)
    2. Process corpus documents
    3. Run structured extraction
    4. Aggregate and review results
    """

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_complete_workflow_tutorial(
        self, mock_api_call: MagicMock, tmp_path: Path
    ) -> None:
        """Tutorial: Complete structured extraction workflow.

        This example shows the complete workflow from schema definition
        through extraction to JSONL output. It demonstrates:
        - Extended Schema format (all fields are arrays)
        - Per-document processing
        - Result aggregation
        - JSONL output for visualization
        """

        # Step 1: Define Extended Schema (all fields are arrays)
        # This allows extracting multiple values per field from each document
        class CharacterModel(BaseModel):
            character: list[str]  # Can extract multiple characters per document
            emotion: list[str]  # Can extract multiple emotions per document
            relationship: list[str]  # Can extract multiple relationships per document

        # Step 2: Mock API response (in real usage, calls Ollama/OpenAI/Gemini)
        # The API returns data matching the Extended Schema structure
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo"],  # Multiple characters found
            "emotion": ["longing", "love"],  # Multiple emotions found
            "relationship": ["sister"],  # Single relationship found
        }

        # Step 3: Create corpus documents
        corpus_docs = [
            CorpusDocument(
                doc_id="doc_romeo_juliet_act1",
                markdown=(
                    "Lady Juliet gazed longingly at the stars. "
                    "Romeo loved her deeply. She was his sister's friend."
                ),
                source_map={"file_path": "romeo_juliet.txt", "spans": {}},
            ),
        ]

        # Step 4: Run structured extraction
        # This processes each document individually and extracts structured data
        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",  # or "openai", "gemini"
            model="llama3.1",  # or "gpt-4o", "gemini-2.5-flash"
            fuzzy_threshold=80,  # Minimum similarity for fuzzy matching
        )

        # Step 5: Verify results
        # Each document produces one JSONLLine with all its extractions
        assert len(jsonl_lines) == 1
        jsonl_line = jsonl_lines[0]
        assert jsonl_line.document_id == "doc_romeo_juliet_act1"

        # Each extraction includes field name, value, and location
        extraction_classes = {ext.extraction_class for ext in jsonl_line.extractions}
        assert "character" in extraction_classes
        assert "emotion" in extraction_classes

        # Step 6: Write results to JSONL file for visualization
        output_path = tmp_path / "test_output.jsonl"
        write_jsonl(jsonl_lines, output_path)
        assert output_path.exists()

        # Verify JSONL format
        with output_path.open() as f:
            line = json.loads(f.readline())
            assert "extractions" in line
            assert "text" in line
            assert "document_id" in line

        # Verify API was called correctly
        mock_api_call.assert_called_once()
        call_args = mock_api_call.call_args
        assert call_args[1]["provider"] == "ollama"
        assert call_args[1]["model"] == "llama3.1"


# ============================================================================
# INTEGRATION TESTS: COMPONENT INTERACTIONS
# These tests verify that components work together correctly.
# ============================================================================


class TestStructuredExtractExtractIntegration:
    """Integration tests: structured_extract + extract.py.

    Tests that structured extraction integrates correctly with the main
    extraction pipeline (extract.py).
    """

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_extraction_record_to_candidate_integration(
        self, mock_api_call: MagicMock
    ) -> None:
        """Test integration: ExtractionRecord -> Candidate conversion.

        Verifies that ExtractionRecord (from structured extraction) correctly
        converts to Candidate (used by aggregation pipeline).
        """
        # Mock API to return data
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
        }

        # Create ExtractionRecord (as structured extraction would)
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )

        # Convert to Candidate (as extract.py does)
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {
            "file_path": "test.txt",
            "pages": {"1": {"start": 0, "end": 100}},
        }

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        # Verify Candidate structure matches aggregation pipeline expectations
        assert candidate.value == "Lady Juliet"
        assert candidate.confidence == 0.9  # match_exact = 0.9
        assert len(candidate.sources) == 1
        assert candidate.sources[0].doc_id == "doc1"
        assert candidate.sources[0].location == "page 1"

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_run_extraction_uses_structured_extract(
        self, mock_api_call: MagicMock
    ) -> None:
        """Test integration: run_extraction uses structured extraction.

        Verifies that the main extraction pipeline (run_extraction) correctly
        uses structured extraction functions.
        """
        # Mock API response
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo"],
            "emotion": ["love"],
        }

        # Define Extended Schema
        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]

        # Create corpus
        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet and Romeo felt love.",
                source_map={"spans": {}},
            ),
        ]

        # Run extraction (uses structured extraction internally)
        extraction_result, _instrumentation = run_extraction(
            model=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
            model_name="llama3.1",
        )

        # Verify results
        assert len(extraction_result.results) == 2  # character and emotion fields
        assert extraction_result.run_id is not None

        # Verify API was called
        assert mock_api_call.call_count == 1


class TestStructuredExtractAggregateIntegration:
    """Integration tests: structured_extract + aggregate.py.

    Tests that structured extraction results integrate correctly with
    aggregation (deduplication, consensus detection).
    """

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_candidates_aggregate_correctly(self, mock_api_call: MagicMock) -> None:
        """Test integration: Candidates from structured extraction aggregate correctly.

        Verifies that Candidates created from structured extraction work
        correctly with the aggregation pipeline (deduplication, consensus).
        """
        # Mock API to return same value from multiple documents
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
        }

        # Create multiple Candidates with same value (as would come from
        # multiple documents)
        candidates: list[Candidate] = []
        for doc_id in ["doc1", "doc2", "doc3"]:
            source_ref = SourceRef(
                doc_id=doc_id,
                path="test.txt",
                location="char-range [0:11]",
                snippet="Lady Juliet gazed",
                metadata={},
            )
            candidate = Candidate(
                value="Lady Juliet",
                normalized=None,
                confidence=0.9,
                sources=[source_ref],
            )
            candidates.append(candidate)

        # Aggregate candidates (as aggregate.py does)
        field_results = aggregate_field_results(
            field_name="character",
            candidates=candidates,
            field_type=str,
        )

        # Verify aggregation results
        # Aggregation deduplicates identical values, so we expect 1 candidate
        # with all 3 sources merged
        assert len(field_results.candidates) == 1
        assert field_results.candidates[0].value == "Lady Juliet"
        # All 3 sources should be preserved in the merged candidate
        assert len(field_results.candidates[0].sources) == 3
        # Consensus should be detected if confidence is high enough
        if field_results.consensus:
            assert field_results.consensus.value == "Lady Juliet"


class TestStructuredExtractSchemaIOIntegration:
    """Integration tests: structured_extract + schema_io.

    Tests that structured extraction works correctly with schema conversion.
    """

    def test_json_schema_to_pydantic_integration(self) -> None:
        """Test integration: JSON Schema -> Pydantic model -> extraction.

        Verifies that JSON Schema can be converted to Pydantic model and
        used for structured extraction.
        """
        # Step 1: Define JSON Schema
        json_schema = {
            "type": "object",
            "properties": {
                "character": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "emotion": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

        # Step 2: Convert to Pydantic model (as schema_io does)
        schema_model = convert_json_schema_to_pydantic(json_schema)

        # Step 3: Verify model structure
        assert hasattr(schema_model, "model_json_schema")
        schema_dict = schema_model.model_json_schema()
        assert "properties" in schema_dict
        assert "character" in schema_dict["properties"]
        assert "emotion" in schema_dict["properties"]

        # Step 4: Verify model can be used for extraction
        # (actual extraction tested in E2E tests)


# ============================================================================
# E2E TESTS: COMPLETE WORKFLOWS
# These tests verify the entire system works together end-to-end.
# ============================================================================


class TestStructuredExtractionE2E:
    """E2E tests: Complete structured extraction workflows."""

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_ollama_extraction_workflow(self, mock_api_call: MagicMock) -> None:
        """E2E: Complete workflow with Ollama provider."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo"],
            "emotion": ["longing", "love"],
            "relationship": ["sister"],
        }

        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]
            relationship: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet gazed longingly at the stars. Romeo loved her.",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
            model="llama3.1",
            fuzzy_threshold=80,
        )

        assert len(jsonl_lines) == 1
        assert isinstance(jsonl_lines[0], JSONLLine)
        assert jsonl_lines[0].document_id == "doc1"
        assert len(jsonl_lines[0].extractions) > 0
        mock_api_call.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"})
    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_openai_extraction_workflow(self, mock_api_call: MagicMock) -> None:
        """E2E: Complete workflow with OpenAI provider."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
            "emotion": ["longing"],
        }

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
    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_gemini_extraction_workflow(self, mock_api_call: MagicMock) -> None:
        """E2E: Complete workflow with Gemini provider."""
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
            provider="gemini",
            model="gemini-2.5-flash",
        )

        assert len(jsonl_lines) == 1
        assert jsonl_lines[0].document_id == "doc1"

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_multi_document_workflow(self, mock_api_call: MagicMock) -> None:
        """E2E: Process multiple documents correctly."""
        mock_api_call.side_effect = [
            {"character": ["Lady Juliet"]},
            {"character": ["Romeo"]},
            {"character": ["Mercutio"]},
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

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 3
        assert jsonl_lines[0].document_id == "doc1"
        assert jsonl_lines[1].document_id == "doc2"
        assert jsonl_lines[2].document_id == "doc3"
        assert mock_api_call.call_count == 3

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_jsonl_write_and_read_workflow(
        self, mock_api_call: MagicMock, tmp_path: Path
    ) -> None:
        """E2E: JSONL file write and read workflow."""
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

        output_path = tmp_path / "test_output.jsonl"
        write_jsonl(jsonl_lines, output_path)

        assert output_path.exists()
        with output_path.open() as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["document_id"] == "doc1"
            assert len(data["extractions"]) > 0

    @patch("langextract.visualize")
    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_visualization_workflow(
        self,
        mock_api_call: MagicMock,
        mock_visualize: MagicMock,
        tmp_path: Path,
    ) -> None:
        """E2E: Complete visualization workflow."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet"],
        }
        mock_visualize.return_value = "<html><body>Visualization</body></html>"

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

        jsonl_path = tmp_path / "test.jsonl"
        write_jsonl(jsonl_lines, jsonl_path)

        html_path = tmp_path / "visualization.html"
        html_content = visualize_extractions(jsonl_path, output_html_path=html_path)

        assert html_content == "<html><body>Visualization</body></html>"
        assert html_path.exists()
        mock_visualize.assert_called_once_with(str(jsonl_path))

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_error_handling_continues_processing(
        self, mock_api_call: MagicMock
    ) -> None:
        """E2E: Error handling continues processing other documents."""
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

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 2
        assert jsonl_lines[0].document_id == "doc1"
        assert jsonl_lines[1].document_id == "doc2"
        # First doc should have empty extractions due to error
        assert len(jsonl_lines[0].extractions) == 0
        # Second doc should have extractions
        assert len(jsonl_lines[1].extractions) > 0

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_empty_extraction_results(self, mock_api_call: MagicMock) -> None:
        """E2E: Handle empty extraction results gracefully."""
        # Mock API to return empty results (all None or empty arrays)
        mock_api_call.return_value = {
            "character": [],
            "emotion": None,
        }

        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Some text without matches",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        # Should still produce a JSONLLine, but with empty extractions
        assert len(jsonl_lines) == 1
        assert jsonl_lines[0].document_id == "doc1"
        # Empty results should produce no extractions (flattening skips empty arrays)
        assert len(jsonl_lines[0].extractions) == 0

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_nested_schema_extraction(self, mock_api_call: MagicMock) -> None:
        """E2E: Extract from nested schema structures."""
        # Extended Schema format: all fields must be arrays
        mock_api_call.return_value = {
            "person": [
                {
                    "name": "Lady Juliet",
                    "age": 25,
                }
            ],
            "location": [
                {
                    "city": "Verona",
                    "country": "Italy",
                }
            ],
        }

        class PersonModel(BaseModel):
            person: list[dict[str, object]]
            location: list[dict[str, object]]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet, age 25, lived in Verona, Italy.",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=PersonModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 1
        # Nested fields should be flattened with array index notation
        extraction_classes = {
            ext.extraction_class for ext in jsonl_lines[0].extractions
        }
        # Generic dicts in arrays are flattened as field[index].key
        assert "person[0].name" in extraction_classes
        assert "person[0].age" in extraction_classes
        assert "location[0].city" in extraction_classes
        assert "location[0].country" in extraction_classes

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_extended_schema_multiple_values(self, mock_api_call: MagicMock) -> None:
        """E2E: Extended Schema extracts multiple values per field."""
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo", "Mercutio"],
            "emotion": ["love", "longing", "jealousy"],
        }

        class CharacterModel(BaseModel):
            character: list[str]
            emotion: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown=(
                    "Lady Juliet felt love. Romeo felt longing. Mercutio felt jealousy."
                ),
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 1
        # Should have multiple extractions per field
        character_extractions = [
            ext
            for ext in jsonl_lines[0].extractions
            if ext.extraction_class == "character"
        ]
        assert len(character_extractions) == 3
        emotion_extractions = [
            ext
            for ext in jsonl_lines[0].extractions
            if ext.extraction_class == "emotion"
        ]
        assert len(emotion_extractions) == 3

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_error_handling_extraction_failure(self, mock_api_call: MagicMock) -> None:
        """E2E: Error handling when extraction API fails."""
        # Simulate API failure
        mock_api_call.side_effect = RuntimeError("API connection failed")

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet",
                source_map={"spans": {}},
            ),
        ]

        # Should not raise, but return empty extraction record
        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 1
        assert len(jsonl_lines[0].extractions) == 0
        assert jsonl_lines[0].document_id == "doc1"
        assert jsonl_lines[0].text == "Lady Juliet"

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_error_handling_empty_dict_extraction(
        self, mock_api_call: MagicMock
    ) -> None:
        """E2E: Handle empty dict extraction result."""
        # Empty dict is a valid extraction result (no matches found)
        mock_api_call.return_value = {}

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="No characters here",
                source_map={"spans": {}},
            ),
        ]

        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 1
        assert len(jsonl_lines[0].extractions) == 0
        assert jsonl_lines[0].document_id == "doc1"

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_error_handling_flattening_failure(self, mock_api_call: MagicMock) -> None:
        """E2E: Handle case where flattening produces no results despite data."""
        # Return data that won't flatten properly (e.g., wrong structure)
        mock_api_call.return_value = {
            "character": "not an array",  # Should be array but isn't
        }

        class CharacterModel(BaseModel):
            character: list[str]

        corpus_docs = [
            CorpusDocument(
                doc_id="doc1",
                markdown="Lady Juliet",
                source_map={"spans": {}},
            ),
        ]

        # Should handle gracefully and produce empty extractions
        jsonl_lines = run_structured_extraction(
            schema=CharacterModel,
            corpus_docs=corpus_docs,
            provider="ollama",
        )

        assert len(jsonl_lines) == 1
        # Flattening will skip the non-array value, producing no extractions
        assert len(jsonl_lines[0].extractions) == 0
