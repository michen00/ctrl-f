"""Unit tests for structured extraction module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ctrlf.app.models import Candidate, SourceRef
from ctrlf.app.structured_extract import (
    ExtractionRecord,
    JSONLLine,
    _call_structured_extraction_api,
    _extraction_record_to_candidate,
    _flatten_extractions,
    _is_extended_schema,
    _retry_with_exponential_backoff,
    _validate_jsonl_format,
    check_ollama_setup,
    estimate_cost,
    find_char_interval,
    validate_api_key,
    visualize_extractions,
    write_jsonl,
)

# ============================================================================
# EXECUTABLE DOCUMENTATION TESTS
# These tests are "above the fold" - well-commented, DAMP (Descriptive And
# Meaningful Phrases), and serve as tutorials for how to use each component.
# ============================================================================


class TestStructuredExtractionWorkflow:
    """Executable documentation: Complete structured extraction workflow.

    This test demonstrates the complete workflow from schema definition
    through extraction to JSONL output. It serves as a tutorial for users.
    """

    @patch("ctrlf.app.structured_extract._call_structured_extraction_api")
    def test_complete_extraction_workflow_tutorial(
        self, mock_api_call: MagicMock, tmp_path: Path
    ) -> None:
        """Tutorial: How to extract structured data from documents.

        This example shows:
        1. Define an Extended Schema (all fields are arrays)
        2. Create corpus documents
        3. Run structured extraction
        4. Process results

        Extended Schema means all fields are lists, allowing multiple
        extractions per field (e.g., multiple characters in a document).
        """

        # Step 1: Define Extended Schema - all fields are arrays
        # This allows extracting multiple values per field
        class CharacterModel(BaseModel):
            character: list[str]  # Can extract multiple characters
            emotion: list[str]  # Can extract multiple emotions
            relationship: list[str]  # Can extract multiple relationships

        # Step 2: Mock the API response (in real usage, this calls Ollama/OpenAI/Gemini)
        # The API returns data matching the Extended Schema structure
        mock_api_call.return_value = {
            "character": ["Lady Juliet", "Romeo"],  # Multiple characters found
            "emotion": ["longing", "love"],  # Multiple emotions found
            "relationship": ["sister"],  # Single relationship found
        }

        # Step 3: Create corpus documents (in real usage, from process_corpus())
        from ctrlf.app.ingest import CorpusDocument  # noqa: PLC0415

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
        # This processes each document and extracts structured data
        from ctrlf.app.structured_extract import run_structured_extraction  # noqa: PLC0415, I001

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

        # Each extraction includes the field name, value, and location
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


class TestFindCharIntervalTutorial:
    """Executable documentation: Finding character positions in documents.

    This demonstrates how fuzzy matching locates extracted text in source documents.
    """

    def test_find_exact_match_tutorial(self) -> None:
        """Tutorial: Find exact text matches in documents.

        When extracted text exactly matches document text, we get
        "match_exact" status with precise character positions.
        """
        document_text = "Lady Juliet gazed longingly at the stars"
        extracted_text = "Lady Juliet"

        # Find where the extracted text appears in the document
        char_interval, alignment_status = find_char_interval(
            text=document_text,
            extraction_text=extracted_text,
            fuzzy_threshold=80,  # Minimum similarity score (0-100)
        )

        # Exact matches return precise positions
        assert alignment_status == "match_exact"
        assert char_interval["start_pos"] == 0
        assert char_interval["end_pos"] == 11  # Length of "Lady Juliet"

    def test_find_fuzzy_match_tutorial(self) -> None:
        """Tutorial: Find approximate text matches using fuzzy matching.

        When extracted text doesn't exactly match (typos, formatting differences),
        fuzzy matching finds the best approximate location with "match_fuzzy" status.
        """
        document_text = "The quick brown fox jumps over the lazy dog"
        extracted_text = "quick brown fox"  # Exact match in this case

        char_interval, alignment_status = find_char_interval(
            text=document_text,
            extraction_text=extracted_text,
            fuzzy_threshold=80,  # Requires 80% similarity
        )

        # Fuzzy matching finds approximate positions
        assert alignment_status in ("match_exact", "match_fuzzy")
        assert char_interval["start_pos"] >= 0
        assert char_interval["end_pos"] > char_interval["start_pos"]

    def test_find_no_match_tutorial(self) -> None:
        """Tutorial: Handle cases where text cannot be found.

        When extracted text doesn't appear in the document (below threshold),
        we return "no_match" with zero positions.
        """
        document_text = "Lady Juliet gazed at the stars"
        extracted_text = "Completely different text that doesn't exist"

        char_interval, alignment_status = find_char_interval(
            text=document_text,
            extraction_text=extracted_text,
            fuzzy_threshold=80,
        )

        # No match returns zero positions
        assert alignment_status == "no_match"
        assert char_interval == {"start_pos": 0, "end_pos": 0}


class TestFlattenExtractionsTutorial:
    """Executable documentation: Flattening nested extraction data.

    This demonstrates how structured data is flattened into individual
    extractions for processing.
    """

    def test_flatten_simple_schema_tutorial(self) -> None:
        """Tutorial: Flatten simple flat schema (no nesting).

        Simple schemas with string fields produce one extraction per field.
        """
        # Input: Extracted data matching a simple schema
        extracted_data = {
            "character": "Lady Juliet",
            "emotion": "longing",
            "relationship": "sister",
        }

        # Schema definition (JSON Schema format)
        schema = {
            "type": "object",
            "properties": {
                "character": {"type": "string"},
                "emotion": {"type": "string"},
                "relationship": {"type": "string"},
            },
        }

        # Flatten into individual extractions
        extractions = _flatten_extractions(extracted_data, schema)

        # Each field becomes one extraction
        assert len(extractions) == 3
        field_names = {ext.field_name for ext in extractions}
        assert field_names == {"character", "emotion", "relationship"}

        # Each extraction contains the value
        char_extraction = next(
            ext for ext in extractions if ext.field_name == "character"
        )
        assert char_extraction.value == "Lady Juliet"

    def test_flatten_array_schema_tutorial(self) -> None:
        """Tutorial: Flatten Extended Schema with array fields.

        Extended Schema (all fields are arrays) allows multiple values per field.
        Each array item becomes a separate extraction with an index attribute.
        """
        # Input: Extended Schema data (all fields are arrays)
        extracted_data = {
            "character": ["Lady Juliet", "Romeo", "Mercutio"],  # Multiple characters
            "emotion": ["love", "longing"],  # Multiple emotions
        }

        # Extended Schema definition (all fields are arrays)
        schema = {
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

        # Flatten: Each array item becomes a separate extraction
        extractions = _flatten_extractions(extracted_data, schema)

        # Total: 3 characters + 2 emotions = 5 extractions
        assert len(extractions) == 5

        # Character extractions have index attributes
        char_extractions = [ext for ext in extractions if ext.field_name == "character"]
        assert len(char_extractions) == 3
        assert char_extractions[0].value == "Lady Juliet"
        assert char_extractions[0].attributes is not None
        assert char_extractions[0].attributes["index"] == 0

    def test_flatten_nested_schema_tutorial(self) -> None:
        """Tutorial: Flatten nested object schemas.

        Nested objects are flattened with dot notation (e.g., "person.name").
        """
        # Input: Nested data structure
        extracted_data = {
            "person": {
                "name": "Lady Juliet",
                "age": 25,
            },
            "location": {
                "city": "Verona",
                "country": "Italy",
            },
        }

        # Nested schema definition
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                    },
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                },
            },
        }

        # Flatten: Nested fields use dot notation
        extractions = _flatten_extractions(extracted_data, schema)

        # All nested fields are flattened
        field_names = {ext.field_name for ext in extractions}
        assert "person.name" in field_names
        assert "person.age" in field_names
        assert "location.city" in field_names
        assert "location.country" in field_names


class TestExtractionRecordToCandidateTutorial:
    """Executable documentation: Converting ExtractionRecord to Candidate.

    This demonstrates integration with the existing extraction pipeline.
    """

    def test_convert_extraction_record_tutorial(self) -> None:
        """Tutorial: Convert ExtractionRecord to Candidate for pipeline integration.

        ExtractionRecord (from structured extraction) is converted to Candidate
        (used by aggregation pipeline) with proper confidence scores and
        source references.
        """
        # Step 1: Create an ExtractionRecord from structured extraction
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",  # Exact match = high confidence
            extraction_index=1,
            group_index=0,
        )

        # Step 2: Convert to Candidate (for aggregation pipeline)
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {
            "file_path": "romeo_juliet.txt",
            "pages": {"1": {"start": 0, "end": 100}},
        }

        candidate = _extraction_record_to_candidate(
            extraction_record=extraction_record,
            doc_id="doc_romeo_juliet",
            doc_markdown=doc_markdown,
            source_map=source_map,
        )

        # Step 3: Verify Candidate structure
        assert isinstance(candidate, Candidate)
        assert candidate.value == "Lady Juliet"

        # Confidence based on alignment status:
        # - match_exact = 0.9 (high confidence)
        # - match_fuzzy = 0.7 (medium confidence)
        # - no_match = 0.5 (low confidence)
        assert candidate.confidence == 0.9  # match_exact

        # Source reference includes location and snippet
        assert len(candidate.sources) == 1
        source_ref = candidate.sources[0]
        assert isinstance(source_ref, SourceRef)
        assert source_ref.doc_id == "doc_romeo_juliet"
        assert "Lady Juliet" in source_ref.snippet  # Context snippet


# ============================================================================
# COVERAGE & RELIABILITY TESTS
# These tests focus on edge cases, path coverage, and confidence.
# They can be more DRY but must remain readable.
# ============================================================================


class TestExtractionRecord:
    """Test ExtractionRecord model validation."""

    def test_valid_extraction_record(self) -> None:
        """Test creating a valid ExtractionRecord."""
        record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        assert record.extraction_class == "character"
        assert record.extraction_text == "Lady Juliet"
        assert record.char_interval == {"start_pos": 0, "end_pos": 11}
        assert record.alignment_status == "match_exact"
        assert record.extraction_index == 1
        assert record.group_index == 0
        assert record.description is None
        assert record.attributes is None

    def test_extraction_record_with_optional_fields(self) -> None:
        """Test ExtractionRecord with optional description and attributes."""
        record = ExtractionRecord(
            extraction_class="emotion",
            extraction_text="happy",
            char_interval={"start_pos": 5, "end_pos": 10},
            alignment_status="match_fuzzy",
            extraction_index=2,
            group_index=1,
            description="Emotion extraction",
            attributes={"array_index": 0, "nested_path": "person.emotions"},
        )
        assert record.description == "Emotion extraction"
        assert record.attributes == {"array_index": 0, "nested_path": "person.emotions"}

    def test_extraction_record_empty_class_allowed(self) -> None:
        """Test that empty extraction_class is allowed (Pydantic v2 behavior)."""
        record = ExtractionRecord(
            extraction_class="",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        assert record.extraction_class == ""

    def test_extraction_record_empty_text_allowed(self) -> None:
        """Test that empty extraction_text is allowed (Pydantic v2 behavior)."""
        record = ExtractionRecord(
            extraction_class="test",
            extraction_text="",
            char_interval={"start_pos": 0, "end_pos": 0},
            alignment_status="no_match",
            extraction_index=1,
            group_index=0,
        )
        assert record.extraction_text == ""

    def test_extraction_record_invalid_char_interval_allowed(self) -> None:
        """Test that invalid char_interval is allowed (no validation)."""
        record = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 10, "end_pos": 5},  # end < start
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        assert record.char_interval == {"start_pos": 10, "end_pos": 5}

    def test_extraction_record_invalid_alignment_status_allowed(self) -> None:
        """Test that invalid alignment_status is allowed (no enum validation)."""
        record = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="invalid_status",
            extraction_index=1,
            group_index=0,
        )
        assert record.alignment_status == "invalid_status"

    def test_extraction_record_negative_index_allowed(self) -> None:
        """Test that zero/negative extraction_index is allowed (no range validation)."""
        record = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="match_exact",
            extraction_index=0,  # Must be >= 1 per spec, but not validated
            group_index=0,
        )
        assert record.extraction_index == 0


class TestJSONLLine:
    """Test JSONLLine model validation."""

    def test_valid_jsonl_line(self) -> None:
        """Test creating a valid JSONLLine."""
        extraction = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line = JSONLLine(
            extractions=[extraction],
            text="Lady Juliet gazed at the stars",
            document_id="doc_abc123",
        )
        assert len(jsonl_line.extractions) == 1
        assert jsonl_line.text == "Lady Juliet gazed at the stars"
        assert jsonl_line.document_id == "doc_abc123"

    def test_jsonl_line_empty_extractions_allowed(self) -> None:
        """Test that empty extractions list is allowed."""
        jsonl_line = JSONLLine(
            extractions=[],
            text="Document with no extractions",
            document_id="doc_empty",
        )
        assert len(jsonl_line.extractions) == 0
        assert jsonl_line.text == "Document with no extractions"

    def test_jsonl_line_empty_text_allowed(self) -> None:
        """Test that empty text is allowed (Pydantic v2 behavior)."""
        extraction = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line = JSONLLine(
            extractions=[extraction],
            text="",
            document_id="doc_test",
        )
        assert jsonl_line.text == ""

    def test_jsonl_line_empty_document_id_allowed(self) -> None:
        """Test that empty document_id is allowed (Pydantic v2 behavior)."""
        extraction = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line = JSONLLine(
            extractions=[extraction],
            text="Test document",
            document_id="",
        )
        assert jsonl_line.document_id == ""


class TestFindCharInterval:
    """Test find_char_interval function."""

    def test_find_char_interval_exact_match(self) -> None:
        """Test find_char_interval with exact match."""
        text = "Lady Juliet gazed longingly at the stars"
        extraction_text = "Lady Juliet"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert char_interval == {"start_pos": 0, "end_pos": 11}
        assert alignment_status == "match_exact"

    def test_find_char_interval_exact_match_case_insensitive(self) -> None:
        """Test find_char_interval with case-insensitive exact match."""
        text = "lady juliet gazed longingly at the stars"
        extraction_text = "Lady Juliet"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert alignment_status == "match_exact"
        assert char_interval["start_pos"] == 0
        assert char_interval["end_pos"] == 11

    def test_find_char_interval_fuzzy_match(self) -> None:
        """Test find_char_interval with fuzzy match."""
        text = "Lady Juliet gazed longingly at the stars"
        extraction_text = "Lady Juliet"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert alignment_status in ("match_exact", "match_fuzzy")
        assert char_interval["start_pos"] >= 0
        assert char_interval["end_pos"] > char_interval["start_pos"]

    def test_find_char_interval_fuzzy_match_low_threshold(self) -> None:
        """Test find_char_interval with fuzzy match using low threshold."""
        text = "The quick brown fox jumps over the lazy dog"
        extraction_text = "quick brown"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=50
        )
        assert alignment_status in ("match_exact", "match_fuzzy")
        assert char_interval["start_pos"] >= 0

    def test_find_char_interval_no_match(self) -> None:
        """Test find_char_interval with no match."""
        text = "Lady Juliet gazed longingly at the stars"
        extraction_text = "Completely different text that doesn't exist"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert alignment_status == "no_match"
        assert char_interval == {"start_pos": 0, "end_pos": 0}

    def test_find_char_interval_no_match_high_threshold(self) -> None:
        """Test find_char_interval with high threshold requiring exact match."""
        text = "The quick brown fox"
        extraction_text = "The quick brown foxx"  # Extra character
        char_interval, alignment_status = find_char_interval(
            text,
            extraction_text,
            fuzzy_threshold=100,  # Very high threshold
        )
        assert alignment_status in ("match_exact", "match_fuzzy", "no_match")
        assert char_interval["start_pos"] >= 0

    def test_find_char_interval_empty_text(self) -> None:
        """Test find_char_interval with empty text."""
        text = ""
        extraction_text = "test"
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert alignment_status == "no_match"
        assert char_interval == {"start_pos": 0, "end_pos": 0}

    def test_find_char_interval_empty_extraction_text(self) -> None:
        """Test find_char_interval with empty extraction text."""
        text = "Some text here"
        extraction_text = ""
        char_interval, alignment_status = find_char_interval(
            text, extraction_text, fuzzy_threshold=80
        )
        assert alignment_status == "no_match"
        assert char_interval == {"start_pos": 0, "end_pos": 0}


class TestFlattenExtractions:
    """Test _flatten_extractions function."""

    def test_flatten_extractions_flat_schema(self) -> None:
        """Test _flatten_extractions with flat schema."""
        data = {
            "character": "Lady Juliet",
            "emotion": "longing",
            "relationship": "sister",
        }
        schema = {
            "type": "object",
            "properties": {
                "character": {"type": "string"},
                "emotion": {"type": "string"},
                "relationship": {"type": "string"},
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 3
        field_names = [r.field_name for r in result]
        assert "character" in field_names
        assert "emotion" in field_names
        assert "relationship" in field_names
        char_extraction = next(r for r in result if r.field_name == "character")
        assert char_extraction.value == "Lady Juliet"
        assert char_extraction.attributes is None

    def test_flatten_extractions_nested_objects(self) -> None:
        """Test _flatten_extractions with nested objects."""
        data = {
            "person": {
                "name": "Lady Juliet",
                "age": 25,
            },
            "location": {
                "city": "Verona",
                "country": "Italy",
            },
        }
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                    },
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                },
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 4
        field_names = [r.field_name for r in result]
        assert "person.name" in field_names
        assert "person.age" in field_names
        assert "location.city" in field_names
        assert "location.country" in field_names
        name_extraction = next(r for r in result if r.field_name == "person.name")
        assert name_extraction.value == "Lady Juliet"
        assert name_extraction.attributes is None

    def test_flatten_extractions_arrays(self) -> None:
        """Test _flatten_extractions with arrays."""
        data = {
            "characters": ["Lady Juliet", "Romeo", "Mercutio"],
            "emotions": ["love", "longing"],
        }
        schema = {
            "type": "object",
            "properties": {
                "characters": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "emotions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 5  # 3 characters + 2 emotions
        field_names = [r.field_name for r in result]
        assert "characters" in field_names
        assert "emotions" in field_names
        char_extractions = [r for r in result if r.field_name == "characters"]
        assert len(char_extractions) == 3
        char0 = char_extractions[0]
        assert char0.value == "Lady Juliet"
        assert char0.attributes is not None
        assert char0.attributes.get("index") == 0

    def test_flatten_extractions_empty_data(self) -> None:
        """Test _flatten_extractions with empty data."""
        data: dict[str, Any] = {}
        schema = {
            "type": "object",
            "properties": {
                "character": {"type": "string"},
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 0

    def test_flatten_extractions_missing_properties(self) -> None:
        """Test _flatten_extractions with schema missing properties."""
        data = {"character": "Lady Juliet"}
        schema = {"type": "object"}  # No properties key
        result = _flatten_extractions(data, schema)
        assert len(result) == 0

    def test_flatten_extractions_none_values_skipped(self) -> None:
        """Test _flatten_extractions skips None values."""
        data = {
            "character": "Lady Juliet",
            "emotion": None,  # Should be skipped
            "relationship": "sister",
        }
        schema = {
            "type": "object",
            "properties": {
                "character": {"type": "string"},
                "emotion": {"type": "string"},
                "relationship": {"type": "string"},
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 2
        field_names = {r.field_name for r in result}
        assert "character" in field_names
        assert "relationship" in field_names
        assert "emotion" not in field_names

    def test_flatten_extractions_empty_arrays_skipped(self) -> None:
        """Test _flatten_extractions skips empty arrays."""
        data = {
            "character": ["Lady Juliet"],
            "emotion": [],  # Empty array should be skipped
        }
        schema = {
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
        result = _flatten_extractions(data, schema)
        assert len(result) == 1
        assert result[0].field_name == "character"

    def test_flatten_extractions_nested_arrays(self) -> None:
        """Test _flatten_extractions with nested objects in arrays."""
        data = {
            "people": [
                {"name": "Lady Juliet", "age": 25},
                {"name": "Romeo", "age": 20},
            ],
        }
        schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"},
                        },
                    },
                },
            },
        }
        result = _flatten_extractions(data, schema)
        assert len(result) == 4  # 2 people * 2 fields each
        field_names = {r.field_name for r in result}
        assert "people[0].name" in field_names
        assert "people[0].age" in field_names
        assert "people[1].name" in field_names
        assert "people[1].age" in field_names

    def test_flatten_extractions_generic_dict_in_array(self) -> None:
        """Test _flatten_extractions with generic dict[str, object] in arrays."""
        # Generic dicts (without properties schema) should be flattened directly
        data = {
            "people": [
                {"name": "Lady Juliet", "age": 25, "active": True},
                {"name": "Romeo", "age": 20, "active": False},
            ],
        }
        schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        # No properties - generic dict[str, object]
                    },
                },
            },
        }
        result = _flatten_extractions(data, schema)
        # Should flatten all keys from both dicts
        assert len(result) == 6  # 2 people * 3 fields each
        field_names = {r.field_name for r in result}
        assert "people[0].name" in field_names
        assert "people[0].age" in field_names
        assert "people[0].active" in field_names
        assert "people[1].name" in field_names
        assert "people[1].age" in field_names
        assert "people[1].active" in field_names

        # Check values are converted to strings
        age_extraction = next(r for r in result if r.field_name == "people[0].age")
        assert age_extraction.value == "25"
        active_extraction = next(
            r for r in result if r.field_name == "people[0].active"
        )
        assert active_extraction.value == "True"

    def test_flatten_extractions_generic_dict_none_values(self) -> None:
        """Test _flatten_extractions with None values in generic dicts."""
        # None values should be skipped in generic dicts
        data = {
            "people": [
                {"name": "Lady Juliet", "age": None, "city": "Verona"},
            ],
        }
        schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        # No properties - generic dict[str, object]
                    },
                },
            },
        }
        result = _flatten_extractions(data, schema)
        # Should only include non-None values
        assert len(result) == 2  # name and city, age is None
        field_names = {r.field_name for r in result}
        assert "people[0].name" in field_names
        assert "people[0].city" in field_names
        assert "people[0].age" not in field_names

    def test_flatten_extractions_array_field_non_list_value(self) -> None:
        """Test _flatten_extractions when array field has non-list value."""
        # If schema says array but value is not a list, should skip
        data = {
            "characters": "Lady Juliet",  # Should be a list but is a string
            "emotions": ["love", "longing"],  # Correctly a list
        }
        schema = {
            "type": "object",
            "properties": {
                "characters": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "emotions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        result = _flatten_extractions(data, schema)
        # Should only process emotions (correctly a list)
        assert len(result) == 2  # 2 emotions
        field_names = {r.field_name for r in result}
        assert "characters" not in field_names
        assert "emotions" in field_names

    def test_flatten_extractions_nested_with_properties_vs_generic(self) -> None:
        """Test _flatten_extractions distinguishes properties vs generic dict."""
        # Test that nested objects with properties schema use recursive flattening
        # while generic dicts use direct key flattening
        data = {
            "people_with_schema": [
                {"name": "Lady Juliet", "age": 25},
            ],
            "people_generic": [
                {"name": "Romeo", "city": "Verona"},
            ],
        }
        schema = {
            "type": "object",
            "properties": {
                "people_with_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"},
                        },
                    },
                },
                "people_generic": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        # No properties - generic dict
                    },
                },
            },
        }
        result = _flatten_extractions(data, schema)
        # Both should produce extractions but with different field name formats
        field_names = {r.field_name for r in result}
        # Properties schema uses recursive flattening: people_with_schema[0].name
        assert "people_with_schema[0].name" in field_names
        assert "people_with_schema[0].age" in field_names
        # Generic dict uses direct key flattening: people_generic[0].name
        assert "people_generic[0].name" in field_names
        assert "people_generic[0].city" in field_names
        assert len(result) == 4


class TestIsExtendedSchema:
    """Test _is_extended_schema helper function."""

    def test_is_extended_schema_true(self) -> None:
        """Test _is_extended_schema returns True for Extended Schema."""
        schema = {
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
        assert _is_extended_schema(schema) is True

    def test_is_extended_schema_false_non_array(self) -> None:
        """Test _is_extended_schema returns False for non-array fields."""
        schema = {
            "type": "object",
            "properties": {
                "character": {"type": "string"},  # Not an array
                "emotion": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        assert _is_extended_schema(schema) is False

    def test_is_extended_schema_false_mixed(self) -> None:
        """Test _is_extended_schema returns False for mixed array/non-array."""
        schema = {
            "type": "object",
            "properties": {
                "character": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "emotion": {"type": "string"},  # Not an array
            },
        }
        assert _is_extended_schema(schema) is False

    def test_is_extended_schema_false_no_properties(self) -> None:
        """Test _is_extended_schema returns False when no properties."""
        schema = {"type": "object"}
        assert _is_extended_schema(schema) is False

    def test_is_extended_schema_false_empty_properties(self) -> None:
        """Test _is_extended_schema returns False for empty properties."""
        schema = {"type": "object", "properties": {}}
        assert _is_extended_schema(schema) is False


class TestExtractionRecordToCandidate:
    """Test _extraction_record_to_candidate conversion function."""

    def test_extraction_record_to_candidate_match_exact(self) -> None:
        """Test conversion with match_exact alignment status."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {"file_path": "test.txt", "spans": {}}

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        assert isinstance(candidate, Candidate)
        assert candidate.value == "Lady Juliet"
        assert candidate.confidence == 0.9  # match_exact = 0.9
        assert len(candidate.sources) == 1
        assert candidate.sources[0].doc_id == "doc1"

    def test_extraction_record_to_candidate_match_fuzzy(self) -> None:
        """Test conversion with match_fuzzy alignment status."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_fuzzy",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {"file_path": "test.txt", "spans": {}}

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        assert candidate.confidence == 0.7  # match_fuzzy = 0.7

    def test_extraction_record_to_candidate_no_match(self) -> None:
        """Test conversion with no_match alignment status."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="no_match",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {"file_path": "test.txt", "spans": {}}

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        assert candidate.confidence == 0.5  # no_match = 0.5

    def test_extraction_record_to_candidate_with_pages(self) -> None:
        """Test conversion with page information in source_map."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 50, "end_pos": 61},  # Within page range
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Lady Juliet gazed longingly at the stars"
        source_map = {
            "file_path": "test.txt",
            "pages": {
                "1": {"start": 0, "end": 100},
            },
            "spans": {},
        }

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        assert candidate.sources[0].location == "page 1"

    def test_extraction_record_to_candidate_snippet_extraction(self) -> None:
        """Test that snippet is extracted with context."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 50, "end_pos": 61},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Some text before. Lady Juliet gazed longingly. Some text after."
        source_map = {"file_path": "test.txt", "spans": {}}

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        snippet = candidate.sources[0].snippet
        assert "Lady Juliet" in snippet
        assert len(snippet) > len("Lady Juliet")  # Should include context

    def test_extraction_record_to_candidate_short_document(self) -> None:
        """Test snippet extraction for very short documents."""
        extraction_record = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        doc_markdown = "Lady Juliet"  # Very short document
        source_map = {"file_path": "test.txt", "spans": {}}

        candidate = _extraction_record_to_candidate(
            extraction_record, "doc1", doc_markdown, source_map
        )

        # Should use full document as snippet for short docs
        assert candidate.sources[0].snippet == "Lady Juliet"


class TestCallStructuredExtractionAPI:
    """Test _call_structured_extraction_api function."""

    @patch("pydantic_ai.Agent")
    def test_call_structured_extraction_api_ollama(
        self, mock_agent_class: MagicMock
    ) -> None:
        """Test _call_structured_extraction_api with Ollama provider."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {
            "character": ["Lady Juliet"],
            "emotion": ["longing"],
        }
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]
            emotion: list[str]

        text = "Lady Juliet gazed longingly at the stars"
        result = _call_structured_extraction_api(
            text=text,
            schema_model=TestSchema,
            provider="ollama",
            model="llama3.1",
        )
        assert isinstance(result, dict)
        assert "character" in result
        assert "emotion" in result
        assert result["character"] == ["Lady Juliet"]
        assert result["emotion"] == ["longing"]

    @patch("pydantic_ai.Agent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_call_structured_extraction_api_openai(
        self, mock_agent_class: MagicMock
    ) -> None:
        """Test _call_structured_extraction_api with OpenAI provider."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {
            "character": ["Lady Juliet"],
        }
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]

        text = "Lady Juliet gazed longingly"
        result = _call_structured_extraction_api(
            text=text,
            schema_model=TestSchema,
            provider="openai",
            model="gpt-4o",
        )
        assert isinstance(result, dict)
        assert "character" in result

    @patch("pydantic_ai.Agent")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_google_key_12345678901234567890"})
    def test_call_structured_extraction_api_gemini(
        self, mock_agent_class: MagicMock
    ) -> None:
        """Test _call_structured_extraction_api with Gemini provider."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {
            "character": ["Lady Juliet"],
        }
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]

        text = "Lady Juliet gazed longingly"
        result = _call_structured_extraction_api(
            text=text,
            schema_model=TestSchema,
            provider="gemini",
            model="gemini-2.5-flash",
        )
        assert isinstance(result, dict)
        assert "character" in result

    @patch("pydantic_ai.Agent")
    def test_call_structured_extraction_api_error_handling(
        self, mock_agent_class: MagicMock
    ) -> None:
        """Test _call_structured_extraction_api error handling."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = RuntimeError("API call failed")
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"
        with pytest.raises(RuntimeError, match="Extraction failed"):
            _call_structured_extraction_api(
                text=text,
                schema_model=TestSchema,
                provider="ollama",
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_call_structured_extraction_api_missing_api_key(self) -> None:
        """Test _call_structured_extraction_api with missing API key."""

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _call_structured_extraction_api(
                text=text,
                schema_model=TestSchema,
                provider="openai",
            )


class TestRetryWithExponentialBackoff:
    """Test _retry_with_exponential_backoff function."""

    def test_retry_succeeds_on_first_attempt(self) -> None:
        """Test retry succeeds immediately without retries."""
        call_count = 0

        def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = _retry_with_exponential_backoff(successful_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_retries(self) -> None:
        """Test retry succeeds after initial failures."""
        call_count = 0

        def retryable_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                msg = "Temporary failure"
                raise TimeoutError(msg)
            return "success"

        result = _retry_with_exponential_backoff(
            retryable_func, max_retries=3, initial_delay=0.01
        )
        assert result == "success"
        assert call_count == 2

    def test_retry_exhausts_retries(self) -> None:
        """Test retry raises exception after max retries."""
        call_count = 0

        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            msg = "Persistent failure"
            raise TimeoutError(msg)

        with pytest.raises(TimeoutError, match="Persistent failure"):
            _retry_with_exponential_backoff(
                failing_func, max_retries=2, initial_delay=0.01
            )
        assert call_count == 3  # Initial + 2 retries

    def test_retry_non_retryable_error_immediate_fail(self) -> None:
        """Test retry immediately fails on non-retryable errors."""
        call_count = 0

        def auth_error_func() -> str:
            nonlocal call_count
            call_count += 1
            msg = "Authentication error 401"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Authentication error"):
            _retry_with_exponential_backoff(
                auth_error_func, max_retries=3, initial_delay=0.01
            )
        assert call_count == 1  # No retries for auth errors

    def test_retry_rate_limit_error(self) -> None:
        """Test retry handles rate limit errors."""
        call_count = 0

        def rate_limit_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                msg = "Rate limit 429"
                raise RuntimeError(msg)
            return "success"

        result = _retry_with_exponential_backoff(
            rate_limit_func, max_retries=3, initial_delay=0.01
        )
        assert result == "success"
        assert call_count == 2


class TestCheckOllamaSetup:
    """Test check_ollama_setup function."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_ollama_setup_success(
        self, mock_subprocess: MagicMock, mock_which: MagicMock
    ) -> None:
        """Test check_ollama_setup succeeds when Ollama is available."""
        mock_which.return_value = "/usr/local/bin/ollama"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "llama3.1\nllama3\n"
        mock_subprocess.return_value = mock_result

        # Should not raise
        check_ollama_setup("llama3.1")

    @patch("shutil.which")
    def test_check_ollama_setup_not_installed(self, mock_which: MagicMock) -> None:
        """Test check_ollama_setup raises when Ollama not installed."""
        mock_which.return_value = None

        with pytest.raises(RuntimeError, match="Ollama is not installed"):
            check_ollama_setup("llama3.1")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_ollama_setup_not_running(
        self, mock_subprocess: MagicMock, mock_which: MagicMock
    ) -> None:
        """Test check_ollama_setup raises when Ollama not running."""
        mock_which.return_value = "/usr/local/bin/ollama"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection refused"
        mock_subprocess.return_value = mock_result

        with pytest.raises(RuntimeError, match="Ollama is not running"):
            check_ollama_setup("llama3.1")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_ollama_setup_model_not_available(
        self, mock_subprocess: MagicMock, mock_which: MagicMock
    ) -> None:
        """Test check_ollama_setup raises when model not available."""
        mock_which.return_value = "/usr/local/bin/ollama"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "llama3\n"  # llama3.1 not in list
        mock_subprocess.return_value = mock_result

        with pytest.raises(RuntimeError, match="not available"):
            check_ollama_setup("llama3.1")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_ollama_setup_timeout(
        self, mock_subprocess: MagicMock, mock_which: MagicMock
    ) -> None:
        """Test check_ollama_setup handles timeout."""
        import subprocess  # noqa: PLC0415

        mock_which.return_value = "/usr/local/bin/ollama"
        mock_subprocess.side_effect = subprocess.TimeoutExpired("ollama", 5)

        with pytest.raises(RuntimeError, match="not responding"):
            check_ollama_setup("llama3.1")


class TestEstimateCost:
    """Test estimate_cost function."""

    def test_estimate_cost_ollama_free(self) -> None:
        """Test cost estimation for Ollama (free)."""
        cost = estimate_cost(
            tokens_input=1000,
            tokens_output=500,
            provider="ollama",
            model="llama3.1",
        )
        assert cost["total_cost"] == 0.0
        assert cost["provider"] == "ollama"
        assert cost["note"] == "Ollama is free (local only, no API costs)"

    def test_estimate_cost_openai(self) -> None:
        """Test cost estimation for OpenAI."""
        cost = estimate_cost(
            tokens_input=1000000,  # 1M tokens
            tokens_output=500000,  # 0.5M tokens
            provider="openai",
            model="gpt-4o",
        )
        # gpt-4o: $2.50/$10 per 1M tokens
        # Input: 1M * $2.50 = $2.50
        # Output: 0.5M * $10 = $5.00
        # Total: $7.50
        assert cost["total_cost"] == pytest.approx(7.50, abs=0.01)
        assert cost["provider"] == "openai"
        assert cost["model"] == "gpt-4o"

    def test_estimate_cost_gemini(self) -> None:
        """Test cost estimation for Gemini."""
        cost = estimate_cost(
            tokens_input=1000000,  # 1M tokens
            tokens_output=500000,  # 0.5M tokens
            provider="gemini",
            model="gemini-2.5-flash",
        )
        # gemini-2.5-flash: $0.075/$0.30 per 1M tokens
        # Input: 1M * $0.075 = $0.075
        # Output: 0.5M * $0.30 = $0.15
        # Total: $0.225
        assert cost["total_cost"] == pytest.approx(0.225, abs=0.01)
        assert cost["provider"] == "gemini"
        assert cost["model"] == "gemini-2.5-flash"

    def test_estimate_cost_default_model(self) -> None:
        """Test cost estimation with default model."""
        cost = estimate_cost(
            tokens_input=1000,
            tokens_output=500,
            provider="openai",
            model=None,  # Should use default
        )
        assert cost["model"] == "gpt-4o"  # Default for OpenAI


class TestWriteJSONL:
    """Test write_jsonl function."""

    def test_write_jsonl_single_line(self, tmp_path: Path) -> None:
        """Test write_jsonl with single JSONLLine."""
        extraction = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line = JSONLLine(
            extractions=[extraction],
            text="Lady Juliet gazed at the stars",
            document_id="doc1",
        )
        output_path = tmp_path / "test.jsonl"
        write_jsonl([jsonl_line], output_path)
        assert output_path.exists()
        with output_path.open() as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["document_id"] == "doc1"
            assert len(data["extractions"]) == 1

    def test_write_jsonl_multiple_lines(self, tmp_path: Path) -> None:
        """Test write_jsonl with multiple JSONLLine objects."""
        extraction1 = ExtractionRecord(
            extraction_class="character",
            extraction_text="Lady Juliet",
            char_interval={"start_pos": 0, "end_pos": 11},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        extraction2 = ExtractionRecord(
            extraction_class="character",
            extraction_text="Romeo",
            char_interval={"start_pos": 0, "end_pos": 5},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line1 = JSONLLine(
            extractions=[extraction1],
            text="Lady Juliet gazed",
            document_id="doc1",
        )
        jsonl_line2 = JSONLLine(
            extractions=[extraction2],
            text="Romeo spoke",
            document_id="doc2",
        )
        output_path = tmp_path / "test.jsonl"
        write_jsonl([jsonl_line1, jsonl_line2], output_path)
        assert output_path.exists()
        with output_path.open() as f:
            lines = f.readlines()
            assert len(lines) == 2
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            assert data1["document_id"] == "doc1"
            assert data2["document_id"] == "doc2"

    def test_write_jsonl_empty_extractions(self, tmp_path: Path) -> None:
        """Test write_jsonl with empty extractions."""
        jsonl_line = JSONLLine(
            extractions=[],
            text="Document with no extractions",
            document_id="doc_empty",
        )
        output_path = tmp_path / "test.jsonl"
        write_jsonl([jsonl_line], output_path)
        assert output_path.exists()
        with output_path.open() as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert len(data["extractions"]) == 0

    def test_write_jsonl_invalid_path(self) -> None:
        """Test write_jsonl with invalid path raises error."""
        extraction = ExtractionRecord(
            extraction_class="test",
            extraction_text="test",
            char_interval={"start_pos": 0, "end_pos": 4},
            alignment_status="match_exact",
            extraction_index=1,
            group_index=0,
        )
        jsonl_line = JSONLLine(
            extractions=[extraction],
            text="Test",
            document_id="doc1",
        )
        invalid_path = Path("/nonexistent/directory/test.jsonl")
        with pytest.raises((OSError, PermissionError)):
            write_jsonl([jsonl_line], invalid_path)


class TestValidateAPIKey:
    """Test validate_api_key function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_validate_api_key_openai_valid(self) -> None:
        """Test validate_api_key with valid OpenAI key."""
        validate_api_key("openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_key_openai_missing(self) -> None:
        """Test validate_api_key with missing OpenAI key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            validate_api_key("openai")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_google_key_12345678901234567890"})
    def test_validate_api_key_gemini_valid(self) -> None:
        """Test validate_api_key with valid Gemini key."""
        validate_api_key("gemini")

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_key_gemini_missing(self) -> None:
        """Test validate_api_key with missing Gemini key."""
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            validate_api_key("gemini")

    def test_validate_api_key_ollama_no_key_needed(self) -> None:
        """Test validate_api_key with Ollama (no key needed)."""
        validate_api_key("ollama")

    def test_validate_api_key_invalid_provider(self) -> None:
        """Test validate_api_key with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            validate_api_key("invalid_provider")


class TestProviderSpecificBehavior:
    """Test provider-specific behavior."""

    @patch("pydantic_ai.Agent")
    def test_call_structured_extraction_api_provider_model_strings(
        self, mock_agent_class: MagicMock
    ) -> None:
        """Test that correct model strings are used for each provider."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {"character": ["test"]}
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"

        # Test Ollama with default model
        _call_structured_extraction_api(
            text=text, schema_model=TestSchema, provider="ollama", model=None
        )
        assert mock_agent_class.called
        call_args = mock_agent_class.call_args
        assert call_args[0][0] == "ollama:llama3.1"

        mock_agent_class.reset_mock()

        # Test OpenAI with default model
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="openai", model=None
            )
            call_args = mock_agent_class.call_args
            assert call_args[0][0] == "openai:gpt-4o"

        mock_agent_class.reset_mock()

        # Test Gemini with default model
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_google_key_12345678901234567890"}
        ):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="gemini", model=None
            )
            call_args = mock_agent_class.call_args
            assert call_args[0][0] == "google-gla:gemini-2.5-flash"

    @patch("pydantic_ai.Agent")
    @patch("ctrlf.app.structured_extract.check_ollama_setup")
    def test_call_structured_extraction_api_custom_models(
        self,
        _mock_check_ollama: MagicMock,  # noqa: PT019
        mock_agent_class: MagicMock,
    ) -> None:
        """Test that custom models are used when provided."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {"character": ["test"]}
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"

        # Test Ollama with custom model
        _call_structured_extraction_api(
            text=text, schema_model=TestSchema, provider="ollama", model="llama3.2"
        )
        call_args = mock_agent_class.call_args
        assert call_args[0][0] == "ollama:llama3.2"

        mock_agent_class.reset_mock()

        # Test OpenAI with custom model
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            _call_structured_extraction_api(
                text=text,
                schema_model=TestSchema,
                provider="openai",
                model="gpt-4-turbo",
            )
            call_args = mock_agent_class.call_args
            assert call_args[0][0] == "openai:gpt-4-turbo"

        mock_agent_class.reset_mock()

        # Test Gemini with custom model
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_google_key_12345678901234567890"}
        ):
            _call_structured_extraction_api(
                text=text,
                schema_model=TestSchema,
                provider="gemini",
                model="gemini-2.0-flash-exp",
            )
            call_args = mock_agent_class.call_args
            assert call_args[0][0] == "google-gla:gemini-2.0-flash-exp"


class TestProviderSpecificErrorHandling:
    """Test provider-specific error handling."""

    @patch("ctrlf.app.structured_extract.check_ollama_setup")
    @patch("pydantic_ai.Agent")
    def test_provider_specific_api_key_validation(
        self,
        mock_agent_class: MagicMock,
        _mock_check_ollama: MagicMock,  # noqa: PT019
    ) -> None:
        """Test API key validation per provider."""

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"

        # Ollama doesn't require API key
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.output.model_dump.return_value = {"character": ["test"]}
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        with patch.dict(os.environ, {}, clear=True):
            result = _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="ollama"
            )
            assert result == {"character": ["test"]}

        # OpenAI requires API key
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY"),
        ):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="openai"
            )

        # Gemini requires API key
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GOOGLE_API_KEY"),
        ):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="gemini"
            )

    def test_provider_specific_error_messages(self) -> None:
        """Test that error messages are provider-specific."""

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"

        # Test OpenAI API key error message
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY") as exc_info,
        ):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="openai"
            )
        assert "openai" in str(exc_info.value).lower()

        # Test Gemini API key error message
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GOOGLE_API_KEY") as exc_info,
        ):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="gemini"
            )
        assert (
            "gemini" in str(exc_info.value).lower()
            or "google" in str(exc_info.value).lower()
        )

    def test_provider_specific_unsupported_provider_error(self) -> None:
        """Test error handling for unsupported providers."""

        class TestSchema(BaseModel):
            character: list[str]

        text = "Test text"

        with pytest.raises(ValueError, match="Unsupported provider"):
            _call_structured_extraction_api(
                text=text, schema_model=TestSchema, provider="invalid_provider"
            )


class TestVisualizeExtractions:
    """Test visualize_extractions function."""

    def test_visualize_extractions_with_valid_jsonl(self, tmp_path: Path) -> None:
        """Test visualize_extractions with valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[
                    ExtractionRecord(
                        extraction_class="character",
                        extraction_text="Lady Juliet",
                        char_interval={"start_pos": 0, "end_pos": 11},
                        alignment_status="match_exact",
                        extraction_index=1,
                        group_index=0,
                    )
                ],
                text="Lady Juliet was a character in the play.",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = "<html><body>Visualization</body></html>"

            html_content = visualize_extractions(jsonl_file)

            assert html_content == "<html><body>Visualization</body></html>"
            mock_visualize.assert_called_once_with(str(jsonl_file))

    def test_visualize_extractions_with_string_return(self, tmp_path: Path) -> None:
        """Test visualize_extractions when langextract returns string directly."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = "<html>String HTML</html>"

            html_content = visualize_extractions(jsonl_file)

            assert html_content == "<html>String HTML</html>"

    def test_visualize_extractions_with_data_attribute(self, tmp_path: Path) -> None:
        """Test visualize_extractions when langextract returns object with .data."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        mock_html_obj = MagicMock()
        mock_html_obj.data = "<html>Data HTML</html>"

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = mock_html_obj

            html_content = visualize_extractions(jsonl_file)

            assert html_content == "<html>Data HTML</html>"

    def test_visualize_extractions_with_html_attribute(self, tmp_path: Path) -> None:
        """Test visualize_extractions when langextract returns object with .html."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        mock_html_obj = MagicMock()
        mock_html_obj.html = "<html>HTML Attribute</html>"
        del mock_html_obj.data

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = mock_html_obj

            html_content = visualize_extractions(jsonl_file)

            assert html_content == "<html>HTML Attribute</html>"

    def test_visualize_extractions_saves_to_file(self, tmp_path: Path) -> None:
        """Test visualize_extractions saves HTML to output file."""
        jsonl_file = tmp_path / "test.jsonl"
        output_html = tmp_path / "output.html"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = "<html>Saved HTML</html>"

            html_content = visualize_extractions(jsonl_file, output_html)

            assert html_content == "<html>Saved HTML</html>"
            assert output_html.exists()
            assert output_html.read_text() == "<html>Saved HTML</html>"

    def test_visualize_extractions_file_not_found(self) -> None:
        """Test visualize_extractions raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            visualize_extractions("nonexistent.jsonl")

    def test_visualize_extractions_invalid_jsonl_format(self, tmp_path: Path) -> None:
        """Test visualize_extractions raises ValueError for invalid JSONL format."""
        invalid_jsonl = tmp_path / "invalid.jsonl"
        invalid_jsonl.write_text("not valid json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            visualize_extractions(invalid_jsonl)

    def test_visualize_extractions_empty_html_content(self, tmp_path: Path) -> None:
        """Test visualize_extractions raises ValueError for empty HTML content."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.return_value = ""

            with pytest.raises(ValueError, match="empty HTML content"):
                visualize_extractions(jsonl_file)

    def test_visualize_extractions_visualization_failure(self, tmp_path: Path) -> None:
        """Test visualize_extractions handles visualization failures."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Test document",
                document_id="test_doc_1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        with patch("langextract.visualize") as mock_visualize:
            mock_visualize.side_effect = ValueError("Visualization error")

            with pytest.raises(ValueError, match="Visualization failed"):
                visualize_extractions(jsonl_file)


class TestJSONLFormatValidation:
    """Test JSONL format validation."""

    def test_validate_jsonl_format_valid_file(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format with valid JSONL file."""
        jsonl_file = tmp_path / "valid.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[
                    ExtractionRecord(
                        extraction_class="character",
                        extraction_text="Lady Juliet",
                        char_interval={"start_pos": 0, "end_pos": 11},
                        alignment_status="match_exact",
                        extraction_index=1,
                        group_index=0,
                    )
                ],
                text="Lady Juliet was a character.",
                document_id="doc1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        _validate_jsonl_format(jsonl_file)

    def test_validate_jsonl_format_file_not_found(self) -> None:
        """Test _validate_jsonl_format raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            _validate_jsonl_format(Path("nonexistent.jsonl"))

    def test_validate_jsonl_format_invalid_json(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format raises ValueError for invalid JSON."""
        invalid_file = tmp_path / "invalid.jsonl"
        invalid_file.write_text("not valid json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            _validate_jsonl_format(invalid_file)

    def test_validate_jsonl_format_missing_required_fields(
        self, tmp_path: Path
    ) -> None:
        """Test _validate_jsonl_format raises ValueError for missing required fields."""
        invalid_file = tmp_path / "missing_fields.jsonl"
        invalid_file.write_text(
            '{"text": "test"}\n'
        )  # Missing extractions and document_id

        with pytest.raises(ValueError, match="Missing required fields"):
            _validate_jsonl_format(invalid_file)

    def test_validate_jsonl_format_wrong_field_types(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format raises TypeError for wrong field types."""
        invalid_file = tmp_path / "wrong_types.jsonl"
        invalid_file.write_text(
            '{"extractions": "not a list", "text": "test", "document_id": "doc1"}\n'
        )

        with pytest.raises(TypeError, match="must be a list"):
            _validate_jsonl_format(invalid_file)

    def test_validate_jsonl_format_empty_file(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format handles empty file gracefully."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        _validate_jsonl_format(empty_file)

    def test_validate_jsonl_format_multiple_lines(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format validates multiple lines correctly."""
        jsonl_file = tmp_path / "multi.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Document 1",
                document_id="doc1",
            ),
            JSONLLine(
                extractions=[],
                text="Document 2",
                document_id="doc2",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)

        _validate_jsonl_format(jsonl_file)

    def test_validate_jsonl_format_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _validate_jsonl_format skips empty lines."""
        jsonl_file = tmp_path / "with_empty.jsonl"
        jsonl_lines = [
            JSONLLine(
                extractions=[],
                text="Document 1",
                document_id="doc1",
            ),
        ]
        write_jsonl(jsonl_lines, jsonl_file)
        with jsonl_file.open("a") as f:
            f.write("\n\n")

        _validate_jsonl_format(jsonl_file)
