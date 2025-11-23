"""Structured extraction module using OpenAI/Gemini structured outputs.

This module provides an alternative extraction approach that:
1. Uses structured outputs from OpenAI/Gemini with JSON Schema
2. Processes each document individually with the schema
3. Uses fuzzy regex to find character positions in documents
4. Outputs results in JSONL format for visualization

This is a draft implementation that doesn't interfere with existing extraction logic.

Example usage:

    from ctrlf.app.structured_extract import (
        run_structured_extraction,
        write_jsonl,
        visualize_extractions,
    )
    from ctrlf.app.ingest import process_corpus
    from ctrlf.app.schema_io import convert_json_schema_to_pydantic

    # Load schema and corpus
    schema_json = '{"type": "object", "properties": {"character": {"type": "string"}}}'
    schema = convert_json_schema_to_pydantic(json.loads(schema_json))
    corpus_docs = process_corpus("path/to/corpus")

    # Run structured extraction
    jsonl_lines = run_structured_extraction(
        schema=schema,
        corpus_docs=corpus_docs,
        provider="openai",
        model="gpt-4o",
    )

    # Write JSONL file
    write_jsonl(jsonl_lines, "extraction_results.jsonl")

    # Visualize
    html_content = visualize_extractions(
        "extraction_results.jsonl",
        output_html_path="visualization.html",
    )
"""

from __future__ import annotations

__all__ = (
    "find_char_interval",
    "run_structured_extraction",
    "visualize_extractions",
    "write_jsonl",
)

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from thefuzz import fuzz

from ctrlf.app.logging_conf import get_logger

if TYPE_CHECKING:
    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)


class ExtractionRecord(BaseModel):
    """Represents a single extraction with character interval information.

    Attributes:
        extraction_class: The class/type of extraction (field name)
        extraction_text: The extracted text value
        char_interval: Character position interval in the document
        alignment_status: Status of alignment (match_exact, match_fuzzy, no_match)
        extraction_index: Index of this extraction in the sequence
        group_index: Group index for related extractions
        description: Optional description
        attributes: Optional additional attributes
    """

    extraction_class: str
    extraction_text: str
    char_interval: dict[str, int]  # {"start_pos": int, "end_pos": int}
    alignment_status: str
    extraction_index: int
    group_index: int
    description: str | None = None
    attributes: dict[str, Any] | None = None


class JSONLLine(BaseModel):
    """Represents a single line in the JSONL output.

    Attributes:
        extractions: List of extractions for this document
        text: The full document text
        document_id: Unique identifier for the document
    """

    extractions: list[ExtractionRecord]
    text: str
    document_id: str


def find_char_interval(
    text: str,
    extraction_text: str,
    fuzzy_threshold: int = 80,
) -> tuple[dict[str, int], str]:
    """Find character interval for an extraction using fuzzy regex matching.

    Args:
        text: Full document text to search in
        extraction_text: The extracted text to locate
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching

    Returns:
        Tuple of (char_interval dict, alignment_status)
        char_interval: {"start_pos": int, "end_pos": int}
        alignment_status: "match_exact", "match_fuzzy", or "no_match"
    """
    # First try exact match
    exact_match = text.find(extraction_text)
    if exact_match != -1:
        return (
            {"start_pos": exact_match, "end_pos": exact_match + len(extraction_text)},
            "match_exact",
        )

    # Try case-insensitive exact match
    text_lower = text.lower()
    extraction_lower = extraction_text.lower()
    case_insensitive_match = text_lower.find(extraction_lower)
    if case_insensitive_match != -1:
        return (
            {
                "start_pos": case_insensitive_match,
                "end_pos": case_insensitive_match + len(extraction_text),
            },
            "match_exact",
        )

    # Try fuzzy matching using sliding window
    extraction_len = len(extraction_text)
    best_match_pos = -1
    best_match_score = 0
    window_size = min(extraction_len * 2, len(text))

    # Slide window through text
    for i in range(len(text) - window_size + 1):
        window = text[i : i + window_size]
        score = fuzz.partial_ratio(extraction_text, window)
        if score > best_match_score:
            best_match_score = score
            best_match_pos = i

    # If we found a good fuzzy match, try to find exact boundaries
    if best_match_score >= fuzzy_threshold and best_match_pos != -1:
        # Find the best substring match within the window
        window = text[best_match_pos : best_match_pos + window_size]
        # Use token-based matching to find better boundaries
        extraction_tokens = extraction_text.split()
        window_tokens = window.split()

        # Find best token alignment
        best_token_start = 0
        best_token_score = 0
        for i in range(len(window_tokens) - len(extraction_tokens) + 1):
            window_subset = " ".join(window_tokens[i : i + len(extraction_tokens)])
            score = fuzz.ratio(extraction_text, window_subset)
            if score > best_token_score:
                best_token_score = score
                best_token_start = i

        # Calculate character positions from token positions
        token_start_char = len(" ".join(window_tokens[:best_token_start]))
        if best_token_start > 0:
            token_start_char += 1  # Account for space before first token

        matched_text = " ".join(
            window_tokens[best_token_start : best_token_start + len(extraction_tokens)]
        )
        start_pos = best_match_pos + token_start_char
        end_pos = start_pos + len(matched_text)

        return (
            {"start_pos": start_pos, "end_pos": end_pos},
            "match_fuzzy",
        )

    # No match found
    return ({"start_pos": 0, "end_pos": 0}, "no_match")


def _call_structured_extraction_api(
    text: str,
    schema: dict[str, Any],
    provider: str = "openai",
    model: str | None = None,
) -> dict[str, Any]:
    """Call OpenAI or Gemini API with structured outputs.

    This is a placeholder implementation. In practice, you would:
    - For OpenAI: Use openai client with response_format containing
      {"type": "json_schema", "json_schema": {"schema": schema}}
    - For Gemini: Use google-genai client with response_schema

    Args:
        text: Document text to extract from
        schema: JSON Schema for structured output
        provider: API provider ("openai" or "gemini")
        model: Model name (e.g., "gpt-4o", "gemini-2.0-flash-exp")

    Returns:
        Extracted data as dict matching the schema

    Raises:
        NotImplementedError: This is a draft implementation
    """
    # NOTE: Implement actual API calls when ready
    # For OpenAI:
    #   from openai import OpenAI
    #   client = OpenAI()
    #   response = client.chat.completions.create(
    #       model=model or "gpt-4o",
    #       messages=[{"role": "user", "content": text}],
    #       response_format={
    #           "type": "json_schema",
    #           "json_schema": {"schema": schema, "strict": True}
    #       }
    #   )
    #   return json.loads(response.choices[0].message.content)
    #
    # For Gemini:
    #   import google.generativeai as genai
    #   genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #   model = genai.GenerativeModel(
    #       model_name=model or "gemini-2.0-flash-exp",
    #       generation_config={
    #           "response_mime_type": "application/json",
    #           "response_schema": schema
    #       }
    #   )
    #   response = model.generate_content(text)
    #   return json.loads(response.text)

    msg = "Structured extraction API calls not yet implemented"
    raise NotImplementedError(msg)


def _flatten_extractions(
    data: dict[str, Any],
    schema: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Flatten extracted data into (field_name, value, attributes) tuples.

    Handles nested objects and arrays according to the schema.

    Args:
        data: Extracted data dict
        schema: JSON Schema definition
        prefix: Prefix for nested field names

    Returns:
        List of (field_name, value, attributes) tuples
    """
    extractions: list[tuple[str, str, dict[str, Any] | None]] = []

    if "properties" not in schema:
        return extractions

    for field_name, field_schema in schema["properties"].items():
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name
        field_value = data.get(field_name)

        if field_value is None:
            continue

        field_type = field_schema.get("type")

        if field_type == "array":
            # Handle array of values
            items_schema = field_schema.get("items", {})
            if isinstance(field_value, list):
                for idx, item in enumerate(field_value):
                    if isinstance(item, str):
                        extractions.append((full_field_name, item, {"index": idx}))
                    elif isinstance(item, dict):
                        # Nested object in array
                        nested = _flatten_extractions(
                            item, items_schema, f"{full_field_name}[{idx}]"
                        )
                        extractions.extend(nested)
        elif field_type == "object":
            # Handle nested object
            nested = _flatten_extractions(field_value, field_schema, full_field_name)
            extractions.extend(nested)
        elif isinstance(field_value, str):
            # Primitive string value
            extractions.append((full_field_name, field_value, None))
        elif field_value is not None:
            # Convert non-string primitives to string
            extractions.append((full_field_name, str(field_value), None))

    return extractions


def run_structured_extraction(
    schema: dict[str, Any] | type[BaseModel],
    corpus_docs: list[CorpusDocument],
    provider: str = "openai",
    model: str | None = None,
    fuzzy_threshold: int = 80,
) -> list[JSONLLine]:
    """Run structured extraction on corpus documents.

    Args:
        schema: JSON Schema dict or Pydantic model
        corpus_docs: List of corpus documents
        provider: API provider ("openai" or "gemini")
        model: Model name (optional, uses provider default)
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)

    Returns:
        List of JSONLLine objects, one per document

    Raises:
        NotImplementedError: If structured extraction API is not implemented
    """
    # Convert Pydantic model to JSON Schema if needed
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.model_json_schema()
    else:
        schema_dict = schema

    jsonl_lines: list[JSONLLine] = []

    for doc in corpus_docs:
        try:
            # Call structured extraction API
            extracted_data = _call_structured_extraction_api(
                doc.markdown, schema_dict, provider=provider, model=model
            )

            # Flatten extractions
            flat_extractions = _flatten_extractions(extracted_data, schema_dict)

            # Find character intervals and create ExtractionRecord objects
            extraction_records: list[ExtractionRecord] = []
            for idx, (field_name, value, attributes) in enumerate(flat_extractions):
                char_interval, alignment_status = find_char_interval(
                    doc.markdown, value, fuzzy_threshold=fuzzy_threshold
                )

                extraction_record = ExtractionRecord(
                    extraction_class=field_name,
                    extraction_text=value,
                    char_interval=char_interval,
                    alignment_status=alignment_status,
                    extraction_index=idx + 1,
                    group_index=idx,  # Simple grouping - can be enhanced
                    description=None,
                    attributes=attributes,
                )
                extraction_records.append(extraction_record)

            # Create JSONL line
            jsonl_line = JSONLLine(
                extractions=extraction_records,
                text=doc.markdown,
                document_id=doc.doc_id,
            )
            jsonl_lines.append(jsonl_line)

        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Structured extraction failed for document %s: %s", doc.doc_id, e
            )
            # Create empty extraction record for failed documents
            jsonl_line = JSONLLine(
                extractions=[],
                text=doc.markdown,
                document_id=doc.doc_id,
            )
            jsonl_lines.append(jsonl_line)

    return jsonl_lines


def write_jsonl(
    jsonl_lines: list[JSONLLine],
    output_path: str | Path,
) -> None:
    """Write JSONL lines to file.

    Args:
        jsonl_lines: List of JSONLLine objects
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for line in jsonl_lines:
            # Convert to dict and write as JSON line
            line_dict = line.model_dump()
            f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

    logger.info("Wrote %d lines to %s", len(jsonl_lines), output_path)


def visualize_extractions(
    jsonl_path: str | Path,
    output_html_path: str | Path | None = None,
) -> str:
    """Visualize extractions from JSONL file using langextract.

    Args:
        jsonl_path: Path to input JSONL file
        output_html_path: Optional path to save HTML visualization
            (if None, returns HTML as string)

    Returns:
        HTML content as string

    Raises:
        ImportError: If langextract is not available
        FileNotFoundError: If jsonl_path doesn't exist
    """
    try:
        import langextract as lx  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as e:
        msg = "langextract is required for visualization"
        raise ImportError(msg) from e

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        msg = f"JSONL file not found: {jsonl_path}"
        raise FileNotFoundError(msg)

    # Use langextract's visualize function
    html_content = lx.visualize(str(jsonl_path))

    # Handle different return types from langextract
    html_str = html_content.data if hasattr(html_content, "data") else str(html_content)

    # Write to file if output path provided
    if output_html_path:
        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html_str)
        logger.info("Wrote visualization to %s", output_path)

    return html_str
