"""Structured extraction module using PydanticAI with Ollama/OpenAI/Gemini.

This module provides the primary extraction approach that:
1. Uses PydanticAI Agent with Pydantic models for structured outputs
2. Supports Ollama (default), OpenAI, and Gemini providers
3. Processes each document individually with the Extended Schema model
4. Uses fuzzy regex to find character positions in documents
5. Outputs results in JSONL format for visualization (via langextract.visualize())

Note: langextract is now only used for visualization, not for extraction.
PydanticAI unifies schema handling across all LLM providers.

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

    # Run structured extraction (Ollama is default)
    jsonl_lines = run_structured_extraction(
        schema=schema,
        corpus_docs=corpus_docs,
        provider="ollama",
        model="llama3",
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
    "FlattenedExtraction",
    "_call_structured_extraction_api",
    "_extraction_record_to_candidate",
    "find_char_interval",
    "run_structured_extraction",
    "visualize_extractions",
    "write_jsonl",
)

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel
from thefuzz import fuzz

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import Candidate, SourceRef

if TYPE_CHECKING:
    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)


class FlattenedExtraction(NamedTuple):
    """Represents a flattened extraction from structured data.

    Attributes:
        field_name: Full field name (may include prefix for nested fields)
        value: Extracted value as string
        attributes: Optional attributes dictionary (e.g., index for array items)
    """

    field_name: str
    value: str
    attributes: dict[str, Any] | None


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
        # Handle case where window has fewer tokens than extraction
        num_tokens_to_match = min(len(window_tokens), len(extraction_tokens))
        best_token_start = 0
        best_token_score = 0

        # Only iterate if we have enough tokens to slide
        if len(window_tokens) >= len(extraction_tokens):
            # Normal case: window has enough tokens, slide through
            for i in range(len(window_tokens) - len(extraction_tokens) + 1):
                window_subset = " ".join(window_tokens[i : i + len(extraction_tokens)])
                score = fuzz.ratio(extraction_text, window_subset)
                if score > best_token_score:
                    best_token_score = score
                    best_token_start = i
        else:
            # Edge case: window has fewer tokens than extraction
            # Use all available tokens and find best alignment
            window_subset = " ".join(window_tokens)
            best_token_score = fuzz.ratio(extraction_text, window_subset)
            best_token_start = 0

        # Calculate character positions from token positions
        token_start_char = len(" ".join(window_tokens[:best_token_start]))
        if best_token_start > 0:
            token_start_char += 1  # Account for space before first token

        # Use actual number of matched tokens (may be less than extraction_tokens)
        matched_text = " ".join(
            window_tokens[best_token_start : best_token_start + num_tokens_to_match]
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
    schema_model: type[BaseModel],
    provider: str = "ollama",
    model: str | None = None,
) -> dict[str, Any]:
    """Call LLM API with structured outputs using PydanticAI.

    Uses PydanticAI Agent to extract structured data from text according to the schema.
    Supports Ollama (default), OpenAI, and Gemini providers.

    Args:
        text: Document text to extract from
        schema_model: Pydantic model (Extended Schema) defining the output structure
        provider: API provider ("ollama", "openai", or "gemini", default: "ollama")
        model: Model name (e.g., "llama3", "gpt-4o", "gemini-2.5-flash")
            For Ollama, defaults to "llama3" if not specified
            For Gemini, defaults to "gemini-2.5-flash" if not specified

    Returns:
        Extracted data as dict matching the schema (validated Pydantic model instance)

    Raises:
        ValueError: If provider is not supported
        RuntimeError: If extraction fails
    """
    try:
        from pydantic_ai import Agent  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "pydantic-ai is required for structured extraction. "
            "Install with: uv add pydantic-ai (or pip install pydantic-ai). "
            "If using pydantic-ai-slim, ensure google extras are installed: "
            'pip install "pydantic-ai-slim[google]"'
        )
        raise ImportError(msg) from e

    # Determine model string based on provider
    # Format: "provider:model-name" where provider matches PydanticAI's provider
    # identifiers. Note: google-gla is the correct provider identifier (not
    # google-genai). See: https://ai.pydantic.dev/api/models/google/
    if provider == "ollama":
        model_str = f"ollama:{model or 'llama3'}"
    elif provider == "openai":
        model_str = f"openai:{model or 'gpt-4o'}"
    elif provider == "gemini":
        # google-gla is the PydanticAI provider identifier for Generative Language API
        # (package name is google-genai, but provider identifier is google-gla)
        model_str = f"google-gla:{model or 'gemini-2.5-flash'}"
    else:
        msg = f"Unsupported provider: {provider}. Supported: ollama, openai, gemini"
        raise ValueError(msg)

    try:
        # Create PydanticAI Agent with the Extended Schema as output_type
        agent = Agent(
            model_str,
            output_type=schema_model,
            system_prompt=(
                "Extract structured data from the provided document text. "
                "Return only the extracted data matching the specified schema. "
                "For array fields, return all matching values found in the document."
            ),
        )

        # Run extraction synchronously
        result = agent.run_sync(text)

        # Convert Pydantic model instance to dict
        extracted_data = result.output.model_dump()

        logger.debug(
            "Extracted data for document using %s: %s fields extracted",
            model_str,
            len(extracted_data),
        )
    except Exception as e:
        logger.exception(
            "Structured extraction failed with %s: %s", model_str, exc_info=e
        )
        msg = f"Extraction failed: {e}"
        raise RuntimeError(msg) from e
    else:
        return extracted_data


def _extraction_record_to_candidate(
    extraction_record: ExtractionRecord,
    doc_id: str,
    doc_markdown: str,
    source_map: dict[str, Any],
) -> Candidate:
    """Convert ExtractionRecord to Candidate format for existing pipeline.

    Args:
        extraction_record: ExtractionRecord from structured extraction
        doc_id: Document identifier
        doc_markdown: Full document markdown text
        source_map: Source mapping from document

    Returns:
        Candidate object compatible with existing extraction pipeline
    """
    # Map alignment_status to confidence score
    # match_exact = high confidence (0.9)
    # match_fuzzy = medium (0.7)
    # no_match = low (0.5)
    confidence_map = {
        "match_exact": 0.9,
        "match_fuzzy": 0.7,
        "no_match": 0.5,
    }
    confidence = confidence_map.get(extraction_record.alignment_status, 0.5)

    # Extract character positions
    start_pos = extraction_record.char_interval.get("start_pos", 0)
    end_pos = extraction_record.char_interval.get("end_pos", start_pos)

    # Create location descriptor
    location = f"char-range [{start_pos}:{end_pos}]"
    if "pages" in source_map:
        pages = source_map.get("pages", {})
        if isinstance(pages, dict):
            for page_num, page_info in pages.items():
                if isinstance(page_info, dict):
                    page_start = page_info.get("start", 0)
                    page_end = page_info.get("end", float("inf"))
                    if page_start <= start_pos <= page_end:
                        location = f"page {page_num}"
                        break

    # Extract snippet (similar to extract.py logic)
    context = 50
    snippet_start = max(0, start_pos - context)
    snippet_end = min(len(doc_markdown), end_pos + context)
    snippet = doc_markdown[snippet_start:snippet_end]
    if len(snippet) < 7:  # MIN_SNIPPET_LENGTH
        snippet = doc_markdown[:100] if len(doc_markdown) > 100 else doc_markdown

    # Create SourceRef
    source_ref = SourceRef(
        doc_id=doc_id,
        path=source_map.get("file_path", source_map.get("file_name", "unknown")),
        location=location,
        snippet=snippet,
        metadata={
            "span_start": start_pos,
            "span_end": end_pos,
            "alignment_status": extraction_record.alignment_status,
            **source_map,
        },
    )

    # Create Candidate
    return Candidate(
        value=extraction_record.extraction_text,
        normalized=None,  # Will be normalized in aggregate.py
        confidence=confidence,
        sources=[source_ref],
    )


def _flatten_extractions(
    data: dict[str, Any],
    schema: dict[str, Any],
    prefix: str = "",
) -> list[FlattenedExtraction]:
    """Flatten extracted data into FlattenedExtraction objects.

    Handles nested objects and arrays according to the schema.

    Args:
        data: Extracted data dict
        schema: JSON Schema definition
        prefix: Prefix for nested field names

    Returns:
        List of FlattenedExtraction objects
    """
    extractions: list[FlattenedExtraction] = []

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
                        extractions.append(
                            FlattenedExtraction(full_field_name, item, {"index": idx})
                        )
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
            extractions.append(FlattenedExtraction(full_field_name, field_value, None))
        elif field_value is not None:
            # Convert non-string primitives to string
            extractions.append(
                FlattenedExtraction(full_field_name, str(field_value), None)
            )

    return extractions


def run_structured_extraction(
    schema: type[BaseModel],
    corpus_docs: list[CorpusDocument],
    provider: str = "ollama",
    model: str | None = None,
    fuzzy_threshold: int = 80,
) -> list[JSONLLine]:
    """Run structured extraction on corpus documents using PydanticAI.

    Args:
        schema: Pydantic model class (Extended Schema) defining the output structure
        corpus_docs: List of corpus documents
        provider: API provider ("ollama", "openai", or "gemini", default: "ollama")
        model: Model name (optional, uses provider default)
            For Ollama: defaults to "llama3"
            For OpenAI: defaults to "gpt-4o"
            For Gemini: defaults to "gemini-2.5-flash"
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)

    Returns:
        List of JSONLLine objects, one per document

    Raises:
        ValueError: If schema is not a Pydantic model
        RuntimeError: If extraction fails
    """
    # Schema validation: function signature enforces type[BaseModel], so no runtime
    # check needed. The type system guarantees schema is a BaseModel subclass.

    # Get JSON Schema dict for flattening
    schema_dict = schema.model_json_schema()

    jsonl_lines: list[JSONLLine] = []

    for doc in corpus_docs:
        try:
            # Call structured extraction API with PydanticAI
            extracted_data = _call_structured_extraction_api(
                doc.markdown, schema, provider=provider, model=model
            )

            # Flatten extractions
            flat_extractions = _flatten_extractions(extracted_data, schema_dict)

            # Find character intervals and create ExtractionRecord objects
            extraction_records: list[ExtractionRecord] = []
            for idx, extraction in enumerate(flat_extractions):
                char_interval, alignment_status = find_char_interval(
                    doc.markdown, extraction.value, fuzzy_threshold=fuzzy_threshold
                )

                extraction_record = ExtractionRecord(
                    extraction_class=extraction.field_name,
                    extraction_text=extraction.value,
                    char_interval=char_interval,
                    alignment_status=alignment_status,
                    extraction_index=idx + 1,
                    group_index=idx,  # Simple grouping - can be enhanced
                    description=None,
                    attributes=extraction.attributes,
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
        import langextract as lx  # noqa: PLC0415
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
    # langextract may return a Jupyter display object with .data attribute or a string
    html_str = getattr(html_content, "data", None) or str(html_content)

    # Write to file if output path provided
    if output_html_path:
        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html_str)
        logger.info("Wrote visualization to %s", output_path)

    return html_str
