"""Field extraction module using langextract."""

from __future__ import annotations

__all__ = ("MIN_SNIPPET_LENGTH", "run_extraction")

import hashlib
import json
import tempfile
import types
import uuid
import webbrowser
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin

from google import genai
from langextract import extract, visualize
from langextract import io as lx_io
from langextract.data import AnnotatedDocument, ExampleData, Extraction

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import (
    Candidate,
    ExtractionResult,
    FieldResult,
    PrePromptInstrumentation,
    PrePromptInteraction,
    SourceRef,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)

MIN_SNIPPET_LENGTH = 7
"""Minimum length for extracted snippets (in characters)"""


def _is_union_type(origin: Any) -> bool:  # noqa: ANN401
    """Check if origin is a union type (handles both typing.Union and | syntax).

    Args:
        origin: The origin returned by typing.get_origin()

    Returns:
        True if origin represents a union type, False otherwise
    """
    return origin is Union or origin is types.UnionType


def _create_source_ref(
    doc_id: str,
    path: str,
    location: str,
    snippet: str,
    metadata: dict[str, Any] | None = None,
) -> SourceRef:
    """Create a SourceRef from extraction span information.

    Returns:
        SourceRef instance
    """
    return SourceRef(
        doc_id=doc_id,
        path=path,
        location=location,
        snippet=snippet,
        metadata=metadata or {},
    )


def _extract_snippet(
    markdown: str,
    start: int,
    end: int,
    context: int = 50,
    min_length: int = MIN_SNIPPET_LENGTH,
) -> str:
    """Extract a snippet of text around a span.

    Args:
        markdown: Full markdown content
        start: Start position
        end: End position
        context: Number of characters of context on each side
        min_length: Minimum length of the snippet (defaults to MIN_SNIPPET_LENGTH)

    Returns:
        Snippet string (guaranteed to be at least min_length characters)
    """
    snippet_start = max(0, start - context)
    snippet_end = min(len(markdown), end + context)
    snippet = markdown[snippet_start:snippet_end]
    # Ensure snippet is at least min_length characters
    if len(snippet) < min_length:
        snippet = markdown[max(0, start - context) : min(len(markdown), end + context)]
        # If still too short, use entire document or pad if necessary
        if len(snippet) < min_length:
            snippet = markdown + " " * (min_length - len(markdown))
    return snippet


def _extract_location_from_source_map(
    source_map: dict[str, Any], span_start: int, span_end: int
) -> str:
    """Extract location from source map.

    Returns:
        Location descriptor
    """
    if "pages" in source_map:
        pages = source_map["pages"]
        if isinstance(pages, dict):
            for page_num, page_info in pages.items():
                if isinstance(page_info, dict):
                    page_start = page_info.get("start", 0)
                    page_end = page_info.get("end", float("inf"))
                    if page_start <= span_start <= page_end:
                        return f"page {page_num}"
    return f"char-range [{span_start}:{span_end}]"


def _extract_response_metadata(
    response: Any,  # noqa: ANN401
) -> tuple[int | None, int | None, str | None, dict[str, object]]:
    """Extract metadata from a Google GenAI response.

    Args:
        response: The response object from Google GenAI

    Returns:
        Tuple of (prompt_tokens, completion_tokens, finish_reason, response_metadata)
    """
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None
    response_metadata: dict[str, object] = {}

    # Try to extract token usage
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        if usage is not None:
            if hasattr(usage, "prompt_token_count"):
                prompt_tokens = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                completion_tokens = usage.candidates_token_count
            if hasattr(usage, "total_token_count"):
                response_metadata["total_tokens"] = usage.total_token_count

    # Try to extract finish reason from candidates
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason"):
            finish_reason = str(candidate.finish_reason)
        if hasattr(candidate, "safety_ratings"):
            safety_ratings = candidate.safety_ratings
            if safety_ratings is not None:
                response_metadata["safety_ratings"] = [
                    {
                        "category": str(r.category) if hasattr(r, "category") else None,
                        "probability": str(r.probability)
                        if hasattr(r, "probability")
                        else None,
                    }
                    for r in safety_ratings
                ]

    return prompt_tokens, completion_tokens, finish_reason, response_metadata


def generate_synthetic_example(schema: str) -> tuple[str, PrePromptInteraction]:
    """Generate synthetic example text based on the schema using Google Gen AI.

    Args:
        schema: The schema definition (JSON Schema or Pydantic model)

    Returns:
        Tuple of (generated example text, instrumentation data)
    """
    client = genai.Client()
    model_name = "gemini-2.5-flash"

    prompt = f"""
fabricate brief example text from which structured data may be extracted by this schema:

{schema}

-=-=-=-=-=-=-

Example text:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    completion_text = response.text if response.text else ""

    # Extract metadata from response
    prompt_tokens, completion_tokens, finish_reason, response_metadata = (
        _extract_response_metadata(response)
    )

    instrumentation = PrePromptInteraction(
        step_name="generate_synthetic_example",
        prompt=prompt,
        completion=completion_text,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_metadata=response_metadata,
    )

    return completion_text, instrumentation


def generate_example_extractions(
    schema: str, example_text: str
) -> tuple[list[Extraction], PrePromptInteraction]:
    """Generate example extractions using Google Gen AI.

    Returns:
        Tuple of (list of langextract Extraction objects, instrumentation data)
    """
    client = genai.Client()
    model_name = "gemini-2.5-flash"

    prompt = f"""
Extract structured data from the input text according to the schema.
For each field in the schema, find the actual text snippet that appears in the input text.

IMPORTANT:
- The "text" field MUST be a simple string value that appears verbatim in the input text
- Do NOT create nested JSON structures or complex objects
- Do NOT make up values - only extract text that actually appears in the input
- Each extraction should be a simple key-value pair where "class" is the field name and "text" is the extracted value

Output format: One JSON object per line, each with "class" and "text" fields.
Do NOT output a single large JSON object.
Do NOT wrap in markdown code blocks.

Example:
{{"class": "field_name", "text": "actual text from input"}}
{{"class": "another_field", "text": "another actual value"}}

<input_text>
{example_text}
</input_text>

<schema>
{schema}
</schema>

Output:
"""  # noqa: E501

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    # Parse the output
    extractions: list[Extraction] = []
    response_text = response.text or ""

    # Extract metadata from response
    prompt_tokens, completion_tokens, finish_reason, response_metadata = (
        _extract_response_metadata(response)
    )

    if response_text:
        text_to_parse = response_text.strip()
        json_parsed_successfully = False

        # Try to parse as a single JSON object/array first (handling multi-line JSON)
        try:
            # Strip markdown code fences if present
            if text_to_parse.startswith("```"):
                lines = text_to_parse.splitlines()
                # Remove first line (```json or ```)
                if len(lines) > 1:
                    lines = lines[1:]
                # Remove last line if it is just ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_to_parse = "\n".join(lines)

            parsed_data = json.loads(text_to_parse)

            # If it's a list, assume it's a list of extraction objects
            if isinstance(parsed_data, list):
                extractions.extend(
                    Extraction(
                        extraction_class=item["class"],
                        extraction_text=str(item["text"]),
                    )
                    for item in parsed_data
                    if isinstance(item, dict) and "class" in item and "text" in item
                )
                json_parsed_successfully = True
            # If it's a dict, flatten it to extractions
            elif isinstance(parsed_data, dict):
                # For now, we skip complex JSON dict parsing as we enforce flat output
                # via prompt. This block catches valid JSON that isn't a flat list.
                pass

        except json.JSONDecodeError:
            # Not a single valid JSON object, likely line-delimited
            pass

        # Fallback to line-delimited processing (robust)
        # Only run if JSON array parsing didn't succeed to avoid duplicates
        if not json_parsed_successfully:
            for line_raw in response_text.strip().split("\n"):
                line = line_raw.strip()
                if not line:
                    continue
                if line.startswith("```"):
                    continue
                try:
                    data = json.loads(line)
                    # Only accept if it matches our expected format
                    if isinstance(data, dict) and "class" in data and "text" in data:
                        extractions.append(
                            Extraction(
                                extraction_class=data["class"],
                                extraction_text=str(data["text"]),
                            )
                        )
                except Exception as e:  # noqa: BLE001
                    # Only log if it looks like JSON but failed, or if we are debugging
                    if line.startswith("{"):
                        logger.warning(
                            "Failed to parse example extraction line: %s. Error: %s",
                            line,
                            e,
                        )

    instrumentation = PrePromptInteraction(
        step_name="generate_example_extractions",
        prompt=prompt,
        completion=response_text,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_metadata=response_metadata,
    )

    return extractions, instrumentation


def _extract_inner_type_from_extended_schema(field_type: object) -> type:
    """Extract the inner primitive type from Extended Schema field type.

    Handles Extended Schema patterns:
    - list[T] -> T
    - list[T] | None -> T (filters out None from Union first)
    - T -> T (fallback for non-list types)

    Args:
        field_type: Field type annotation from Extended Schema

    Returns:
        Inner primitive type (str, int, float, bool, etc.)
    """
    origin = get_origin(field_type)

    # Handle Union types (Optional) - filter out None first
    if _is_union_type(origin):
        args = get_args(field_type)
        # Filter out None type
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            # All args are None, fallback to str
            return str
        # Use the first non-None type
        field_type = non_none_args[0]
        origin = get_origin(field_type)

    # Handle List types - extract inner type
    if origin is list:
        args = get_args(field_type)
        if args:
            inner_type = args[0]
            # If inner type is still a generic (shouldn't happen in Extended Schema)
            # but handle it gracefully
            inner_origin = get_origin(inner_type)
            if _is_union_type(inner_origin):
                # Handle nested Optional in list (e.g., list[str | None])
                inner_args = get_args(inner_type)
                non_none_inner = [arg for arg in inner_args if arg is not type(None)]
                result_type = non_none_inner[0] if non_none_inner else str
                return cast("type", result_type)
            return cast("type", inner_type)
        # Empty list args, fallback to str
        return str

    # Not a list or Union - return as-is (should be primitive type)
    return field_type if isinstance(field_type, type) else str


def _setup_extraction(
    model: type[BaseModel],
) -> tuple[str, ExampleData, str, PrePromptInstrumentation]:
    """Setup extraction by generating synthetic examples and prompt description.

    Args:
        model: Extended Pydantic model

    Returns:
        Tuple of (schema_str, example_data, prompt_description, instrumentation)
    """
    # Use json_schema string as schema representation for stability
    schema_str = json.dumps(model.model_json_schema(), indent=2)

    try:
        example_text, interaction1 = generate_synthetic_example(schema_str)
        example_extractions, interaction2 = generate_example_extractions(
            schema_str, example_text
        )

        example_data = ExampleData(text=example_text, extractions=example_extractions)

        # Generate prompt description (can be simple or also generated)
        prompt_description = (
            f"Extract structured data based on the following schema: {schema_str}"
        )

        instrumentation = PrePromptInstrumentation(
            interactions=[interaction1, interaction2]
        )

    except Exception:
        logger.exception("Failed to generate synthetic examples")
        raise

    return schema_str, example_data, prompt_description, instrumentation


def _process_document(
    doc: CorpusDocument,
    prompt_description: str,
    example_data: ExampleData,
) -> list[tuple[str, Candidate]]:
    """Process a single document and extract candidates.

    Returns:
        List of (field_name, Candidate) tuples
    """
    extracted_candidates: list[tuple[str, Candidate]] = []

    try:
        # Extract all fields at once using the generated example
        result = extract(
            text_or_documents=doc.markdown,
            prompt_description=prompt_description,
            examples=[example_data],
        )

        annotated_doc: AnnotatedDocument | None
        if isinstance(result, list):
            annotated_doc = result[0] if result else None
        else:
            annotated_doc = result

        if annotated_doc and annotated_doc.extractions:
            for extraction in annotated_doc.extractions:
                # Map extraction to Candidate
                field_name = extraction.extraction_class
                value = extraction.extraction_text

                # Handle confidence if available
                confidence = getattr(extraction, "confidence", 1.0)

                # Grounding
                span_start = getattr(extraction, "char_start", 0)
                span_end = getattr(extraction, "char_end", span_start + len(value))

                # Create location descriptor
                location = _extract_location_from_source_map(
                    doc.source_map, span_start, span_end
                )

                # Extract snippet
                snippet = _extract_snippet(doc.markdown, span_start, span_end)

                source_ref = _create_source_ref(
                    doc_id=doc.doc_id,
                    path=doc.source_map.get(
                        "file_path", doc.source_map.get("file_name", "unknown")
                    ),
                    location=location,
                    snippet=snippet,
                    metadata={
                        "span_start": span_start,
                        "span_end": span_end,
                        **doc.source_map,
                    },
                )

                candidate = Candidate(
                    value=value,
                    normalized=None,
                    confidence=float(confidence),
                    sources=[source_ref],
                )
                extracted_candidates.append((field_name, candidate))

        # Visualization (Side Effect) - handled separately if needed
        # if annotated_doc and annotated_doc.extractions:
        #    _visualize_result(annotated_doc, doc.doc_id)

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Extraction/Visualization failed for document %s: %s", doc.doc_id, e
        )

    return extracted_candidates


def _visualize_result(annotated_doc: AnnotatedDocument, doc_id: str) -> None:
    """Visualize extraction result."""
    try:
        # Create a temporary file to save the visualization
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            doc_list = [annotated_doc]
            lx_io.save_annotated_documents(doc_list, tmp.name)  # type: ignore[arg-type]
            tmp_path = tmp.name

        # Generate visualization HTML
        html_content = visualize(tmp_path)

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        ) as html_tmp:
            html_tmp.write(html_content)
            html_path = html_tmp.name

        logger.info("Opening visualization at %s", html_path)
        webbrowser.open(f"file://{html_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Visualization failed for document %s: %s", doc_id, e)


def _aggregate_final_results(
    model: type[BaseModel],
    field_candidates: dict[str, list[Candidate]],
    schema_version: str,
    run_id: str,
) -> ExtractionResult:
    """Aggregate candidates into final ExtractionResult.

    Returns:
        ExtractionResult
    """
    # Import here to avoid circular dependency
    from ctrlf.app.aggregate import aggregate_field_results  # noqa: PLC0415

    field_results: list[FieldResult] = []
    field_info = model.model_fields

    for field_name, field_info_obj in field_info.items():
        candidates = field_candidates.get(field_name, [])
        field_type = field_info_obj.annotation
        inner_type = _extract_inner_type_from_extended_schema(field_type)

        field_result = aggregate_field_results(
            field_name,
            candidates,
            field_type=inner_type,
        )
        field_results.append(field_result)

    return ExtractionResult(
        results=field_results,
        schema_version=schema_version,
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
    )


def run_extraction(
    model: type[BaseModel],
    corpus_docs: list[CorpusDocument],
) -> tuple[ExtractionResult, PrePromptInstrumentation]:
    """Run extraction for all fields across all documents.

    Returns:
        Tuple of (complete extraction results, pre-prompt instrumentation)
    """
    run_id = str(uuid.uuid4())

    # 1. Setup Phase: Generate synthetic examples
    schema_str, example_data, prompt_description, instrumentation = _setup_extraction(
        model
    )
    schema_version = hashlib.md5(schema_str.encode(), usedforsecurity=False).hexdigest()

    # Collect all candidates per field
    field_candidates: dict[str, list[Candidate]] = {}

    # 2. Extraction Phase: Batch process documents
    for doc in corpus_docs:
        candidates = _process_document(doc, prompt_description, example_data)
        for field_name, candidate in candidates:
            if field_name not in field_candidates:
                field_candidates[field_name] = []
            field_candidates[field_name].append(candidate)

    # 3. Aggregation Phase
    extraction_result = _aggregate_final_results(
        model, field_candidates, schema_version, run_id
    )
    return extraction_result, instrumentation
