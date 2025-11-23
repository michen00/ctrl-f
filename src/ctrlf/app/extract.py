"""Field extraction module using langextract."""

from __future__ import annotations

__all__ = ("MIN_SNIPPET_LENGTH", "run_extraction")

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

from langextract import extract

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import (
    Candidate,
    ExtractionResult,
    FieldResult,
    PrePromptInstrumentation,
    SourceRef,
)
from ctrlf.app.schema_io import _is_union_type

if TYPE_CHECKING:
    from langextract.data import AnnotatedDocument
    from pydantic import BaseModel

    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)

MIN_SNIPPET_LENGTH = 7
"""Minimum length for extracted snippets (in characters)"""


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
        # Increase context and re-extract
        expanded_context = context * 2
        snippet = markdown[
            max(0, start - expanded_context) : min(
                len(markdown), end + expanded_context
            )
        ]
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
) -> tuple[str, str, PrePromptInstrumentation]:
    """Setup extraction by creating prompt description with embedded schema.

    Args:
        model: Extended Pydantic model

    Returns:
        Tuple of (schema_str, prompt_description, instrumentation)
    """
    # Use json_schema string as schema representation for stability
    schema_str = json.dumps(model.model_json_schema(), indent=2)

    # Embed schema in prompt description
    prompt_description = (
        f"Extract structured data based on the following schema:\n\n{schema_str}"
    )

    # No pre-prompt interactions needed with Ollama
    instrumentation = PrePromptInstrumentation(interactions=[])

    return schema_str, prompt_description, instrumentation


def _process_document(
    doc: CorpusDocument,
    prompt_description: str,
) -> list[tuple[str, Candidate]]:
    """Process a single document and extract candidates.

    Returns:
        List of (field_name, Candidate) tuples
    """
    extracted_candidates: list[tuple[str, Candidate]] = []

    try:
        # Extract all fields at once using Ollama
        # Try omitting examples first, fallback to empty list if needed
        try:
            result = extract(
                text_or_documents=doc.markdown,
                prompt_description=prompt_description,
                model_id="gemma2:2b",
                model_url="http://localhost:11434",
                fence_output=False,
                use_schema_constraints=False,
            )
        except TypeError:
            # If examples parameter is required, try with empty list
            result = extract(
                text_or_documents=doc.markdown,
                prompt_description=prompt_description,
                examples=[],
                model_id="gemma2:2b",
                model_url="http://localhost:11434",
                fence_output=False,
                use_schema_constraints=False,
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

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Extraction/Visualization failed for document %s: %s", doc.doc_id, e
        )

    return extracted_candidates


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

    # 1. Setup Phase: Create prompt with embedded schema
    schema_str, prompt_description, instrumentation = _setup_extraction(model)
    schema_version = hashlib.md5(schema_str.encode(), usedforsecurity=False).hexdigest()

    # Collect all candidates per field
    field_candidates: dict[str, list[Candidate]] = {}

    # 2. Extraction Phase: Batch process documents
    for doc in corpus_docs:
        candidates = _process_document(doc, prompt_description)
        for field_name, candidate in candidates:
            if field_name not in field_candidates:
                field_candidates[field_name] = []
            field_candidates[field_name].append(candidate)

    # 3. Aggregation Phase
    extraction_result = _aggregate_final_results(
        model, field_candidates, schema_version, run_id
    )
    return extraction_result, instrumentation
