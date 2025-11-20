"""Field extraction module using langextract."""

from __future__ import annotations

__all__ = "extract_field_candidates", "run_extraction"

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin

from langextract import extract

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import Candidate, ExtractionResult, FieldResult, SourceRef

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)


def _create_source_ref(
    doc_id: str,
    path: str,
    location: str,
    snippet: str,
    meta: dict[str, Any] | None = None,
) -> SourceRef:
    """Create a SourceRef from extraction span information.

    Args:
        doc_id: Document identifier
        path: File path
        location: Location descriptor
        snippet: Context snippet
        meta: Optional metadata

    Returns:
        SourceRef instance
    """
    return SourceRef(
        doc_id=doc_id,
        path=path,
        location=location,
        snippet=snippet,
        meta=meta or {},
    )


def _extract_snippet(markdown: str, start: int, end: int, context: int = 50) -> str:
    """Extract a snippet of text around a span.

    Args:
        markdown: Full markdown content
        start: Start position
        end: End position
        context: Number of characters of context on each side

    Returns:
        Snippet string (guaranteed to be at least 10 characters)
    """
    snippet_start = max(0, start - context)
    snippet_end = min(len(markdown), end + context)
    snippet = markdown[snippet_start:snippet_end]
    # Ensure snippet is at least 10 characters
    if len(snippet) < 10:
        snippet = markdown[max(0, start - 5) : min(len(markdown), end + 5)]
        # If still too short, use entire document or pad if necessary
        if len(snippet) < 10:
            if len(markdown) >= 10:
                # Use entire document if it's long enough
                snippet = markdown
            else:
                # Pad with spaces to reach minimum length
                snippet = markdown + " " * (10 - len(markdown))
    return snippet


def extract_field_candidates(  # noqa: PLR0913
    field_name: str,
    field_type: type,
    field_description: str | None,
    markdown_content: str,
    doc_id: str,
    source_map: dict[str, Any],
) -> list[Candidate]:
    """Extract candidate values for a single field from a single document.

    Args:
        field_name: Name of schema field
        field_type: Python type of field (str, int, float, date, etc.)
        field_description: Optional field description from schema
        markdown_content: Document content in Markdown
        doc_id: Document identifier
        source_map: Source mapping for span location

    Returns:
        List of candidate values with confidence scores and source references
    """
    candidates: list[Candidate] = []

    try:
        # Build extraction query from field info
        if field_description:
            pass

        # Use langextract to extract candidates
        # Note: This is a simplified implementation
        # Real langextract API may differ
        extracted = extract(
            text_or_documents=markdown_content,
            field_name=field_name,
            field_type=field_type.__name__
            if hasattr(field_type, "__name__")
            else "str",
            description=field_description,
        )

        # Process each extracted result
        for result in extracted:
            value = result.get("value")
            confidence = result.get("confidence", 0.5)
            span_start = result.get("span_start", 0)
            span_end = result.get("span_end", 0)

            # Create location descriptor
            location = source_map.get(
                "location", f"char-range [{span_start}:{span_end}]"
            )
            if "location" not in source_map:
                # Try to build from source_map
                if "pages" in source_map:
                    location = _extract_location_from_source_map(
                        source_map, span_start, span_end
                    )
                else:
                    location = f"char-range [{span_start}:{span_end}]"

            # Extract snippet
            snippet = _extract_snippet(markdown_content, span_start, span_end)

            # Create source reference
            source_ref = _create_source_ref(
                doc_id=doc_id,
                path=source_map.get(
                    "file_path", source_map.get("file_name", "unknown")
                ),
                location=location,
                snippet=snippet,
                meta={
                    "span_start": span_start,
                    "span_end": span_end,
                    **source_map,
                },
            )

            # Create candidate (normalization happens later in aggregate_field_results)
            candidate = Candidate(
                value=value,
                normalized=None,
                confidence=float(confidence),
                sources=[source_ref],
            )
            candidates.append(candidate)

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "field_extraction_failed",
            field_name=field_name,
            doc_id=doc_id,
            error=str(e),
        )

    return candidates


def _extract_location_from_source_map(
    source_map: dict[str, Any], span_start: int, span_end: int
) -> str:
    """Extract location from source map.

    Args:
        source_map: Source mapping dictionary
        span_start: Start position
        span_end: End position

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
    if origin is Union or origin is type(None):
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
            if inner_origin is Union or inner_origin is type(None):
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


def run_extraction(
    model: type[BaseModel],
    corpus_docs: list[CorpusDocument],
) -> ExtractionResult:
    """Run extraction for all fields across all documents.

    Args:
        model: Extended Pydantic model (all fields as arrays)
        corpus_docs: List of CorpusDocument instances

    Returns:
        Complete extraction results with all field results
    """
    run_id = str(uuid.uuid4())
    schema_version = str(hash(str(model.model_json_schema())))

    # Collect all candidates per field
    field_candidates: dict[str, list[Candidate]] = {}

    # Get field information from model
    field_info = model.model_fields

    # Extract candidates for each field from each document
    for doc in corpus_docs:
        for field_name, field_info_obj in field_info.items():
            # Get field type (should be List[type] or Optional[List[type]]
            # in Extended Schema)
            field_type = field_info_obj.annotation
            # Extract inner type from Extended Schema pattern
            inner_type = _extract_inner_type_from_extended_schema(field_type)

            field_description = field_info_obj.description

            # Extract candidates for this field
            candidates = extract_field_candidates(
                field_name=field_name,
                field_type=inner_type,
                field_description=field_description,
                markdown_content=doc.markdown,
                doc_id=doc.doc_id,
                source_map=doc.source_map,
            )

            # Add to field candidates
            if field_name not in field_candidates:
                field_candidates[field_name] = []
            field_candidates[field_name].extend(candidates)

    # Aggregate field results (will be done in aggregate module)
    # Import here to avoid circular dependency
    from ctrlf.app.aggregate import aggregate_field_results  # noqa: PLC0415

    field_results: list[FieldResult] = []
    for field_name, candidates in field_candidates.items():
        # Get field type from model for proper normalization
        field_info_obj = field_info[field_name]
        field_type = field_info_obj.annotation
        # Extract inner type from Extended Schema pattern
        # (handles both list[T] and list[T] | None)
        inner_type = _extract_inner_type_from_extended_schema(field_type)

        field_result = aggregate_field_results(
            field_name,
            candidates,
            field_type=inner_type,
        )
        field_results.append(field_result)

    # Create extraction result
    return ExtractionResult(
        results=field_results,
        schema_version=schema_version,
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
    )
