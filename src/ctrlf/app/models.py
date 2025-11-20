"""Pydantic data models for schema-grounded corpus extractor."""

from __future__ import annotations

__all__ = (
    "Candidate",
    "ExtractionResult",
    "FieldResult",
    "PersistedRecord",
    "PrePromptInstrumentation",
    "PrePromptInteraction",
    "Resolution",
    "SourceRef",
)

from datetime import datetime

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class SourceRef(BaseModel):
    """Represents the exact location where a candidate value was found.

    Attributes:
        doc_id: Stable internal identifier for the source document
        path: Original file path or filename
        location: Location descriptor (e.g., "page 3, line 120" or
            char-range "[3521:3630]")
        snippet: Small window of text around the extracted span
            (context for user viewing)
        meta: Additional metadata (mtime, converter used, checksum, etc.)
    """

    doc_id: str = Field(..., min_length=1, description="Document identifier")
    path: str = Field(..., min_length=1, description="Original file path")
    location: str = Field(..., min_length=1, description="Location descriptor")
    snippet: str = Field(
        ..., min_length=7, description="Context snippet around extracted span"
    )
    meta: dict[str, object] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Candidate(BaseModel):
    """Represents a potential value for a schema field extracted from the corpus.

    Attributes:
        value: Raw extracted value (type depends on schema field type)
        normalized: Optional canonicalized form (e.g., lowercase email, ISO date)
        confidence: Extractor's confidence score (0.0 to 1.0)
        sources: List of source references where this value was found
    """

    value: object = Field(..., description="Raw extracted value")
    normalized: object | None = Field(None, description="Normalized/canonicalized form")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: list[SourceRef] = Field(..., min_length=1, description="Source references")

    @field_validator("sources")
    @classmethod
    def validate_sources_non_empty(cls, v: list[SourceRef]) -> list[SourceRef]:
        """Ensure sources are non-empty (zero fabrication requirement)."""
        if not v:
            msg = "Sources must be non-empty (zero fabrication requirement)"
            raise ValueError(msg)
        return v


class FieldResult(BaseModel):
    """Aggregation of all candidates for a single schema field.

    Attributes:
        field_name: Name of the schema field
        candidates: List of candidate values (after deduplication, sorted by confidence)
        consensus: Auto-suggested candidate if consensus detected
    """

    field_name: str = Field(..., min_length=1, description="Schema field name")
    candidates: list[Candidate] = Field(
        default_factory=list, description="Candidate values (sorted by confidence)"
    )
    consensus: Candidate | None = Field(
        None, description="Consensus candidate if detected"
    )

    @field_validator("consensus")
    @classmethod
    def validate_consensus_in_candidates(
        cls, v: Candidate | None, info: ValidationInfo
    ) -> Candidate | None:
        """Ensure consensus candidate is one of the candidates."""
        if v is not None:
            candidates = info.data.get("candidates", [])
            if v not in candidates:
                msg = "Consensus candidate must be one of the candidates"
                raise ValueError(msg)
        return v


class ExtractionResult(BaseModel):
    """Complete output from the extraction phase.

    Attributes:
        results: Field results for all schema fields
        schema_version: Version identifier for the schema used (hash or user-provided)
        run_id: Unique identifier for this extraction run
        created_at: ISO 8601 timestamp of when extraction completed
    """

    results: list[FieldResult] = Field(..., description="Field results")
    schema_version: str = Field(..., min_length=1, description="Schema version")
    run_id: str = Field(..., min_length=1, description="Extraction run ID")
    created_at: str = Field(..., description="ISO 8601 timestamp")

    @field_validator("created_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v)
        except (ValueError, AttributeError) as e:
            msg = f"Invalid ISO 8601 timestamp: {v}"
            raise ValueError(msg) from e
        return v


class Resolution(BaseModel):
    """User's decision for a single field during review.

    Attributes:
        field_name: Name of the schema field
        chosen_value: User-selected value (from candidate or custom input)
        source_doc_id: Source document ID if value came from a candidate
        source_location: Source location if value came from a candidate
        custom_input: Flag indicating if value was manually entered
    """

    field_name: str = Field(..., min_length=1, description="Schema field name")
    chosen_value: object = Field(..., description="User-selected value")
    source_doc_id: str | None = Field(
        None, description="Source document ID if from candidate"
    )
    source_location: str | None = Field(
        None, description="Source location if from candidate"
    )
    custom_input: bool = Field(
        default=False, description="Whether value was manually entered"
    )

    @field_validator("source_doc_id", "source_location")
    @classmethod
    def validate_source_fields(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate source fields based on custom_input flag."""
        custom_input = info.data.get("custom_input", False)
        if not custom_input and v is None:
            # If not custom input, source should be provided (warning, not error)
            pass
        if custom_input and v is not None:
            msg = "Source fields should be None when custom_input is True"
            raise ValueError(msg)
        return v


class PersistedRecord(BaseModel):
    """Final saved record after user completes review and resolution.

    Attributes:
        record_id: Unique identifier for this record
        resolved: Final validated record with field values
            (all arrays per Extended Schema)
        provenance: Source references per field (keyed by field name)
        audit: Audit trail containing run_id, app_version, timestamp, user, config
    """

    record_id: str = Field(..., min_length=1, description="Unique record identifier")
    resolved: dict[str, object] = Field(..., description="Resolved field values")
    provenance: dict[str, list[SourceRef]] = Field(
        ..., description="Source references per field"
    )
    audit: dict[str, object] = Field(..., description="Audit trail")

    @field_validator("audit")
    @classmethod
    def validate_audit_timestamp(cls, v: dict[str, object]) -> dict[str, object]:
        """Validate audit timestamp is ISO 8601 format."""
        timestamp = v.get("timestamp")
        if timestamp:
            try:
                datetime.fromisoformat(str(timestamp))
            except (ValueError, AttributeError) as e:
                msg = f"Invalid ISO 8601 timestamp in audit: {timestamp}"
                raise ValueError(msg) from e
        return v


class PrePromptInteraction(BaseModel):
    """Represents a single pre-prompt interaction before langextract.extract is called.

    Attributes:
        step_name: Name of the step (e.g., "generate_synthetic_example",
            "generate_example_extractions")
        prompt: The prompt sent to the LLM
        completion: The response/completion from the LLM
        model: The model used (e.g., "gemini-2.5-flash")
    """

    step_name: str = Field(..., min_length=1, description="Name of the pre-prompt step")
    prompt: str = Field(..., min_length=1, description="Prompt sent to LLM")
    completion: str = Field(..., description="LLM response/completion")
    model: str = Field(..., min_length=1, description="Model used for this interaction")


class PrePromptInstrumentation(BaseModel):
    """Instrumentation data for pre-prompts before langextract.extract is called.

    Attributes:
        interactions: List of pre-prompt interactions
    """

    interactions: list[PrePromptInteraction] = Field(
        default_factory=list, description="List of pre-prompt interactions"
    )
