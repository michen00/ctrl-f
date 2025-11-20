# Data Model

**Feature**: Schema-Grounded Corpus Extractor
**Date**: 2025-11-12

## Core Entities

### SourceRef

Represents the exact location where a candidate value was found in a source document.

**Fields**:

- `doc_id: str` - Stable internal identifier for the source document (generated from file path/checksum)
- `path: str` - Original file path or filename
- `location: str` - Location descriptor (e.g., "page 3, line 120" or char-range "[3521:3630]")
- `snippet: str` - Small window of text around the extracted span (context for user viewing, minimum 7 characters)
- `metadata: Dict[str, Any]` - Additional metadata (mtime, converter used, checksum, etc.)

**Validation Rules**:

- `doc_id` must be non-empty
- `path` must be non-empty
- `location` must be non-empty (prefer page/line format, char-range as fallback)
- `snippet` must be at least 7 characters

**Relationships**:

- Referenced by `Candidate.sources` (many-to-many: one candidate can have multiple sources, one source can be referenced by multiple candidates)

### Candidate

Represents a potential value for a schema field extracted from the corpus.

**Fields**:

- `value: Any` - Raw extracted value (type depends on schema field type)
- `normalized: Any | None` - Optional canonicalized form (e.g., lowercase email, ISO date)
- `confidence: float` - Extractor's confidence score (0.0 to 1.0)
- `sources: List[SourceRef]` - List of source references where this value was found

**Validation Rules**:

- `confidence` must be in range [0.0, 1.0]
- `sources` must be non-empty (zero fabrication requirement - FR-004)
- `value` must match schema field type
- `normalized` must match schema field type if provided

**Relationships**:

- Contained in `FieldResult.candidates`
- May be selected as `FieldResult.consensus`

### FieldResult

Aggregation of all candidates for a single schema field after deduplication and clustering.

**Fields**:

- `field_name: str` - Name of the schema field
- `candidates: List[Candidate]` - List of candidate values (after deduplication, sorted by confidence)
- `consensus: Candidate | None` - Auto-suggested candidate if consensus detected (confidence ≥0.75 and margin ≥0.20)

**Validation Rules**:

- `field_name` must match a field in the schema
- `candidates` can be empty (no values found)
- `consensus` must be one of the candidates if present
- Candidates should be sorted by confidence (descending)

**Relationships**:

- Contained in `ExtractionResult.results` (one per schema field)
- Used to generate `Resolution` objects during user review

### ExtractionResult

Complete output from the extraction phase, containing results for all schema fields.

**Fields**:

- `results: List[FieldResult]` - Field results for all schema fields
- `schema_version: str` - Version identifier for the schema used (hash or user-provided)
- `run_id: str` - Unique identifier for this extraction run
- `created_at: str` - ISO 8601 timestamp of when extraction completed

**Validation Rules**:

- `results` must contain one FieldResult per schema field
- `run_id` must be unique (UUID or timestamp-based)
- `created_at` must be valid ISO 8601 format

**State Transitions**:

- Created after extraction completes
- Used as input to review UI
- Persisted temporarily during review session

### Resolution

User's decision for a single field during review.

**Fields**:

- `field_name: str` - Name of the schema field
- `chosen_value: Any` - User-selected value (from candidate or custom input)
- `source_doc_id: str | None` - Source document ID if value came from a candidate
- `source_location: str | None` - Source location if value came from a candidate
- `custom_input: bool` - Flag indicating if value was manually entered (not from extraction)

**Validation Rules**:

- `field_name` must match a field in the schema
- `chosen_value` must match schema field type
- If `custom_input` is True, `source_doc_id` and `source_location` must be None (enforced)
- If `custom_input` is False, `source_doc_id` and `source_location` may be None (warning only, not enforced)

**Relationships**:

- Multiple Resolution objects collected to create `PersistedRecord`

### PersistedRecord

Final saved record after user completes review and resolution.

**Fields**:

- `record_id: str` - Unique identifier for this record (slugified, timestamp-based, or UUID)
- `resolved: Dict[str, Any]` - Final validated record with field values (all arrays per Extended Schema)
- `provenance: Dict[str, List[SourceRef]]` - Source references per field (keyed by field name)
- `audit: Dict[str, Any]` - Audit trail containing:
  - `run_id: str` - Extraction run ID
  - `app_version: str` - Application version
  - `timestamp: str` - ISO 8601 timestamp of save
  - `user: str | None` - User identifier (None for single-user v0)
  - `config: Dict[str, Any]` - Configuration used (null policy, thresholds, etc.)

**Validation Rules**:

- `record_id` must be unique within TinyDB table
- `resolved` must validate against Extended Schema (all fields as arrays)
- `provenance` keys must match field names in `resolved`
- `audit.timestamp` must be valid ISO 8601 format

**Relationships**:

- Stored in TinyDB table (keyed by `record_id`)
- Can be exported to JSON (FR-018)

### PrePromptInteraction

Represents a single pre-prompt interaction before langextract.extract is called.

**Fields**:

- `step_name: str` - Name of the step (e.g., "generate_synthetic_example", "generate_example_extractions")
- `prompt: str` - The prompt sent to the LLM
- `completion: str` - The response/completion from the LLM
- `model: str` - The model used (e.g., "gemini-2.5-flash")

**Validation Rules**:

- `step_name` must be non-empty
- `prompt` must be non-empty
- `model` must be non-empty

**Relationships**:

- Contained in `PrePromptInstrumentation.interactions`

### PrePromptInstrumentation

Instrumentation data for pre-prompts before langextract.extract is called.

**Fields**:

- `interactions: List[PrePromptInteraction]` - List of pre-prompt interactions

**Validation Rules**:

- `interactions` can be empty (no pre-prompts executed)

**Relationships**:

- Returned alongside `ExtractionResult` from `run_extraction`
- Used for debugging and transparency of LLM interactions

## Schema Extension

### Extended Schema

The original user-provided schema (JSON Schema or Pydantic model) with all leaf fields coerced to arrays.

**Transformation Rules**:

- Primitive field `field_name: str` → `field_name: List[str]`
- Already array `field_name: List[int]` → `field_name: List[int]` (unchanged)
- Nested BaseModel objects → Recursively extended (all fields in nested model become arrays)
- Optional fields `field_name: str | None` → `field_name: List[str] | None`
- List of BaseModel `field_name: List[Address]` → `field_name: List[ExtendedAddress]` (nested model extended recursively)

**Purpose**:

- Uniform handling of single vs multiple values
- Supports multiple candidates from multiple sources
- Simplifies validation logic

**Examples**:

```python
# Example 1: Simple schema
# Original Schema
class Person(BaseModel):
    name: str
    email: str
    age: int

# Extended Schema
class ExtendedPerson(BaseModel):
    name: List[str]
    email: List[str]
    age: List[int]

# Example 2: Nested objects (supported)
# Original Schema
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

# Extended Schema
class ExtendedAddress(BaseModel):
    street: List[str]
    city: List[str]

class ExtendedPerson(BaseModel):
    name: List[str]
    address: List[ExtendedAddress]
```

## Data Flow

1. **Ingestion**: Corpus files → Markdown conversion → SourceRef creation
2. **Extraction Setup**: Schema → Synthetic example generation → Example extraction generation → PrePromptInstrumentation
3. **Extraction**: Markdown + Schema + Examples → Candidate generation → SourceRef attachment
4. **Aggregation**: Candidates → Deduplication → Normalization → Consensus detection → FieldResult
5. **Review**: FieldResult → User selection → Resolution
6. **Persistence**: Resolutions → Validation → PersistedRecord → TinyDB

## Validation Points

1. **Schema Input**: Validate JSON Schema format or Pydantic model syntax
2. **Schema Extension**: Recursively extend all fields (including nested BaseModel types) to arrays
3. **Candidate Creation**: Ensure all candidates have non-empty sources
4. **FieldResult**: Verify consensus candidate meets thresholds and is one of the candidates
5. **Resolution**: Validate chosen values against Extended Schema; ensure source fields are None when custom_input is True
6. **PersistedRecord**: Final validation before storage; validate audit timestamp format
