# Data Model

**Feature**: Structured Extraction with OpenAI/Gemini API Integration
**Date**: 2025-01-27

## Core Entities

### ExtractionRecord

Represents a single extraction with character interval information for JSONL output format.

**Fields**:

- `extraction_class: str` - The class/type of extraction (field name from schema)
- `extraction_text: str` - The extracted text value
- `char_interval: Dict[str, int]` - Character position interval in the document `{"start_pos": int, "end_pos": int}`
- `alignment_status: str` - Status of alignment: `"match_exact"`, `"match_fuzzy"`, or `"no_match"`
- `extraction_index: int` - Index of this extraction in the sequence (1-based, per-document)
- `group_index: int` - Group index for related extractions (for grouping related fields). Default value is 0 for the first extraction, increments with each extraction in the document (0-based index from extraction sequence)
- `description: str | None` - Optional description (currently unused, reserved for future use)
- `attributes: Dict[str, Any] | None` - Optional additional attributes (e.g., array index, nested field path)

**Validation Rules**:

- `extraction_class` must be non-empty
- `extraction_text` must be non-empty
- `char_interval` must contain `start_pos` and `end_pos` as integers
- `start_pos` must be >= 0
- `end_pos` must be > `start_pos` (or 0 if `alignment_status == "no_match"`)
- `alignment_status` must be one of: `"match_exact"`, `"match_fuzzy"`, `"no_match"`
- `extraction_index` must be >= 1
- `group_index` must be >= 0

**Relationships**:

- Contained in `JSONLLine.extractions` (many-to-one: multiple extractions per document)

**State Transitions**:

- Created after API response is parsed and character intervals are calculated
- Used to generate JSONL output
- Used for visualization via `langextract.visualize()`

### JSONLLine

Represents a single line in the JSONL output file, containing all extractions for one document.

**Fields**:

- `extractions: List[ExtractionRecord]` - List of extractions for this document
- `text: str` - The full document text (markdown format)
- `document_id: str` - Unique identifier for the document (from `CorpusDocument.doc_id`). Format: MD5 hexadecimal hash (32 lowercase hex characters) generated from the file path. Example: `"a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"`

**Validation Rules**:

- `extractions` can be empty (no extractions found)
- `text` must be non-empty (document content)
- `document_id` must be non-empty, match a valid document ID, and be a 32-character hexadecimal string (MD5 hash format)

**Relationships**:

- One per `CorpusDocument` (one-to-one)
- Used to generate JSONL file (one line per document)
- Used as input to `langextract.visualize()`

**State Transitions**:

- Created after processing a single document through structured extraction
- Written to JSONL file
- Used for visualization

## Supporting Data Structures

### API Configuration

Configuration for API calls (not a Pydantic model, but used throughout).

**Fields**:

- `provider: str` - API provider: `"openai"` or `"gemini"`
- `model: str` - Model name (e.g., `"gpt-4o"`, `"gemini-2.0-flash-exp"`)
- `api_key: str` - API key for authentication
- `max_retries: int` - Maximum number of retry attempts (default: 3)
- `timeout: float` - Request timeout in seconds (default: 60.0)
- `fuzzy_threshold: int` - Minimum similarity score for fuzzy matching (0-100, default: 80)

**Validation Rules**:

- `provider` must be `"openai"` or `"gemini"`
- `model` must be non-empty and valid for the provider
- `api_key` must be non-empty
- `max_retries` must be >= 0
- `timeout` must be > 0
- `fuzzy_threshold` must be in range [0, 100]

### Character Interval

Character position interval in document (embedded in `ExtractionRecord`).

**Fields**:

- `start_pos: int` - Start character position (0-based)
- `end_pos: int` - End character position (exclusive, 0-based)

**Validation Rules**:

- `start_pos` >= 0
- `end_pos` > `start_pos` (or both 0 if no match found)
- Both values must be within document text length

### Alignment Status

Status of character alignment for an extraction (enum-like string).

**Values**:

- `"match_exact"` - Exact match found in document text (case-sensitive or case-insensitive)
- `"match_fuzzy"` - Fuzzy match found (similarity >= threshold)
- `"no_match"` - No match found (char_interval will be {0, 0})

## Data Flow

1. **Input**: `CorpusDocument` (from `ingest.py`) + JSON Schema (from `schema_io.py`)
2. **Schema Conversion**: JSON Schema → Pydantic model (via `schema_io.convert_json_schema_to_pydantic()`) - converts JSON Schema draft-07 to Pydantic v2 model for API consumption
3. **API Call**: Document text (markdown) + Pydantic model (as `output_type`) → PydanticAI Agent → Provider API (Ollama/OpenAI/Gemini) → Structured Pydantic model instance response
4. **API Response Validation**: Pydantic model instance → Validate schema compliance (types match, required fields present, structure matches schema) - validation performed by PydanticAI at API level, but system MUST verify response is valid
5. **Response Processing**: Pydantic model instance → Convert to dict → Flatten nested structures (via `_flatten_extractions()`) → Flattened extractions (field_name, value, attributes)
6. **Character Alignment**: Extraction text + Document text → Fuzzy string matching (via `find_char_interval()`) → Character intervals + Alignment status (match_exact, match_fuzzy, no_match)
7. **Record Creation**: Flattened extractions + Character intervals + Alignment status → `ExtractionRecord` objects (one per extracted field)
8. **Line Creation**: `ExtractionRecord` list + Document text (markdown) + Document ID → `JSONLLine` object (one per document)
9. **Output**: `JSONLLine` objects → JSONL file (one line per document) → Visualization (via `langextract.visualize()`)

**API Response to JSONL Conversion Details**:

- API returns Pydantic model instance (structured dict matching schema)
- System converts Pydantic model to dict (if not already dict)
- System flattens nested structures using dot notation (e.g., "person.name", "items.0.value")
- System creates `ExtractionRecord` for each flattened field with character interval and alignment status
- System groups `ExtractionRecord` objects into `JSONLLine` with document text and document_id
- System writes `JSONLLine` objects to JSONL file (one JSON object per line)

## Integration with Existing Models

### Reused from Existing Codebase

- **`CorpusDocument`** (from `ingest.py`): Input document with `doc_id`, `markdown`, `source_map`
  - `doc_id`: Generated via MD5 hash of file path (32-character hex string) for stable, unique identifiers
- **Schema models** (from `schema_io.py`): JSON Schema or Pydantic models for extraction

### Relationship to Existing Extraction Models

This feature uses different models than the existing `extract.py` pipeline:

- **Existing**: `Candidate`, `FieldResult`, `ExtractionResult` (used by both extraction pipelines)
- **New**: `ExtractionRecord`, `JSONLLine` (for structured API extraction)

The two pipelines are independent and can coexist. The new pipeline focuses on JSONL output for visualization, while the existing pipeline focuses on candidate aggregation and consensus detection.

## Validation Points

1. **Before API Call**: (a) Schema validation (JSON Schema format, compatibility with API capabilities per NFR-004), (b) Schema-to-Pydantic conversion (via `schema_io.convert_json_schema_to_pydantic()`), (c) API key validation (presence, format, precedence order per FR-008), (d) Document text validation (non-empty string), (e) Provider selection validation (valid provider, model string format)
2. **After API Response**: (a) JSON parsing (if API returns raw JSON, parse to dict), (b) Schema compliance check (response matches Pydantic model structure, types match, required fields present), (c) Pydantic validation (response passes Pydantic model validation - enforced by PydanticAI but verified by system), (d) Response format validation (valid structured data, no parsing errors)
3. **After Flattening**: Extraction record validation (non-empty field_name and extraction_text, valid attributes dict, valid types)
4. **After Character Alignment**: Interval validation (start_pos >= 0, end_pos > start_pos, intervals within document bounds, alignment_status is valid value)
5. **Before JSONL Write**: `JSONLLine` validation (all required fields present: extractions array, text string, document_id string, valid document_id matching input document)
6. **Before Visualization**: JSONL file format validation (each line is valid JSON, format matches `langextract.visualize()` expectations)

**API Response Error Scenarios**:

- **Invalid JSON**: API returns malformed JSON → Parse error → Skip document, log error, create empty JSONLLine
- **Schema Mismatch**: API response doesn't match Pydantic model structure → Validation error → Skip document, log error, create empty JSONLLine
- **Parsing Errors**: JSON parsing fails (syntax errors, encoding issues) → Parse error → Skip document, log error, create empty JSONLLine
- **Type Mismatch**: API response types don't match schema (e.g., string instead of number) → Pydantic validation error → Skip document, log error, create empty JSONLLine
- **Missing Required Fields**: API response missing required schema fields → Pydantic validation error → Skip document, log error, create empty JSONLLine

## Error Handling

- **API Errors**: Logged, document skipped, empty `JSONLLine` created with error metadata
- **Parsing Errors**: Logged, document skipped, empty `JSONLLine` created
- **Alignment Errors**: Marked as `"no_match"`, extraction still included with {0, 0} interval
- **Validation Errors**: Logged, document skipped, error message in logs

### Error Metadata Structure

When an error occurs and an empty `JSONLLine` is created, error information is stored in the `extractions` array as a special error record (if supported) or logged separately. Error metadata includes:

- **Error Type**: Type of error (e.g., `"rate_limit"`, `"timeout"`, `"parsing_error"`, `"validation_error"`, `"api_error"`, `"conversion_failure"`)
- **Error Message**: Human-readable error message describing what went wrong
- **Document ID**: The document identifier where the error occurred
- **Provider**: API provider name (e.g., `"ollama"`, `"openai"`, `"gemini"`)
- **Retry Attempt**: Number of retry attempts made (if applicable, 0-indexed)
- **Timestamp**: When the error occurred (ISO 8601 format)

**Note**: In the current implementation, errors are primarily logged via structured logging rather than stored in the JSONLLine structure. Empty `JSONLLine` objects are created with empty `extractions` arrays, and error details are logged separately. Future versions may include error metadata directly in the JSONLLine structure.
