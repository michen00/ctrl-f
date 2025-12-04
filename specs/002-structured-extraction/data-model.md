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
- `extraction_index: int` - Index of this extraction in the sequence (1-based)
- `group_index: int` - Group index for related extractions (for grouping related fields)
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
- `document_id: str` - Unique identifier for the document (from `CorpusDocument.doc_id`)

**Validation Rules**:

- `extractions` can be empty (no extractions found)
- `text` must be non-empty (document content)
- `document_id` must be non-empty and match a valid document ID

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
2. **API Call**: Document text + Schema → OpenAI/Gemini API → Structured JSON response
3. **Flattening**: Structured JSON → Flattened extractions (field_name, value, attributes)
4. **Character Alignment**: Extraction text + Document text → Character intervals + Alignment status
5. **Record Creation**: Flattened extractions + Character intervals → `ExtractionRecord` objects
6. **Line Creation**: `ExtractionRecord` list + Document text + Document ID → `JSONLLine`
7. **Output**: `JSONLLine` objects → JSONL file → Visualization

## Integration with Existing Models

### Reused from Existing Codebase

- **`CorpusDocument`** (from `ingest.py`): Input document with `doc_id`, `markdown`, `source_map`
- **Schema models** (from `schema_io.py`): JSON Schema or Pydantic models for extraction

### Relationship to Existing Extraction Models

This feature uses different models than the existing `extract.py` pipeline:

- **Existing**: `Candidate`, `FieldResult`, `ExtractionResult` (used by both extraction pipelines)
- **New**: `ExtractionRecord`, `JSONLLine` (for structured API extraction)

The two pipelines are independent and can coexist. The new pipeline focuses on JSONL output for visualization, while the existing pipeline focuses on candidate aggregation and consensus detection.

## Validation Points

1. **Before API Call**: Schema validation, API key validation, document text validation
2. **After API Response**: JSON parsing, schema compliance check
3. **After Flattening**: Extraction record validation (non-empty fields, valid types)
4. **After Character Alignment**: Interval validation (within document bounds)
5. **Before JSONL Write**: `JSONLLine` validation (all required fields, valid document_id)
6. **Before Visualization**: JSONL file format validation

## Error Handling

- **API Errors**: Logged, document skipped, empty `JSONLLine` created with error metadata
- **Parsing Errors**: Logged, document skipped, empty `JSONLLine` created
- **Alignment Errors**: Marked as `"no_match"`, extraction still included with {0, 0} interval
- **Validation Errors**: Logged, document skipped, error message in logs
