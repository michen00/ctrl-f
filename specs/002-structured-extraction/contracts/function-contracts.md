# Function Contracts

**Feature**: Structured Extraction with OpenAI/Gemini API Integration
**Date**: 2025-01-27

## Structured Extraction Module (`structured_extract.py`)

### `find_char_interval(text: str, extraction_text: str, fuzzy_threshold: int = 80) -> Tuple[Dict[str, int], str]`

Finds character interval for an extraction using fuzzy regex matching.

**Inputs**:

- `text: str` - Full document text to search in
- `extraction_text: str` - The extracted text to locate
- `fuzzy_threshold: int` - Minimum similarity score (0-100) for fuzzy matching (default: 80)

**Outputs**:

- `Tuple[Dict[str, int], str]` - Tuple of (char_interval dict, alignment_status)
  - `char_interval: Dict[str, int]` - `{"start_pos": int, "end_pos": int}`
  - `alignment_status: str` - `"match_exact"`, `"match_fuzzy"`, or `"no_match"`

**Errors**:

- No exceptions raised (always returns valid result, may return `{"start_pos": 0, "end_pos": 0}` with `"no_match"` if no match found)

**Preconditions**: `text` and `extraction_text` are non-empty strings, `fuzzy_threshold` in range [0, 100]

**Postconditions**: Returned `char_interval` has `start_pos >= 0` and `end_pos >= start_pos`, `alignment_status` is one of the valid values

---

### `_call_structured_extraction_api(text: str, schema: Dict[str, Any], provider: str = "openai", model: str | None = None) -> Dict[str, Any]`

Calls OpenAI or Gemini API with structured outputs.

**Inputs**:

- `text: str` - Document text to extract from
- `schema: Dict[str, Any]` - JSON Schema for structured output
- `provider: str` - API provider (`"openai"` or `"gemini"`, default: `"openai"`)
- `model: str | None` - Model name (optional, uses provider default)

**Outputs**:

- `Dict[str, Any]` - Extracted data as dict matching the schema

**Errors**:

- Raises `NotImplementedError` if API integration not yet implemented (placeholder)
- Raises `ValueError` if provider is invalid
- Raises `openai.APIError` or `google.api_core.exceptions.GoogleAPIError` for API errors
- Raises `TimeoutError` if request times out
- Raises `json.JSONDecodeError` if API response is not valid JSON

**Preconditions**: `text` is non-empty, `schema` is valid JSON Schema, `provider` is `"openai"` or `"gemini"`, API key is configured

**Postconditions**: Returned dict matches the provided schema structure

---

### `_flatten_extractions(data: Dict[str, Any], schema: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str, Dict[str, Any] | None]]`

Flattens extracted data into (field_name, value, attributes) tuples.

**Inputs**:

- `data: Dict[str, Any]` - Extracted data dict from API
- `schema: Dict[str, Any]` - JSON Schema definition
- `prefix: str` - Prefix for nested field names (default: `""`)

**Outputs**:

- `List[Tuple[str, str, Dict[str, Any] | None]]` - List of (field_name, value, attributes) tuples
  - `field_name: str` - Full field name (with dot notation for nested fields)
  - `value: str` - Extracted text value
  - `attributes: Dict[str, Any] | None` - Optional attributes (e.g., array index, nested path)

**Errors**:

- No exceptions raised (handles missing fields gracefully, returns empty list if no properties in schema)

**Preconditions**: `data` and `schema` are valid dicts, `schema` has `"properties"` key if nested structures exist

**Postconditions**: All returned tuples have non-empty `field_name` and `value`, `attributes` may be None

---

### `run_structured_extraction(schema: Dict[str, Any] | type[BaseModel], corpus_docs: List[CorpusDocument], provider: str = "openai", model: str | None = None, fuzzy_threshold: int = 80) -> List[JSONLLine]`

Runs structured extraction on corpus documents.

**Inputs**:

- `schema: Dict[str, Any] | type[BaseModel]` - JSON Schema dict or Pydantic model
- `corpus_docs: List[CorpusDocument]` - List of corpus documents
- `provider: str` - API provider (`"openai"` or `"gemini"`, default: `"openai"`)
- `model: str | None` - Model name (optional, uses provider default)
- `fuzzy_threshold: int` - Minimum similarity score for fuzzy matching (0-100, default: 80)

**Outputs**:

- `List[JSONLLine]` - List of JSONLLine objects, one per document

**Errors**:

- Continues processing on individual document errors (logs warnings, creates empty JSONLLine)
- Raises `NotImplementedError` if API integration not yet implemented
- Raises `ValueError` if provider is invalid or schema cannot be converted

**Preconditions**: `schema` is valid JSON Schema or Pydantic model, `corpus_docs` is non-empty list, `provider` is valid, API key is configured

**Postconditions**: Returned list has same length as `corpus_docs`, each JSONLLine has valid `document_id` matching input document

---

### `write_jsonl(jsonl_lines: List[JSONLLine], output_path: str | Path) -> None`

Writes JSONL lines to file.

**Inputs**:

- `jsonl_lines: List[JSONLLine]` - List of JSONLLine objects
- `output_path: str | Path` - Path to output JSONL file

**Outputs**:

- `None` (writes to file)

**Errors**:

- Raises `PermissionError` if file cannot be written
- Raises `OSError` if directory cannot be created
- Raises `ValueError` if `output_path` is invalid

**Preconditions**: `jsonl_lines` is non-empty list, `output_path` is valid path string or Path object

**Postconditions**: JSONL file exists at `output_path`, contains one JSON line per `JSONLLine` object, file is valid JSONL format

---

### `visualize_extractions(jsonl_path: str | Path, output_html_path: str | Path | None = None) -> str`

Visualizes extractions from JSONL file using langextract.

**Inputs**:

- `jsonl_path: str | Path` - Path to input JSONL file
- `output_html_path: str | Path | None` - Optional path to save HTML visualization (if None, returns HTML as string)

**Outputs**:

- `str` - HTML content as string

**Errors**:

- Raises `ImportError` if `langextract` is not available
- Raises `FileNotFoundError` if `jsonl_path` doesn't exist
- Raises `ValueError` if JSONL file is invalid format
- Raises `PermissionError` if HTML file cannot be written (when `output_html_path` provided)

**Preconditions**: `jsonl_path` exists and is valid JSONL file, `langextract` is installed

**Postconditions**: If `output_html_path` provided, HTML file exists at path; returned string contains valid HTML

---

## Integration Points

### Reused Functions

- **`schema_io.convert_json_schema_to_pydantic()`**: Converts JSON Schema to Pydantic model
- **`schema_io.validate_json_schema()`**: Validates JSON Schema format
- **`ingest.process_corpus()`**: Processes corpus and returns `CorpusDocument` objects
- **`ingest.convert_document_to_markdown()`**: Converts documents to Markdown

### Data Flow Contracts

1. **Schema Input**: JSON Schema or Pydantic model → Validated and converted to dict
2. **Document Processing**: `CorpusDocument` → Markdown text extracted
3. **API Call**: Markdown text + Schema → Structured JSON response
4. **Flattening**: Structured JSON → List of (field_name, value, attributes) tuples
5. **Character Alignment**: Extraction text + Document text → Character intervals
6. **Record Creation**: Flattened data + Character intervals → `ExtractionRecord` objects
7. **Line Creation**: `ExtractionRecord` list → `JSONLLine` object
8. **File Output**: `JSONLLine` list → JSONL file
9. **Visualization**: JSONL file → HTML visualization

## Error Handling Contracts

### API Error Handling

- **Rate Limit Errors (429)**: Retry with exponential backoff (max 3 retries), log warning, continue with next document
- **Timeout Errors**: Retry with exponential backoff (max 3 retries), log warning, continue with next document
- **Authentication Errors (401)**: Log error, raise exception (cannot continue)
- **Invalid Request Errors (400)**: Log error, skip document, create empty JSONLLine
- **Server Errors (5xx)**: Retry with exponential backoff (max 3 retries), log error, continue with next document

### Validation Error Handling

- **Schema Validation Errors**: Log error, raise exception (cannot continue without valid schema)
- **Character Interval Errors**: Log warning, set interval to {0, 0}, mark as "no_match", continue
- **JSONL Format Errors**: Log error, raise exception (cannot write invalid format)

### Graceful Degradation

- Individual document failures do not stop processing of other documents
- Empty `JSONLLine` objects are created for failed documents (with empty `extractions` list)
- Errors are logged with structured logging (document_id, error type, error message)
- User receives clear error messages and can review failed documents
