# Function Contracts

**Feature**: Schema-Grounded Corpus Extractor
**Date**: 2025-11-12

## Schema I/O Module (`schema_io.py`)

### `validate_json_schema(schema_json: str) -> Dict[str, Any]`

Validates JSON Schema format and returns parsed schema.

**Inputs**:

- `schema_json: str` - JSON Schema as string

**Outputs**:

- `Dict[str, Any]` - Validated and parsed JSON Schema

**Errors**:

- Raises `jsonschema.ValidationError` if schema is invalid
- Raises `json.JSONDecodeError` if JSON is malformed

**Preconditions**: None
**Postconditions**: Returned schema is valid JSON Schema

---

### `convert_json_schema_to_pydantic(schema: Dict[str, Any]) -> type[BaseModel]`

Converts JSON Schema to Pydantic v2 model class.

**Inputs**:

- `schema: Dict[str, Any]` - Validated JSON Schema

**Outputs**:

- `type[BaseModel]` - Pydantic model class

**Errors**:

- Raises `ValueError` if schema contains nested objects/arrays (v0 limitation)
- Raises `TypeError` if schema types cannot be mapped to Pydantic types

**Preconditions**: Schema is valid JSON Schema, flat structure only
**Postconditions**: Returned model can be instantiated and validated

---

### `import_pydantic_model(code: str) -> type[BaseModel]`

Imports Pydantic model from Python code string.

**Inputs**:

- `code: str` - Python code containing Pydantic model class definition

**Outputs**:

- `type[BaseModel]` - Pydantic model class

**Errors**:

- Raises `SyntaxError` if code is invalid Python
- Raises `ImportError` if model cannot be imported
- Raises `ValueError` if model contains nested objects/arrays (v0 limitation)

**Preconditions**: Code is valid Python, contains exactly one BaseModel subclass
**Postconditions**: Returned model can be instantiated and validated

---

### `extend_schema(model_cls: type[BaseModel]) -> type[BaseModel]`

Creates Extended Schema by coercing all fields to arrays.

**Inputs**:

- `model_cls: type[BaseModel]` - Original Pydantic model

**Outputs**:

- `type[BaseModel]` - Extended model with all fields as List[type]

**Errors**:

- Raises `ValueError` if model contains nested objects/arrays

**Preconditions**: Model is flat (no nested structures)
**Postconditions**: All fields in returned model are List types

---

## Ingest Module (`ingest.py`)

### `convert_document_to_markdown(file_path: str) -> Tuple[str, Dict[str, Any]]`

Converts a document to Markdown and returns source mapping.

**Inputs**:

- `file_path: str` - Path to document file

**Outputs**:

- `Tuple[str, Dict[str, Any]]` - (markdown_content, source_map)
  - `markdown_content: str` - Converted Markdown text
  - `source_map: Dict[str, Any]` - Mapping of Markdown positions to original locations (page/line/char ranges)

**Errors**:

- Raises `FileNotFoundError` if file doesn't exist
- Raises `ValueError` if file format is unsupported
- Raises `RuntimeError` if conversion fails (corrupted file)

**Preconditions**: File exists and is readable
**Postconditions**: Markdown preserves content, source_map enables span mapping

---

### `process_corpus(corpus_path: str, progress_callback: Callable[[int, int], None] | None = None) -> List[Tuple[str, str, Dict[str, Any]]]`

Processes entire corpus, converting all documents to Markdown.

**Inputs**:

- `corpus_path: str` - Path to corpus (directory, zip, or tar archive)
- `progress_callback: Callable[[int, int], None] | None` - Optional callback(doc_count, total_docs) for progress updates

**Outputs**:

- `List[Tuple[str, str, Dict[str, Any]]]` - List of (doc_id, markdown, source_map) tuples

**Errors**:

- Raises `ValueError` if corpus_path is invalid
- Continues processing on individual file errors, collects errors for summary

**Preconditions**: Corpus path exists and is readable
**Postconditions**: All successfully converted documents returned, errors collected for reporting

---

## Extract Module (`extract.py`)

### `extract_field_candidates(field_name: str, field_type: type, field_description: str | None, markdown_content: str, doc_id: str, source_map: Dict[str, Any]) -> List[Candidate]`

Extracts candidate values for a single field from a single document.

**Inputs**:

- `field_name: str` - Name of schema field
- `field_type: type` - Python type of field (str, int, float, date, etc.)
- `field_description: str | None` - Optional field description from schema
- `markdown_content: str` - Document content in Markdown
- `doc_id: str` - Document identifier
- `source_map: Dict[str, Any]` - Source mapping for span location

**Outputs**:

- `List[Candidate]` - List of candidate values with confidence scores and source references

**Errors**:

- Returns empty list if extraction fails (no candidates found)
- Logs warnings for extraction errors but doesn't raise

**Preconditions**: markdown_content is valid, source_map is valid
**Postconditions**: All candidates have non-empty sources (zero fabrication)

---

### `run_extraction(model: type[BaseModel], corpus_docs: List[Tuple[str, str, Dict[str, Any]]]) -> ExtractionResult`

Runs extraction for all fields across all documents.

**Inputs**:

- `model: type[BaseModel]` - Extended Pydantic model (all fields as arrays)
- `corpus_docs: List[Tuple[str, str, Dict[str, Any]]]` - List of (doc_id, markdown, source_map)

**Outputs**:

- `ExtractionResult` - Complete extraction results with all field results

**Errors**:

- Continues processing on individual field/document errors
- Logs errors but doesn't raise (graceful degradation)

**Preconditions**: Model is Extended Schema, corpus_docs are valid
**Postconditions**: ExtractionResult contains FieldResult for each schema field

---

## Aggregate Module (`aggregate.py`)

### `normalize_value(value: Any, field_type: type) -> Any`

Normalizes a candidate value based on field type.

**Inputs**:

- `value: Any` - Raw candidate value
- `field_type: type` - Schema field type

**Outputs**:

- `Any` - Normalized value (e.g., lowercase email, ISO date, trimmed string)

**Errors**: Returns original value if normalization fails

**Preconditions**: value matches field_type
**Postconditions**: Normalized value is canonical form

---

### `deduplicate_candidates(candidates: List[Candidate], similarity_threshold: float = 0.85) -> List[Candidate]`

Groups near-duplicate candidates using similarity matching.

**Inputs**:

- `candidates: List[Candidate]` - List of candidate values
- `similarity_threshold: float` - Minimum similarity (0-1) to consider duplicates

**Outputs**:

- `List[Candidate]` - Deduplicated candidates with merged sources and averaged confidence

**Errors**: None (always returns valid list)

**Preconditions**: All candidates have non-empty sources
**Postconditions**: Returned candidates are sorted by confidence (descending)

---

### `detect_consensus(candidates: List[Candidate], confidence_threshold: float = 0.75, margin_threshold: float = 0.20) -> Candidate | None`

Detects if there's a consensus candidate meeting thresholds.

**Inputs**:

- `candidates: List[Candidate]` - List of deduplicated candidates (sorted by confidence)
- `confidence_threshold: float` - Minimum confidence for consensus (default 0.75)
- `margin_threshold: float` - Minimum margin over next candidate (default 0.20)

**Outputs**:

- `Candidate | None` - Consensus candidate if thresholds met, None otherwise

**Errors**: None

**Preconditions**: Candidates are sorted by confidence (descending)
**Postconditions**: Returned candidate (if any) meets both thresholds

---

### `aggregate_field_results(field_name: str, candidates: List[Candidate]) -> FieldResult`

Aggregates candidates for a field into FieldResult with consensus detection.

**Inputs**:

- `field_name: str` - Schema field name
- `candidates: List[Candidate]` - All candidates for this field

**Outputs**:

- `FieldResult` - Aggregated results with consensus if detected

**Errors**: None

**Preconditions**: All candidates are for the same field
**Postconditions**: FieldResult.candidates are deduplicated and sorted

---

## Storage Module (`storage.py`)

### `save_record(record: PersistedRecord, table_name: str | None = None) -> str`

Saves a persisted record to TinyDB.

**Inputs**:

- `record: PersistedRecord` - Validated record to save
- `table_name: str | None` - Optional table name (defaults to schema hash)

**Outputs**:

- `str` - Record ID of saved record

**Errors**:

- Raises `ValueError` if record validation fails
- Raises `RuntimeError` if database write fails

**Preconditions**: Record is validated against Extended Schema
**Postconditions**: Record is persisted and retrievable by record_id

---

### `export_record(record_id: str, table_name: str | None = None) -> Dict[str, Any]`

Exports a record as JSON-serializable dictionary.

**Inputs**:

- `record_id: str` - Record identifier
- `table_name: str | None` - Optional table name

**Outputs**:

- `Dict[str, Any]` - JSON-serializable record data

**Errors**:

- Raises `KeyError` if record not found

**Preconditions**: Record exists in database
**Postconditions**: Returned dict is JSON-serializable

---

## UI Module (`ui.py`)

### `create_upload_interface() -> gr.Blocks`

Creates Gradio interface for schema and corpus upload.

**Inputs**: None

**Outputs**:

- `gr.Blocks` - Gradio interface component

**Preconditions**: None
**Postconditions**: Interface accepts schema (JSON/Python) and corpus (directory/zip)

---

### `create_review_interface(extraction_result: ExtractionResult) -> gr.Blocks`

Creates Gradio interface for reviewing and resolving candidates.

**Inputs**:

- `extraction_result: ExtractionResult` - Extraction results to review

**Outputs**:

- `gr.Blocks` - Gradio interface component with field accordions

**Preconditions**: ExtractionResult is valid
**Postconditions**: Interface allows selection/resolution for all fields

---

### `show_source_context(sources: List[SourceRef]) -> str`

Generates formatted source context display.

**Inputs**:

- `sources: List[SourceRef]` - Source references to display

**Outputs**:

- `str` - Formatted Markdown string with snippets and metadata

**Preconditions**: Sources are non-empty
**Postconditions**: Returned string is valid Markdown

---

## Server Module (`server.py`)

### `main() -> None`

Main entrypoint to launch Gradio application.

**Inputs**: None

**Outputs**: None (runs server)

**Preconditions**: All dependencies installed
**Postconditions**: Gradio server running on localhost
