# Research & Technology Decisions

**Feature**: Schema-Grounded Corpus Extractor
**Date**: 2025-11-12

## Technology Stack Decisions

### Document Conversion: markitdown

**Decision**: Use markitdown library for converting documents (PDF, DOCX, HTML, TXT) to Markdown format.

**Rationale**:

- markitdown provides unified interface for multiple document formats
- Preserves source location mapping (page/line information) which is critical for provenance tracking
- Actively maintained and supports the formats required (PDF, DOCX, HTML, TXT)
- Python-native, integrates well with existing stack

**Alternatives considered**:

- pdfplumber + python-docx + BeautifulSoup: Would require format-specific handling and custom source mapping logic
- pandoc: More complex, requires external dependencies, less Python-native
- Custom parsers: Too much development overhead for v0

### Field Extraction: Structured Extraction (OpenAI/Gemini)

**Decision**: Use OpenAI/Gemini structured outputs for extracting candidate values from documents based on schema fields.

**Rationale**:

- Native structured output support with JSON Schema constraints
- Can condition extraction directly on schema without requiring in-context examples
- Supports grounding (source span tracking) via fuzzy matching (mandatory requirement FR-004)
- Returns confidence scores needed for consensus detection
- Works with both local models (Ollama) and cloud APIs (OpenAI, Gemini)

**Why not langextract**:

- **Requires in-context examples**: langextract requires few-shot examples in the prompt, which adds complexity and token overhead
- **Cannot condition on schema**: Unlike modern APIs, langextract cannot directly use JSON Schema to constrain outputs - it relies on prompt engineering with examples
- **Less flexible**: The need for examples makes it harder to adapt to different schemas dynamically

**Migration Note**: Previously used langextract for extraction, but this has been replaced with structured extraction via PydanticAI. langextract is now only used for visualization (`langextract.visualize()`).

**Alternatives considered**:

- Regex-only approach: Too brittle, doesn't handle variations well
- langextract: Requires in-context examples and cannot condition on schema directly
- Custom NLP pipeline: Too complex for v0

### Similarity Matching: TheFuzz

**Decision**: Use TheFuzz for candidate deduplication and similarity matching.

**Rationale**:

- Clean, Pythonic API for fuzzy string matching (wrapper around rapidfuzz)
- Fast performance (uses rapidfuzz under the hood with C++ backend)
- Well-established library (3.5k+ stars, used by 4.9k+ repositories)
- Provides multiple similarity algorithms (ratio, partial_ratio, token_sort_ratio, token_set_ratio)
- Simple API with `fuzz.ratio()` and `process.extract()` functions perfect for deduplication
- Handles normalization differences (whitespace, case, token order)
- MIT license, actively maintained

**Alternatives considered**:

- **rapidfuzz**: Fast but lower-level API. TheFuzz provides cleaner interface while using rapidfuzz internally.
- **PolyFuzz**: More complex, designed for larger-scale matching tasks. Overkill for v0 deduplication needs.
- **string2string**: Comprehensive but overkill for v0, more complex API
- **DeezyMatch**: Deep learning-based, too heavy for v0, requires training
- **Exact string matching**: Too strict, misses legitimate duplicates with minor variations
- **difflib**: Built-in but slower, less feature-rich
- **Custom similarity logic**: Unnecessary when proven library exists

### Database: TinyDB

**Decision**: Use TinyDB for local persistence of extracted records.

**Rationale**:

- Lightweight, file-based JSON database perfect for local-only requirement (FR-019)
- No external dependencies or server setup
- Simple API for storing structured data
- Supports indexing for quick lookups
- Python-native, integrates seamlessly

**Alternatives considered**:

- SQLite: More complex, overkill for single-user local application
- JSON files: No query/indexing capabilities, manual concurrency handling
- PostgreSQL/MySQL: Violates local-only requirement, requires server setup

### UI Framework: Gradio

**Decision**: Use Gradio for web-based user interface.

**Rationale**:

- Rapid prototyping and development of interactive UIs
- Built-in support for file uploads, progress indicators, and dynamic components
- Python-native, integrates with existing codebase
- Supports accordions, modals, and complex layouts needed for review interface
- Can be run locally, satisfies local-only requirement

**Alternatives considered**:

- Streamlit: Less flexible for complex review interface, more opinionated
- Flask/FastAPI + React: Too much development overhead for v0
- CLI-only: Rejected - interactive review requires visual interface (see Constitution Check)

### Schema Validation: Pydantic v2 + jsonschema

**Decision**: Use Pydantic v2 for data models and validation, jsonschema for JSON Schema validation.

**Rationale**:

- Pydantic v2 provides excellent type safety and validation (constitution requirement)
- Native support for JSON Schema conversion
- Fast validation performance
- Rich error messages for user feedback
- jsonschema validates JSON Schema format before conversion

**Alternatives considered**:

- Pydantic v1: v2 has better performance and features
- Marshmallow: Less type-safe, more verbose
- Custom validation: Unnecessary when Pydantic provides comprehensive solution

## Architecture Patterns

### Schema Extension Pattern

**Decision**: Coerce all schema fields to arrays (List[type]) for uniform handling.

**Rationale**:

- Simplifies extraction logic (always expect multiple candidates)
- Aligns with requirement that fields can have multiple values from multiple sources
- Makes validation consistent across all fields
- Original cardinality can be tracked for future UX improvements

**Implementation**: Create Extended Model from base schema by wrapping all leaf field types in List[] unless already arrays.

### Provenance Tracking Pattern

**Decision**: Every candidate value must include SourceRef with exact span location.

**Rationale**:

- Mandatory requirement (FR-004, FR-005) - zero fabrication
- Enables user trust through verifiable source locations
- Supports "View source" functionality in UI
- Critical for audit trail and debugging

**Implementation**: PydanticAI extraction returns structured data; use fuzzy matching to locate spans in documents; map to SourceRef with doc_id, path, location (page/line or char range), snippet, and metadata.

### Consensus Detection Pattern

**Decision**: Use fixed thresholds (confidence ≥0.75, margin ≥0.20) for consensus detection.

**Rationale**:

- Provides clear, testable criteria (from clarifications)
- Balances automation with accuracy
- Reduces false positives while flagging ambiguous cases
- Can be made configurable in future versions

**Implementation**: After deduplication, sort candidates by confidence, check if top candidate meets thresholds, mark as consensus or disagreement.

### Error Handling Pattern

**Decision**: Continue processing on errors, show warnings inline, summarize at end.

**Rationale**:

- Maximizes value from partial results
- User can see what succeeded and what failed
- Aligns with graceful degradation requirement (FR-013)
- Better UX than stopping on first error

**Implementation**: Try-except around document processing, collect errors, continue, display summary with failed items and reasons.

## Performance Considerations

### Document Processing

**Strategy**: Batch processing with parallelization by file where possible.

**Rationale**:

- markitdown conversion can be parallelized per document
- Extraction can process documents independently
- Progress tracking easier with batch approach
- Aligns with clarification Q5 (batch with progress indicator)

### Caching Strategy

**Decision**: Cache document conversions and extraction results keyed by corpus checksum + schema hash.

**Rationale**:

- Avoids reprocessing unchanged documents
- Speeds up re-runs with same corpus/schema
- Checksum-based invalidation ensures correctness

**Implementation**: Store converted Markdown and extraction results in cache directory, key by hash of corpus files + schema content.

## Security & Privacy

### Local Processing

**Decision**: All processing happens locally, no network transmission.

**Rationale**:

- Requirement FR-019 (local processing only)
- Protects sensitive document content
- No external dependencies or API calls
- User maintains full control of data

### PII Redaction

**Decision**: Optional PII redaction in UI previews with user-configurable patterns.

**Rationale**:

- Requirement FR-020
- Protects sensitive data in UI while preserving full data in storage
- User control over what gets redacted
- Patterns can be extended for different PII types

## Open Questions Resolved

1. **Normalization catalog**: Start with common types (emails, dates, URLs) - can be extended incrementally
2. **Confidence thresholds**: Fixed at 0.75 confidence, 0.20 margin (from clarifications)
3. **Null policy**: User-configurable, default empty list (FR-015)
4. **Large corpus strategy**: Batch processing with progress and cancellation (clarification Q5)
5. **Schema identity**: Use schema hash for TinyDB table naming
6. **Provenance fidelity**: Prefer page+line, fallback to char ranges when unavailable
7. **Offline packaging**: Defer containerization to v1 (per roadmap)
8. **Internationalization**: English-first, can be extended later
