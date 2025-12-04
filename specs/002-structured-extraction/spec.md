# Feature Specification: Structured Extraction with OpenAI/Gemini API Integration

**Feature Branch**: `002-structured-extraction`
**Created**: 2025-01-27
**Status**: Draft
**Input**: User description: "Structured Extraction with OpenAI/Gemini API Integration"

## Feature Description

Replace the existing langextract-based extraction with a new extraction pipeline that uses PydanticAI with Ollama (default), OpenAI, or Gemini. This is now the primary extraction method, integrated into the existing Gradio UI. PydanticAI unifies schema handling across all LLM providers using Pydantic models. The pipeline supports local development with Ollama and can use cloud APIs (OpenAI/Gemini) for production.

**Why langextract was replaced**: langextract requires in-context examples (few-shot learning) and cannot condition extraction directly on the schema like modern APIs. PydanticAI allows us to pass the schema directly as a Pydantic model, eliminating the need for example generation and providing better schema adherence.

Note: langextract is now only used for visualization (`langextract.visualize()`), not for extraction.

## Clarifications

### Session 2025-01-27

- Q: Should this be a separate extraction mode or integrated into the existing UI? → A: Integrated into existing Gradio UI as the primary extraction option
- Q: How should API keys be managed (env vars, config file, UI input)? → A: Config file support (read from ~/.ctrlf/config.toml, with env var and UI override options)
- Q: Should this support the same disambiguation/consensus logic as the existing pipeline? → A: Users review all unique extractions, plurality vote provides a suggested form (no auto-selection, no disagreement flags, but suggestion mechanism based on most common value)
- Q: Should we support batch processing of multiple documents in a single API call? → A: One document per API call (sequential processing, one API request per document)
- Q: How should we handle schema complexity beyond what the APIs support? → A: Validate before processing, reject incompatible schemas with clear errors (proactive validation)

## Context

The implementation in `src/ctrlf/app/structured_extract.py` is complete and provides:

- Data models (`ExtractionRecord`, `JSONLLine`) matching the JSONL format required by `langextract.visualize()`
- Fuzzy character interval finding using `thefuzz` to locate extracted text in documents
- JSONL file writing and visualization wrapper
- Schema flattening for nested structures
- **PydanticAI integration**: `_call_structured_extraction_api()` uses PydanticAI Agent to support Ollama (default), OpenAI, and Gemini providers with unified schema handling

**Implementation Status**: ✅ Complete - The PydanticAI integration is fully implemented and supports all three providers (Ollama, OpenAI, Gemini) through a unified interface.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Extract Using Structured Outputs (Priority: P1)

A user wants to extract structured data from documents using PydanticAI with Ollama (default), OpenAI, or Gemini. They provide a schema and corpus, and the system processes each document individually using PydanticAI Agent with Pydantic model constraints, then generates a JSONL file for visualization.

**Why this priority**: This is the core value proposition - leveraging PydanticAI for unified schema-based extraction with support for local (Ollama) and cloud (OpenAI/Gemini) providers.

**Provider Details**:

- **Ollama** (default): Local development provider, no API keys required, runs models locally
- **OpenAI**: Cloud provider, requires OPENAI_API_KEY, uses structured outputs via PydanticAI
- **Gemini**: Cloud provider, requires GOOGLE_API_KEY, uses structured outputs via PydanticAI

**Independent Test**: Can be tested by providing a simple schema and a small corpus (2-3 documents), calling PydanticAI Agent with provider="ollama" (or "openai"/"gemini"), and verifying the JSONL output format matches expectations.

**Acceptance Scenarios**:

1. **Given** a user has a Pydantic model schema and corpus documents, **When** they run structured extraction with any provider (Ollama/OpenAI/Gemini), **Then** the system calls PydanticAI Agent with the schema model and returns structured data
2. **Given** PydanticAI Agent returns structured data, **When** the system processes the response, **Then** it generates valid JSONL lines with character intervals and alignment status
3. **Given** extraction completes successfully, **When** the user requests visualization, **Then** the system generates HTML visualization using `langextract.visualize()`

---

### User Story 2 - Provider Selection and Configuration (Priority: P1)

A user wants to select between Ollama (local), OpenAI, or Gemini providers for structured extraction. The system should support all three providers with unified functionality through PydanticAI.

**Why this priority**: Provider flexibility is important for users who prefer local development (Ollama), different cloud APIs (OpenAI/Gemini), or have different cost/performance requirements.

**Independent Test**: Can be tested by switching provider to "ollama", "openai", or "gemini" and verifying PydanticAI Agent calls use the correct provider configuration.

**Acceptance Scenarios**:

1. **Given** a user selects a provider (Ollama/OpenAI/Gemini), **When** they run structured extraction, **Then** the system calls PydanticAI Agent with the correct provider model string (e.g., "ollama:llama3", "openai:gpt-4o", "google-gla:gemini-2.5-flash")
2. **Given** PydanticAI Agent returns structured data from any provider, **When** the system processes the response, **Then** it generates the same JSONL format regardless of provider
3. **Given** API errors occur (rate limits, timeouts), **When** processing documents, **Then** the system continues with other documents and logs clear error messages

---

### User Story 3 - Visualize and Export JSONL Results (Priority: P2)

A user wants to visualize extractions in HTML format and export results as JSONL for downstream processing. The system should integrate with `langextract.visualize()` and support file output.

**Why this priority**: Visualization and export capabilities enable users to review results and integrate with other tools.

**Independent Test**: Can be tested by generating a JSONL file and verifying it can be visualized and that the format matches `langextract.visualize()` expectations.

**Acceptance Scenarios**:

1. **Given** extraction results are available, **When** the user requests visualization, **Then** the system generates HTML using `langextract.visualize()` and saves to file
2. **Given** a JSONL file is generated, **When** the user opens it, **Then** each line contains valid JSON with the expected structure (extractions, text, document_id)
3. **Given** character intervals are calculated, **When** viewing visualizations, **Then** extractions are correctly highlighted in the source text

---

### Edge Cases

- **API Rate Limits (429 errors)**: System MUST implement retry logic with exponential backoff (max 3 retries, initial delay 1s, multiplier 2.0), continue processing other documents, and log rate limit errors clearly with document_id and retry attempt number. If all retries exhausted, skip document and create empty JSONLLine with error metadata.
- **Token Limit Exceeded**: System MUST detect token limits before API calls (estimate tokens from document length), split documents if possible (chunk at sentence boundaries), or skip with clear error message indicating token count and limit. For documents exceeding maximum chunk size, skip entirely.
- **Fuzzy Matching Failure**: System MUST mark alignment_status as "no_match" and set char_interval to {0, 0} when fuzzy matching cannot find a character interval (similarity < threshold), but still include the extraction in results.
- **Schema Complexity Beyond API Support**: System MUST validate schema compatibility before API calls (check for unsupported types, excessive nesting depth > 3 levels, arrays of complex objects) and provide clear, actionable error messages listing unsupported features. Reject incompatible schemas with specific guidance on how to simplify.
- **Missing/Invalid API Keys**: System MUST check for API keys in precedence order (UI input → env vars → config file), provide clear error messages indicating which method was checked and which provider requires keys, and guide users to configure keys via the preferred method. For Ollama provider, no API key check is performed.
- **Network Timeouts**: System MUST implement timeout handling (default 60s, configurable), retry logic with exponential backoff (max 3 retries), and continue processing other documents on timeout. Log timeout errors with document_id and retry attempt.
- **Zero Extractions Found**: System MUST create JSONLLine with empty extractions list when no extractions are found in a document. Document is still included in output with empty extractions array. This scenario occurs when API returns empty object or no matching fields in document. System MUST NOT skip documents with zero extractions - they are included in output for completeness.
- **Partial Extraction Failures**: System MUST include successfully extracted fields and mark failed fields with error metadata. If API returns partial results (some fields extracted, others missing), process available extractions and log warnings for missing fields. Failed fields are not included in extractions list, but error is logged with field name and failure reason. This scenario occurs when API cannot extract all schema fields from document.
- **Empty Corpus**: System MUST validate corpus is non-empty before processing and return clear error message if corpus is empty (e.g., "Corpus is empty. Please provide at least one document."). No API calls should be made for empty corpus. Validation occurs before any processing begins.
- **Malformed Documents**: System MUST handle corrupted files and invalid markdown gracefully. If document conversion fails (handled by `ingest.py`), skip document and create empty JSONLLine with error metadata indicating conversion failure reason (e.g., "Document conversion failed: invalid PDF format"). This scenario is handled upstream by `ingest.py` module, but system MUST handle conversion failures gracefully.
- **Empty Extraction Text**: System MUST handle empty string extraction values from API. Empty strings are valid extractions and should be included with alignment_status "no_match" if not found in source (empty string cannot be matched in document). Empty string extractions are included in results with char_interval {0, 0}.
- **Duplicate Document IDs**: System MUST handle duplicate document IDs in corpus by using first occurrence and logging warning for subsequent duplicates (e.g., "Duplicate document_id 'doc123' found, using first occurrence"). Each document_id should map to exactly one JSONLLine. Subsequent documents with duplicate IDs are processed but logged as warnings.
- **Unicode/Encoding Edge Cases**: System MUST handle Unicode characters, emoji, and special characters correctly. Character intervals MUST be calculated using UTF-8 byte positions, not character counts, for accurate alignment. Fuzzy matching uses UTF-8 encoding. Character intervals are byte-based (0-based indexing of UTF-8 bytes), not character-based. This ensures accurate alignment for multi-byte characters.
- **Visualization Failures**: System MUST handle `langextract.visualize()` errors gracefully. If visualization fails (e.g., invalid JSONL format, library error), log error with details (error type, error message, document_id) and return error message to user (e.g., "Visualization failed: [error message]"). Do not fail entire extraction workflow - extraction results are still available even if visualization fails.
- **JSONL Write Failures**: System MUST handle disk full and permission errors when writing JSONL files. If write fails (disk full, permission denied, path invalid), return clear error message indicating file path and failure reason (e.g., "Failed to write JSONL file: Permission denied at /path/to/file.jsonl"). Do not lose extraction results if write fails - results remain in memory and can be retried or written to different location. System SHOULD attempt to create parent directories if they don't exist.
- **Schema Validation Failures**: System MUST validate JSON Schema format before processing. If schema is invalid (malformed JSON, invalid draft-07 format, unsupported features), reject with clear error message indicating validation errors and line numbers (e.g., "Schema validation failed at line 5: 'null' type not supported"). Schema validation occurs before any API calls (fail fast per NFR-004).
- **Provider Switching Mid-Corpus**: System MUST support same provider for entire corpus processing. Provider selection applies to all documents in corpus. Changing provider mid-corpus is not supported in v0 - provider is selected once at start of processing and used for all documents. Future versions may support provider switching.
- **Very Large Corpora (100+ documents)**: System MUST handle large corpora by processing documents sequentially, respecting rate limits, and providing progress feedback. For corpora exceeding 100 documents, log warning (e.g., "Large corpus detected: 150 documents. Processing may take longer.") and continue processing. Rate limit handling ensures no data loss. Progress feedback SHOULD be provided (e.g., "Processing document 50/150"). Sequential processing ensures predictable behavior for large corpora.
- **Schema Evolution (Schema Changes Between Documents)**: System MUST use same schema for entire corpus processing. Schema is validated once at start and applied to all documents. Schema changes between documents are not supported in v0 - if schema needs to change, process documents in separate batches with different schemas. This scenario is explicitly not supported - schema is fixed for entire corpus.
- **Concurrent Document Processing**: System MUST NOT process documents concurrently in v0. All documents are processed sequentially (one at a time). Concurrent processing is explicitly not supported - see NFR-003 for future parallelization plans. This scenario is out of scope for v0.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST support PydanticAI Agent for structured extraction with Ollama (default), OpenAI, and Gemini providers. PydanticAI unifies schema handling by accepting Pydantic models as `output_type`, eliminating the need for separate API client implementations. PydanticAI provides structured outputs capability that enforces schema compliance at the API level, ensuring extracted data matches the provided Pydantic model structure. Structured outputs means the API returns data that conforms to the Pydantic model schema without requiring post-processing validation - the API enforces the schema constraints directly, reducing extraction errors and ensuring type safety.
- **FR-002**: System MUST use Ollama as the default provider for local development. Ollama requires no API keys and runs models locally, making it ideal for development and testing. Ollama provider uses model string format "ollama:{model_name}" (e.g., "ollama:llama3").
- **FR-003**: System MUST process each document in the corpus individually, making one API call per document (sequential processing, no batching multiple documents into single API requests). This is distinct from batch processing where multiple documents would be sent in one API call. Sequential processing means documents are processed one at a time in the order provided, waiting for each API call to complete before starting the next. This differs from parallel processing where multiple documents would be processed concurrently. Sequential processing ensures predictable behavior, simplifies error handling, and respects rate limits. Future parallelization (see NFR-003) would still maintain one API call per document but process multiple documents concurrently.
- **FR-004**: System MUST use fuzzy string matching (via `thefuzz` library, using `fuzz.ratio()` algorithm) to locate extracted text in source documents and calculate character intervals. This is NOT regex matching - it uses string similarity scoring. Character intervals are 0-based with exclusive end position (i.e., `start_pos` is the first character included, `end_pos` is the first character NOT included, so interval is `[start_pos, end_pos)`). Fuzzy matching uses configurable threshold (default 80, range 0-100) to determine match quality. Threshold is configurable via UI parameter or function argument (default: 80).
- **FR-005**: System MUST generate JSONL files compatible with `langextract.visualize()` format. Each line contains valid JSON with required fields: `extractions` (array of ExtractionRecord objects), `text` (full document markdown text), `document_id` (unique document identifier). Format matches `langextract.visualize()` input expectations exactly.
- **FR-006**: System MUST track alignment status for each extraction with three possible values: `match_exact` (exact text match found, case-sensitive or case-insensitive), `match_fuzzy` (fuzzy match found with similarity >= threshold), `no_match` (no match found, char_interval set to {0, 0}).
- **FR-016**: System MUST deduplicate extractions to show all unique values for user review. Deduplication uses exact string matching (case-sensitive, no fuzzy matching) to identify unique extraction values. All unique values are presented to user for selection. Deduplication is performed per field across all documents in the corpus.
- **FR-017**: System MUST provide a suggested value based on plurality vote (most common extraction value across all documents) but MUST NOT auto-select it. Plurality vote calculates frequency of each unique extraction value (case-sensitive exact match) and suggests the value with highest count. If multiple values have the same highest count (tie), the first value encountered is suggested. Suggestion is visually indicated in UI (e.g., highlighted, checkmarked, or labeled as "suggested") but requires explicit user selection (click, checkbox, or button press) to be accepted. User can select any value, not just the suggested one.
- **FR-007**: System MUST handle API errors gracefully with specific behaviors for each error type: (1) Rate limits (429): Retry with exponential backoff (max 3 retries, initial delay 1s, multiplier 2.0), continue processing other documents, log errors with document_id. If all retries exhausted, skip document and create empty JSONLLine with error metadata. (2) Timeouts: Retry with exponential backoff (same parameters as rate limits), continue processing other documents, log timeout errors with document_id and retry attempt. If all retries exhausted, skip document. (3) Invalid responses (400, 422): Skip document immediately (no retry), create empty JSONLLine with error metadata, log error with details including error message from API. (4) Authentication failures (401): Stop processing immediately (no retry, no document skipping), return error to user with clear message indicating which provider requires authentication and which configuration method was checked. (5) Server errors (5xx): Retry with exponential backoff (max 3 retries), continue processing other documents, log server errors with document_id. If all retries exhausted, skip document. All errors MUST be logged with structured logging including document_id, provider, error type, error message, and retry attempt number. Error messages MUST be actionable (e.g., "API key missing for OpenAI provider, checked: UI input → env vars → config file").
- **FR-008**: System MUST support API key configuration with precedence order: UI input (highest priority) → environment variables (OPENAI_API_KEY, GOOGLE_API_KEY) → config file (~/.ctrlf/config.toml) (lowest priority). Note: Ollama provider does not require API keys. Config file format: TOML with `[api_keys]` section containing `openai_api_key` and `google_api_key` fields. If API key missing for required provider, system MUST provide clear error message indicating which configuration method was checked and guide user to configure keys.
- **FR-009**: System MUST allow selection of provider (Ollama, OpenAI, or Gemini) and model. Default provider is "ollama" with default model "llama3". For OpenAI, default model is "gpt-4o" (model string format: "openai:gpt-4o"). For Gemini, default model is "gemini-2.5-flash" (model string format: "google-gla:gemini-2.5-flash"). Provider selection applies to entire corpus processing (cannot change mid-corpus in v0).
- **FR-010**: System MUST support configurable fuzzy matching threshold. Threshold is integer value in range 0-100 (default: 80), configurable via UI or function parameter. Threshold determines minimum similarity score (using `fuzz.ratio()`) required for fuzzy match. Values below threshold result in `no_match` alignment status.
- **FR-011**: System MUST integrate with `langextract.visualize()` to generate HTML visualizations. `langextract.visualize()` expects JSONL file input with format matching FR-005 specification. System MUST handle visualization errors gracefully (log errors, return error message, do not fail extraction workflow).
- **FR-012**: System MUST support saving visualizations to file (when `output_html_path` provided) or returning HTML as string (when `output_html_path` is None). File save MUST handle permission errors and disk full scenarios gracefully.
- **FR-013**: System MUST flatten nested schema structures for extraction (handle arrays and nested objects). Flattening converts nested structures to flat field names using dot notation (e.g., "person.name", "items.0.value"). Arrays are flattened with index notation. Nested objects are recursively flattened to maximum depth of 3 levels (deeper nesting rejected during schema validation).
- **FR-014**: System MUST integrate into existing Gradio UI as the primary extraction option. "Primary extraction option" means this extraction method is the default/recommended option in the UI (e.g., selected by default, listed first, or prominently displayed), but existing `extract.py` pipeline can coexist. Users can select between extraction methods in UI (e.g., dropdown, radio buttons, or tabs). New pipeline is presented as primary/recommended option but does NOT replace the existing pipeline - both extraction methods are available. This requirement coexists with FR-015 (backward compatibility) - the new pipeline is primary but does not remove or break the existing pipeline.
- **FR-015**: System MUST maintain backward compatibility with existing `extract.py` logic. Backward compatibility means: (1) Existing `extract.py` functions remain unchanged and functional (no breaking changes to function signatures, return types, or behavior), (2) Existing data models (`Candidate`, `FieldResult`, `ExtractionResult`) remain unchanged (no schema changes, no field removals, no type changes), (3) Existing UI components continue to work (all existing UI features function as before), (4) Both extraction pipelines can coexist (users can choose which to use). New pipeline is primary (per FR-014) but does not replace existing pipeline - both are available and functional. This ensures no breaking changes for existing users or integrations.

### Non-Functional Requirements

- **NFR-001**: System SHOULD implement retry logic with exponential backoff for API calls. Retry parameters: max_retries=3, initial_delay=1.0s, multiplier=2.0 (delays: 1s, 2s, 4s). Retry applies to rate limit errors (429), timeout errors, and server errors (5xx). Do not retry authentication errors (401) or invalid request errors (400, 422).
- **NFR-002**: System SHOULD provide cost tracking/estimation for API usage. Cost tracking MUST log token usage (input tokens, output tokens) per API call with document_id and provider. Token usage MUST be logged at INFO level with structured logging fields: `tokens_input`, `tokens_output`, `provider`, `model`, `document_id`. Cost estimation helpers MAY calculate estimated cost based on provider pricing (optional feature, not required). If cost estimation is implemented, it SHOULD use current provider pricing (OpenAI: per-token pricing for gpt-4o, Gemini: per-token pricing for gemini-2.5-flash) and calculate cost as: `(input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)`. Cost tracking granularity is per-document (one log entry per API call). Ollama provider has zero cost (local processing).
- **NFR-003**: System MAY support parallel processing of documents in future (respecting rate limits), but v0 uses sequential processing (one API call per document, processed one at a time). Sequential processing ensures predictable behavior and simplifies error handling. Future parallelization MUST respect provider rate limits (e.g., max concurrent requests per provider), maintain sequential output order (results ordered by input document order), and handle errors gracefully (one document failure does not stop other concurrent documents). Parallel processing is explicitly NOT supported in v0 - all documents are processed sequentially. This requirement clarifies that sequential processing (FR-003) and future parallelization are not conflicting - v0 is sequential, future versions may add parallel processing while maintaining one API call per document.
- **NFR-004**: System MUST validate schema compatibility with API capabilities before processing and MUST reject incompatible schemas with clear, actionable error messages (fail fast, do not attempt API calls with invalid schemas). Schema validation checks: (1) JSON Schema format validity (draft-07 specification), (2) Maximum nesting depth <= 3 levels (nested objects within nested objects, arrays of objects count toward depth), (3) Supported types only (string, number, integer, boolean, array, object - no null, no custom types), (4) No circular references (objects cannot reference themselves directly or indirectly). A schema is incompatible if it violates any of these checks. Incompatible schemas MUST be rejected before any API calls with error message listing specific incompatibilities (e.g., "Schema has nesting depth 4, maximum allowed is 3" or "Schema contains unsupported type 'null'") and guidance on how to simplify schema (e.g., "Flatten nested structures or reduce nesting depth").
- **NFR-005**: System SHOULD handle large documents by detecting token limits and splitting if necessary. Token limit detection SHOULD estimate tokens from document length (approximate 4 characters per token, UTF-8 encoding). If document exceeds provider token limit, split at sentence boundaries (preserve context, split on sentence endings like ". ", "! ", "? " followed by space or newline). If document exceeds maximum chunk size (provider limit) or cannot be split meaningfully, skip document with clear error message indicating token count and limit. Token limits: OpenAI gpt-4o ~128k tokens (input + output combined), Gemini gemini-2.5-flash ~1M tokens (input + output combined), Ollama varies by model (check model-specific limits). Token estimation is approximate - actual API usage may vary.
- **NFR-006**: System SHOULD calculate plurality vote (most common extraction value) for suggestion purposes. Plurality vote algorithm: (1) Count frequency of each unique extraction value (case-sensitive exact match), (2) Select value with highest count, (3) If tie, select first value encountered. Plurality vote is calculated per field across all documents in corpus.
- **NFR-007**: System SHOULD meet performance targets: (1) Process documents at API rate limits (no artificial throttling, respect provider rate limits), (2) Character alignment within 2 seconds per document (fuzzy matching performance for typical document sizes < 10k characters), (3) Generate JSONL files efficiently for large corpora (streaming write, no memory accumulation, O(1) memory per document). Performance targets are best-effort and may vary based on document size and API response times. API call latency depends on provider and network conditions (not controlled by system). Processing time per document = API call time + character alignment time (~2s) + JSONL write time (~0.1s).
- **NFR-008**: System SHOULD support corpus sizes up to 100 documents (same as existing extraction pipeline). For corpora exceeding 100 documents, system SHOULD log warning and continue processing. Rate limit handling ensures no data loss for large corpora. Scalability beyond 100 documents is not guaranteed in v0. System supports single-user, single-session operation (no concurrent users, no multi-session support). Corpus size limit is a guideline, not a hard limit - system will attempt to process larger corpora but performance may degrade.
- **NFR-009**: System MUST implement security best practices for API key handling: (1) API keys MUST NOT be logged in plaintext (mask in logs, show only first 4 characters like "sk-..."), (2) API keys MUST NOT be included in error messages sent to users, (3) API keys SHOULD be stored securely (config file permissions 600 (owner read/write only), env vars preferred over config file), (4) API keys MUST be transmitted over HTTPS for cloud providers (enforced by API clients, not system responsibility), (5) API keys MUST NOT be committed to version control (excluded via .gitignore). Ollama provider does not require API keys (local only, no network transmission). Security requirements apply to all API key storage and transmission mechanisms (UI input, env vars, config file).
- **NFR-010**: System MUST implement structured logging with log levels: INFO (API calls, milestones, token usage), DEBUG (per-extraction details, character alignment details), WARN (API errors, retries, warnings, rate limit warnings), ERROR (fatal errors, authentication failures). Structured logging MUST include contextual fields: `document_id`, `provider`, `model`, `extraction_index`, `error_type`, `retry_attempt`, `tokens_input`, `tokens_output`. Logging configuration uses `structlog` as specified in plan.md. Log output format is JSON for machine parsing or human-readable text for development.
- **NFR-011**: System SHOULD maintain code quality standards: (1) Test coverage >= 80% for new code (measured via pytest-cov), (2) All functions have type hints (Python 3.12+ type hints, no `Any` types without justification), (3) All public functions have docstrings (Google-style docstrings), (4) Code passes `ruff` linting (no errors, warnings acceptable) and `mypy` type checking (strict mode, no type errors). Maintainability requirements align with project standards defined in AGENTS.md. Code organization follows existing project structure (modules in `src/ctrlf/app/`, tests in `tests/unit/` and `tests/integration/`).
- **NFR-012**: System MUST maintain accessibility standards for UI integration. If UI changes are made, they MUST comply with WCAG 2.1 Level AA standards (keyboard navigation, screen reader support, color contrast ratios, focus indicators). Existing UI accessibility features MUST be preserved (no regressions). New UI components (if any) MUST be accessible. Accessibility testing SHOULD be performed for any UI changes.

## Technical Considerations

- API costs and rate limiting
- Handling large documents (token limits)
- Schema complexity (nested objects, arrays)
- Character alignment accuracy for fuzzy matches
- Performance for large corpora (parallelization options)
- API key security and management
- Error handling and retry strategies

## Assumptions

- **A-001**: API providers (OpenAI, Gemini) are generally available and accessible via network. Rate limits may apply but APIs are operational. APIs may experience temporary outages or rate limiting, but are generally reliable. Ollama is assumed to be installed and running locally if used. Network connectivity is assumed for cloud providers (OpenAI, Gemini). API availability is not guaranteed 100% uptime - system handles temporary unavailability via retry logic.
- **A-002**: JSON Schema format follows draft-07 specification. Pydantic models use Pydantic v2 syntax. Schema validation assumes valid JSON Schema structure. Users provide valid schemas - malformed schemas are rejected during validation (NFR-004). Schema format is stable and does not change during processing.
- **A-003**: Document formats (PDF, DOCX, HTML, TXT) can be converted to Markdown via `markitdown` library. Document conversion is handled by existing `ingest.py` module. Malformed documents are rare but handled gracefully. Document conversion failures are handled by existing `ingest.py` module - this feature assumes conversion succeeds or is handled upstream. Supported document formats are: PDF, DOCX, HTML, TXT, ZIP archives containing these formats, TAR/TAR.GZ archives containing these formats.
- **A-004**: User environment has Python 3.12+ installed, network access for cloud API providers, and sufficient disk space for JSONL output files. Ollama provider requires local installation but no network access. User has write permissions for output directory. Disk space is sufficient for JSONL files (approximate size: 1KB per document + extraction data). Network access is required for OpenAI and Gemini providers (HTTPS connections).
- **A-005**: `langextract.visualize()` API is stable and expects JSONL format as documented. Visualization library handles HTML generation correctly. Visualization failures are handled gracefully without breaking extraction workflow. `langextract.visualize()` API contract is stable (input format, output format) and does not change during v0 lifecycle. Visualization library is installed and available (`langextract>=0.1.0`).
- **A-006**: Existing codebase modules (`schema_io`, `ingest`, `models`, `aggregate`, `ui`) remain stable and compatible. Integration points are well-defined and do not change during feature implementation. Existing modules continue to function as documented. No breaking changes to existing module APIs during feature implementation.

## Dependencies

- **D-001**: External dependencies with version requirements: `pydantic>=2` (schema validation, data models), `pydantic-ai>=0.0.14` (unified schema-based extraction with Ollama/OpenAI/Gemini), `langextract>=0.1.0` (visualization only, not used for extraction), `thefuzz>=0.22.0` (fuzzy string matching via `fuzz.ratio()`), `structlog>=25.4.0` (structured logging), `jsonschema>=4.0.0` (JSON Schema validation). See plan.md Technical Context section for complete dependency list. All dependencies are available via `uv` package manager. Version requirements are minimum versions - newer versions are acceptable if backward compatible.
- **D-002**: Existing modules: `schema_io.py` (schema validation and conversion, JSON Schema ↔ Pydantic), `ingest.py` (document processing and Markdown conversion, `CorpusDocument` model), `models.py` (existing Pydantic models: `Candidate`, `FieldResult`, `ExtractionResult`, `SourceRef`), `aggregate.py` (candidate aggregation and deduplication), `ui.py` (Gradio interface, UI components). New module `structured_extract.py` depends on these modules. Integration points: `schema_io.convert_json_schema_to_pydantic()`, `ingest.process_corpus()`, `models.CorpusDocument`. Existing modules remain unchanged (backward compatibility per FR-015).
- **D-003**: `langextract.visualize()` API dependency: Expects JSONL file input with format matching FR-005 specification. Input format: Each line is JSON object with `extractions` (array of extraction objects with `extraction_class`, `extraction_text`, `char_interval`, `alignment_status`), `text` (string, full document markdown), `document_id` (string, unique identifier). Output format: HTML string (if `output_html_path` is None) or writes to file (if `output_html_path` provided). API signature: `visualize_extractions(jsonl_path: str | Path, output_html_path: str | Path | None = None) -> str`. API stability assumed for v0 (no breaking changes expected).
- **D-004**: PydanticAI Agent API: Provides unified interface for Ollama, OpenAI, and Gemini providers. Accepts Pydantic model as `output_type` parameter. Returns structured data matching Pydantic model. Provider model strings: "ollama:{model_name}" (e.g., "ollama:llama3"), "openai:{model_name}" (e.g., "openai:gpt-4o"), "google-gla:{model_name}" (e.g., "google-gla:gemini-2.5-flash"). API stability assumed for v0 (no breaking changes expected). PydanticAI handles API key management, retry logic, and error handling internally (system uses PydanticAI's built-in capabilities).

## Requirements Traceability

### Functional Requirements to User Stories

- **FR-001, FR-002, FR-009** → User Story 1 (Extract Using Structured Outputs), User Story 2 (Provider Selection)
- **FR-003** → User Story 1 (Sequential processing, one API call per document)
- **FR-004, FR-006** → User Story 1 (Character interval finding and alignment)
- **FR-005, FR-011, FR-012** → User Story 3 (Visualize and Export JSONL Results)
- **FR-007** → User Story 2 (Error handling for API errors)
- **FR-008** → User Story 2 (API key configuration)
- **FR-010** → User Story 1 (Fuzzy matching configuration)
- **FR-013** → User Story 1 (Schema flattening for nested structures)
- **FR-014** → User Story 1, User Story 2, User Story 3 (UI integration)
- **FR-015** → All User Stories (Backward compatibility)
- **FR-016, FR-017** → User Story 1 (Deduplication and plurality vote)

### Non-Functional Requirements to Technical Considerations

- **NFR-001** → Error handling and retry strategies
- **NFR-002** → API costs and rate limiting
- **NFR-003** → Performance for large corpora (parallelization options)
- **NFR-004** → Schema complexity (nested objects, arrays)
- **NFR-005** → Handling large documents (token limits)
- **NFR-006** → User Story 1 (Plurality vote calculation)
- **NFR-007** → Performance for large corpora
- **NFR-008** → Performance for large corpora
- **NFR-009** → API key security and management
- **NFR-010** → Error handling and retry strategies (logging)
- **NFR-011** → Code quality and maintainability
- **NFR-012** → UI integration (accessibility)

### Acceptance Scenarios to Requirements

- **User Story 1, Acceptance Scenario 1** → FR-001, FR-002, FR-009
- **User Story 1, Acceptance Scenario 2** → FR-004, FR-005, FR-006, FR-013
- **User Story 1, Acceptance Scenario 3** → FR-011, FR-012
- **User Story 2, Acceptance Scenario 1** → FR-009
- **User Story 2, Acceptance Scenario 2** → FR-005 (same format regardless of provider)
- **User Story 2, Acceptance Scenario 3** → FR-007, NFR-001
- **User Story 3, Acceptance Scenario 1** → FR-011, FR-012
- **User Story 3, Acceptance Scenario 2** → FR-005
- **User Story 3, Acceptance Scenario 3** → FR-004, FR-006

## Questions to Resolve

1. ~~Should this be a separate extraction mode or integrated into the existing UI?~~ → **Resolved**: Integrated into existing Gradio UI as primary option (FR-014)
2. ~~How should API keys be managed (env vars, config file, UI input)?~~ → **Resolved**: Config file support (read from ~/.ctrlf/config.toml, with env var and UI override options) (FR-008)
3. ~~Should we support batch processing of multiple documents in a single API call?~~ → **Resolved**: One document per API call (sequential processing) (FR-003)
4. ~~How should we handle schema complexity beyond what the APIs support?~~ → **Resolved**: Validate before processing, reject incompatible schemas with clear errors (NFR-004)
5. ~~Should we add retry logic with exponential backoff?~~ → **Resolved**: Yes, specified in NFR-001 (exponential backoff with 3 max retries)
6. ~~How should we handle cost tracking/estimation?~~ → **Resolved**: Specified in NFR-002 (token usage logging, optional cost estimation helpers)
7. ~~Should this support the same disambiguation/consensus logic as the existing pipeline?~~ → **Resolved**: Users review all unique extractions, plurality vote provides a suggested form (FR-016, FR-017)

## Success Criteria

- **SC-001**: Successfully call PydanticAI Agent with Ollama, OpenAI, or Gemini providers and parse structured results. **Measurable**: (1) API calls return valid Pydantic model instances matching schema (no `ValidationError` exceptions), (2) No parsing errors (no `JSONDecodeError`, no `ValueError` from parsing), (3) Structured data matches expected schema format (all required fields present, types match schema), (4) All three providers (Ollama, OpenAI, Gemini) successfully return structured data. **Test**: Call API with each provider, verify returned data is valid Pydantic model instance, verify all schema fields present and correctly typed.

- **SC-002**: Generate valid JSONL files that can be visualized with `langextract.visualize()`. **Measurable**: (1) JSONL file passes format validation (each line is valid JSON, no syntax errors), (2) Contains required fields (each line has `extractions` array, `text` string, `document_id` string), (3) Can be successfully loaded by `langextract.visualize()` without errors (no exceptions, HTML generated), (4) JSONL format matches `langextract.visualize()` expectations exactly. **Test**: Generate JSONL file, validate format, call `langextract.visualize()`, verify HTML output generated successfully.

- **SC-003**: Accurately locate extractions in source documents using fuzzy matching. **Measurable**: (1) Character intervals are within document bounds (`start_pos >= 0`, `end_pos <= len(document_text)`, `end_pos > start_pos`), (2) Alignment_status correctly reflects match quality (`match_exact` for exact matches, `match_fuzzy` for similarity >= threshold, `no_match` for similarity < threshold), (3) Fuzzy matches have similarity >= threshold (verified via `fuzz.ratio()`), (4) Extraction text found in source document at specified intervals (substring match at `[start_pos, end_pos)`). **Test**: Extract from document, verify character intervals are valid, verify alignment_status matches actual match quality, verify extraction text matches source at specified interval.

- **SC-004**: Handle errors gracefully and provide useful feedback. **Measurable**: (1) All error types handled according to FR-007 specifications (rate limits retry with backoff, timeouts retry, invalid responses skip document, auth failures stop processing), (2) Error messages include actionable guidance (e.g., "API key missing, checked: UI → env → config file"), (3) Errors logged with structured fields (`document_id`, `provider`, `error_type`, `retry_attempt`), (4) Processing continues for non-fatal errors (rate limits, timeouts, invalid responses don't stop corpus processing). **Test**: Simulate each error type, verify handling matches FR-007, verify error messages are actionable, verify logging includes required fields, verify non-fatal errors don't stop processing.

- **SC-005**: Integrate into existing Gradio UI as the primary extraction option. **Measurable**: (1) New extraction method available in UI (visible, selectable), (2) Presented as primary/recommended option (default selection, listed first, or prominently displayed), (3) Users can select extraction method (dropdown, radio buttons, or tabs work correctly), (4) UI components function correctly (extraction runs, results display, no UI errors), (5) No UI regressions (existing UI features still work). **Test**: Launch UI, verify new extraction method visible and selectable, verify it's presented as primary option, verify extraction runs successfully, verify existing UI features still work.

- **SC-006**: Maintain backward compatibility with existing extraction pipeline. **Measurable**: (1) Existing `extract.py` functions unchanged and functional (same signatures, same return types, same behavior), (2) Existing data models unchanged (no schema changes to `Candidate`, `FieldResult`, `ExtractionResult`), (3) Existing UI components continue to work (all existing UI features function as before), (4) Both pipelines can coexist (users can select either extraction method), (5) No breaking changes to existing APIs (no function signature changes, no return type changes). **Test**: Run existing extraction pipeline, verify functions work as before, verify data models unchanged, verify UI components work, verify both pipelines available in UI.

- **SC-007**: Support local development with Ollama (no API keys required). **Measurable**: (1) Ollama provider works without API key configuration (no API key errors, extraction succeeds), (2) Local models can be used for extraction (e.g., "ollama:llama3" works), (3) No network access required for Ollama provider (works offline), (4) Extraction results match cloud provider quality (structured data returned, schema compliance). **Test**: Use Ollama provider without API keys, verify extraction succeeds, verify works offline, verify results are valid structured data.
