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

A draft implementation already exists in `src/ctrlf/app/structured_extract.py` that provides:

- Data models (`ExtractionRecord`, `JSONLLine`) matching the JSONL format required by `langextract.visualize()`
- Fuzzy character interval finding using `thefuzz` to locate extracted text in documents
- JSONL file writing and visualization wrapper
- Schema flattening for nested structures

However, the API integration is currently a placeholder (`_call_structured_extraction_api()`) and needs to be fully implemented.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Extract Using OpenAI Structured Outputs (Priority: P1)

A user wants to extract structured data from documents using OpenAI's structured output capabilities for better accuracy and schema adherence. They provide a schema and corpus, and the system processes each document individually using OpenAI's API with JSON Schema constraints, then generates a JSONL file for visualization.

**Why this priority**: This is the core value proposition - leveraging cloud APIs for potentially higher-quality extraction with native structured output support.

**Independent Test**: Can be tested by providing a simple schema and a small corpus (2-3 documents), calling OpenAI API with structured outputs, and verifying the JSONL output format matches expectations.

**Acceptance Scenarios**:

1. **Given** a user has a JSON Schema and corpus documents, **When** they run structured extraction with OpenAI provider, **Then** the system calls OpenAI API with `response_format` containing the schema and returns structured JSON
2. **Given** OpenAI API returns structured data, **When** the system processes the response, **Then** it generates valid JSONL lines with character intervals and alignment status
3. **Given** extraction completes successfully, **When** the user requests visualization, **Then** the system generates HTML visualization using `langextract.visualize()`

---

### User Story 2 - Extract Using Gemini Structured Outputs (Priority: P1)

A user wants to use Gemini API as an alternative to OpenAI for structured extraction. The system should support both providers with similar functionality.

**Why this priority**: Provider flexibility is important for users who prefer different APIs or have different cost/performance requirements.

**Independent Test**: Can be tested by switching provider to "gemini" and verifying API calls use Gemini's `response_schema` format correctly.

**Acceptance Scenarios**:

1. **Given** a user selects Gemini as the provider, **When** they run structured extraction, **Then** the system calls Gemini API with `response_schema` in generation config
2. **Given** Gemini API returns structured data, **When** the system processes the response, **Then** it generates the same JSONL format as OpenAI extractions
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

- What happens when API rate limits are hit? → System should implement retry logic with exponential backoff, continue processing other documents, and log rate limit errors clearly
- How does the system handle documents that exceed token limits? → System should detect token limits, split documents if possible, or skip with clear error message
- What happens when fuzzy matching cannot find a character interval? → System should mark alignment_status as "no_match" and set char_interval to {0, 0}, but still include the extraction
- How does the system handle schema complexity beyond what APIs support? → System should validate schema compatibility before API calls and provide clear error messages for unsupported features
- What happens when API keys are missing or invalid? → System should check for API keys in precedence order (UI input → env vars → config file), provide clear error messages indicating which method was checked, and guide users to configure keys via the preferred method
- How does the system handle network timeouts? → System should implement timeout handling, retry logic, and continue processing other documents on timeout

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST support OpenAI API with structured outputs using `response_format={"type": "json_schema", "json_schema": {...}}`
- **FR-002**: System MUST support Gemini API with structured outputs using `response_schema` in generation config
- **FR-003**: System MUST process each document in the corpus individually, making one API call per document (no batching multiple documents into single API requests)
- **FR-004**: System MUST use fuzzy regex matching to locate extracted text in source documents and calculate character intervals
- **FR-005**: System MUST generate JSONL files compatible with `langextract.visualize()` format
- **FR-006**: System MUST track alignment status for each extraction (match_exact, match_fuzzy, no_match)
- **FR-016**: System MUST deduplicate extractions to show all unique values for user review
- **FR-017**: System MUST provide a suggested value based on plurality vote (most common extraction value) but MUST NOT auto-select it
- **FR-007**: System MUST handle API errors gracefully (rate limits, timeouts, invalid responses) and continue processing other documents
- **FR-008**: System MUST support API key configuration with precedence order: UI input (highest priority) → environment variables (OPENAI_API_KEY, GOOGLE_API_KEY) → config file (~/.ctrlf/config.toml) (lowest priority)
- **FR-009**: System MUST allow selection of provider (OpenAI vs Gemini) and model
- **FR-010**: System MUST support configurable fuzzy matching threshold
- **FR-011**: System MUST integrate with `langextract.visualize()` to generate HTML visualizations
- **FR-012**: System MUST support saving visualizations to file or returning as string
- **FR-013**: System MUST flatten nested schema structures for extraction (handle arrays and nested objects)
- **FR-014**: System MUST integrate into existing Gradio UI as the primary extraction option
- **FR-015**: System MUST maintain backward compatibility with existing `extract.py` logic (can coexist but new pipeline is primary)

### Non-Functional Requirements

- **NFR-001**: System SHOULD implement retry logic with exponential backoff for API calls
- **NFR-002**: System SHOULD provide cost tracking/estimation for API usage
- **NFR-003**: System MAY support parallel processing of documents in future (respecting rate limits), but v0 uses sequential processing (one API call per document)
- **NFR-004**: System MUST validate schema compatibility with API capabilities before processing and MUST reject incompatible schemas with clear, actionable error messages (fail fast, do not attempt API calls with invalid schemas)
- **NFR-005**: System SHOULD handle large documents by detecting token limits and splitting if necessary
- **NFR-006**: System SHOULD calculate plurality vote (most common extraction value) for suggestion purposes

## Technical Considerations

- API costs and rate limiting
- Handling large documents (token limits)
- Schema complexity (nested objects, arrays)
- Character alignment accuracy for fuzzy matches
- Performance for large corpora (parallelization options)
- API key security and management
- Error handling and retry strategies

## Questions to Resolve

1. ~~Should this be a separate extraction mode or integrated into the existing UI?~~ → **Resolved**: Integrated into existing Gradio UI as primary option
2. ~~How should API keys be managed (env vars, config file, UI input)?~~ → **Resolved**: Config file support (read from ~/.ctrlf/config.toml, with env var and UI override options)
3. ~~Should we support batch processing of multiple documents in a single API call?~~ → **Resolved**: One document per API call (sequential processing)
4. ~~How should we handle schema complexity beyond what the APIs support?~~ → **Resolved**: Validate before processing, reject incompatible schemas with clear errors
5. ~~Should we add retry logic with exponential backoff?~~ → **Resolved**: Yes, specified in NFR-001 (exponential backoff with 3 max retries)
6. ~~How should we handle cost tracking/estimation?~~ → **Resolved**: Specified in NFR-002 (token usage logging, optional cost estimation helpers)
7. ~~Should this support the same disambiguation/consensus logic as the existing pipeline?~~ → **Resolved**: Users review all unique extractions, plurality vote provides a suggested form

## Success Criteria

- Successfully call OpenAI API with structured outputs and parse results
- Successfully call Gemini API with structured outputs and parse results
- Generate valid JSONL files that can be visualized with `langextract.visualize()`
- Accurately locate extractions in source documents using fuzzy matching
- Handle errors gracefully and provide useful feedback
- Integrate into existing Gradio UI as the primary extraction option
- Maintain backward compatibility with existing extraction pipeline
