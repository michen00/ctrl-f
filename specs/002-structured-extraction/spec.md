# Feature Specification: Structured Extraction with OpenAI/Gemini API Integration

**Feature Branch**: `002-structured-extraction`
**Created**: 2025-01-27
**Status**: Draft
**Input**: User description: "Structured Extraction with OpenAI/Gemini API Integration"

## Feature Description

Add a new extraction pipeline that uses OpenAI and Gemini's native structured output capabilities to extract data from documents. This provides an alternative to the existing langextract-based extraction that leverages cloud APIs for potentially higher accuracy and better schema adherence.

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
- What happens when API keys are missing or invalid? → System should check for API keys at startup, provide clear error messages, and support configuration via environment variables or config file
- How does the system handle network timeouts? → System should implement timeout handling, retry logic, and continue processing other documents on timeout

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST support OpenAI API with structured outputs using `response_format={"type": "json_schema", "json_schema": {...}}`
- **FR-002**: System MUST support Gemini API with structured outputs using `response_schema` in generation config
- **FR-003**: System MUST process each document in the corpus individually, passing the schema with each API call
- **FR-004**: System MUST use fuzzy regex matching to locate extracted text in source documents and calculate character intervals
- **FR-005**: System MUST generate JSONL files compatible with `langextract.visualize()` format
- **FR-006**: System MUST track alignment status for each extraction (match_exact, match_fuzzy, no_match)
- **FR-007**: System MUST handle API errors gracefully (rate limits, timeouts, invalid responses) and continue processing other documents
- **FR-008**: System MUST support API key configuration via environment variables or config file
- **FR-009**: System MUST allow selection of provider (OpenAI vs Gemini) and model
- **FR-010**: System MUST support configurable fuzzy matching threshold
- **FR-011**: System MUST integrate with `langextract.visualize()` to generate HTML visualizations
- **FR-012**: System MUST support saving visualizations to file or returning as string
- **FR-013**: System MUST flatten nested schema structures for extraction (handle arrays and nested objects)
- **FR-014**: System MUST maintain non-interference with existing `extract.py` logic

### Non-Functional Requirements

- **NFR-001**: System SHOULD implement retry logic with exponential backoff for API calls
- **NFR-002**: System SHOULD provide cost tracking/estimation for API usage
- **NFR-003**: System SHOULD support parallel processing of documents when possible (respecting rate limits)
- **NFR-004**: System SHOULD validate schema compatibility with API capabilities before processing
- **NFR-005**: System SHOULD handle large documents by detecting token limits and splitting if necessary

## Technical Considerations

- API costs and rate limiting
- Handling large documents (token limits)
- Schema complexity (nested objects, arrays)
- Character alignment accuracy for fuzzy matches
- Performance for large corpora (parallelization options)
- API key security and management
- Error handling and retry strategies

## Questions to Resolve

1. Should this be a separate extraction mode or integrated into the existing UI?
2. How should API keys be managed (env vars, config file, UI input)?
3. Should we support batch processing of multiple documents in a single API call?
4. How should we handle schema complexity beyond what the APIs support?
5. Should we add retry logic with exponential backoff?
6. How should we handle cost tracking/estimation?
7. Should this support the same disambiguation/consensus logic as the existing pipeline?

## Success Criteria

- Successfully call OpenAI API with structured outputs and parse results
- Successfully call Gemini API with structured outputs and parse results
- Generate valid JSONL files that can be visualized with `langextract.visualize()`
- Accurately locate extractions in source documents using fuzzy matching
- Handle errors gracefully and provide useful feedback
- Maintain non-interference with existing extraction pipeline
