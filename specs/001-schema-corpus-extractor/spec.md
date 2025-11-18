# Feature Specification: Schema-Grounded Corpus Extractor

**Feature Branch**: `001-schema-corpus-extractor`
**Created**: 2025-11-12
**Status**: Draft
**Input**: User description: "Schema‑Grounded Corpus Extractor — Spec v0.1"

## Clarifications

### Session 2025-11-12

- Q: What security and privacy requirements should the system enforce for document processing and data storage? → A: Local processing only, no network transmission; optional PII redaction in UI previews; no authentication required (single-user)
- Q: Should v0 support nested schemas (objects/arrays within objects), or only flat schemas with primitive types? → A: Flat schemas only - all fields must be primitive types (string, number, date, etc.) or arrays of primitives
- Q: How should the system present errors to users during corpus processing and extraction? → A: Continue processing other documents/fields, show warnings inline, display error summary at end with failed items listed
- Q: What specific thresholds should determine consensus vs disagreement for candidate values? → A: Confidence ≥0.75 and margin ≥0.20 over next candidate
- Q: For corpora exceeding typical size (e.g., >100 documents), should processing be incremental/streaming or batch with wait? → A: Batch processing with progress indicator - process all documents, show progress, allow cancellation, display results when complete

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Extract Structured Data from Documents (Priority: P1)

A user needs to extract structured information from a collection of documents (PDFs, Word docs, HTML, etc.) based on a defined schema. They provide a schema (either as JSON Schema or a Pydantic model) and a corpus of documents. The system processes the documents, identifies candidate values for each schema field, and presents them for review. The user can select the correct values, view source locations, and save the extracted record with full provenance.

**Why this priority**: This is the core value proposition - extracting structured data from unstructured documents with schema validation and provenance tracking. Without this, the system has no purpose.

**Independent Test**: Can be fully tested by providing a simple schema (e.g., name, email, date) and a small corpus (2-3 documents) containing the target fields. The system should extract candidates, allow selection, and save a validated record. This delivers immediate value: structured data extraction with source tracking.

**Acceptance Scenarios**:

1. **Given** a user has a JSON Schema defining fields (name, email, date) and a corpus of 3 PDF documents, **When** they upload both and run extraction, **Then** the system displays candidate values for each field with source locations visible
2. **Given** extraction results show multiple candidate values for a field, **When** the user selects one value and views its source, **Then** the system displays the exact document location (page/line) and surrounding context snippet
3. **Given** the user has reviewed and selected values for all fields, **When** they submit the form, **Then** the system validates the record against the schema and saves it with full provenance to persistent storage

---

### User Story 2 - Handle Schema Variations and Custom Values (Priority: P2)

A user needs flexibility in how they define their schema and may need to provide values not found in the corpus. They can provide either JSON Schema or Pydantic model code. When reviewing candidates, they can enter custom values that weren't extracted from documents. The system handles both schema formats and validates custom inputs.

**Why this priority**: Schema flexibility enables users to work with their existing schemas, and custom value entry handles cases where extraction misses valid data or users need to override results.

**Independent Test**: Can be tested by providing a Pydantic model instead of JSON Schema, and by entering a custom value in the "Other" field during review. The system should accept both schema formats and validate custom values. This delivers value: schema format flexibility and manual override capability.

**Acceptance Scenarios**:

1. **Given** a user has a Pydantic model class definition, **When** they upload it as the schema, **Then** the system accepts it and processes extraction using the same workflow as JSON Schema
2. **Given** extraction found no candidates for a required field, **When** the user selects "Other" and enters a custom value, **Then** the system validates the custom value against the field type and accepts it if valid
3. **Given** a user provides an invalid schema (malformed JSON or invalid Pydantic code), **When** they attempt to run extraction, **Then** the system displays clear error messages indicating what's wrong with the schema

---

### User Story 3 - Resolve Disagreements and View Provenance (Priority: P3)

A user encounters situations where multiple documents contain conflicting values for the same field, or where confidence is low. They need to see all candidates with their confidence scores, view source context for each, and make informed decisions. The system highlights disagreements and provides tools to compare sources.

**Why this priority**: Real-world extraction often produces conflicts. Users need transparency into why candidates differ and tools to resolve them confidently. This delivers value: trust and accuracy through provenance visibility.

**Independent Test**: Can be tested by providing a corpus where the same field appears with different values in different documents. The system should show all candidates, mark disagreements, allow source comparison, and save the user's chosen resolution. This delivers value: conflict resolution with full transparency.

**Acceptance Scenarios**:

1. **Given** extraction found 3 different values for a date field from 3 different documents, **When** the user views the review interface, **Then** all 3 candidates are shown with their confidence scores and source indicators
2. **Given** a user is comparing two conflicting candidates, **When** they click "View source" for each, **Then** the system displays side-by-side context snippets showing where each value was found
3. **Given** the system detected low consensus (multiple similar-confidence candidates), **When** the user reviews the field, **Then** the field is visually flagged as needing resolution and no value is pre-selected

---

### Edge Cases

- What happens when a document cannot be converted to Markdown (corrupted file, unsupported format)? → System continues processing other documents, shows warning inline, includes failed document in error summary at end with reason (corrupted/unsupported format)
- How does the system handle documents with no extractable content matching the schema fields? → Document is processed successfully but produces no candidates; no error shown, field results remain empty (handled by null policy)
- What happens when extraction finds no candidates for any field in the schema? → No error; all fields show empty results, user can proceed to review interface and enter custom values or accept empty arrays per null policy
- How does the system handle very large corpora (hundreds of documents) - does it process incrementally or require waiting? → Batch processing with progress indicator; process all documents before showing results, display progress during processing, allow user to cancel operation, show results when 100% complete
- What happens when a user provides a schema with nested objects or arrays? → Not supported in v0; system must reject nested schemas with clear error message indicating only flat schemas (primitive types or arrays of primitives) are supported
- How does the system handle fields that appear multiple times in the same document? → Collect all instances and create separate candidates; when there are disagreements, we need to know which sources say what
- What happens when source location information (page/line) is unavailable for a document format? → When source info is missing, we still need to provide the best context we can that is usable for the user. We should fall back to something sensible when appropriate (e.g., char-range, document-level location, or section heading)
- How does the system handle special characters, encoding issues, or non-text content (images, tables) in documents? → Special characters are preserved as-is (markitdown handles UTF-8 encoding). Encoding issues are handled by markitdown; if conversion fails due to encoding, treat as conversion error (covered by existing error handling). Images: Extract alt text if available in markdown, otherwise skip (cannot extract text from image content). Tables: Extract text from markdown table cells (markitdown converts tables to markdown format)

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST accept schema input in both JSON Schema format and Pydantic model code format (flat schemas only - all fields must be primitive types or arrays of primitives; nested objects/arrays not supported in v0)
- **FR-002**: System MUST accept a corpus of documents in multiple formats (PDF, DOCX, HTML, TXT, etc.) via file upload (directory, zip, or tar archive)
- **FR-003**: System MUST convert all supported document formats to Markdown while preserving source location mapping (file, page, line, or character range)
- **FR-004**: System MUST extract candidate values for each schema field from the corpus with zero fabrication (all candidates must be grounded in actual document content)
- **FR-005**: System MUST provide provenance for each candidate value including: source document identifier, file path, location (page/line or char range), and surrounding context snippet
- **FR-006**: System MUST normalize and deduplicate candidate values using similarity matching to group near-duplicates
- **FR-007**: System MUST compute confidence scores for each candidate and identify consensus when one candidate has confidence ≥0.75 and margin ≥0.20 over the next highest candidate
- **FR-008**: System MUST flag fields with disagreements when no single candidate meets consensus thresholds (confidence <0.75 or margin <0.20)
- **FR-009**: System MUST provide a review interface where users can see all candidates per field, select one value, or enter a custom value
- **FR-010**: System MUST allow users to view source context (snippet, metadata) for any candidate value on demand
- **FR-011**: System MUST validate all user-selected values against the schema before saving (type checking, required fields, array constraints)
- **FR-012**: System MUST save resolved records to persistent storage with: resolved field values, provenance (source references per field), and audit trail (timestamp, run ID, configuration)
- **FR-013**: System MUST handle extraction errors gracefully - continue processing other documents/fields when one fails, show warnings inline during processing, and display a summary of errors at completion with failed items listed (document names, field names, error types)
- **FR-014**: System MUST support schema fields as arrays (multiple values per field) in the final resolved record
- **FR-015**: System MUST allow users to configure null policy (empty list vs explicit null placeholder) for fields with no candidates
- **FR-016**: System MUST provide progress indicators during corpus processing and extraction phases (document count, percentage complete, estimated time remaining [best-effort]) and allow users to cancel processing operations
- **FR-017**: System MUST allow users to filter and search fields in the review interface (e.g., show only unresolved fields)
- **FR-018**: System MUST export saved records in JSON format for download
- **FR-019**: System MUST process all documents locally with no network transmission of document content or extracted data
- **FR-020**: System MUST provide optional PII redaction in UI previews (user-configurable patterns for sensitive data masking)
- **FR-021**: System MUST operate as a single-user application with no authentication or access control requirements

### Key Entities _(include if feature involves data)_

- **Schema**: Defines the structure of data to extract. Contains field definitions with names, types, descriptions, and constraints. Can be provided as JSON Schema or Pydantic model code.

- **Corpus**: Collection of source documents in various formats (PDF, DOCX, HTML, TXT, etc.). Each document has an original file path and will be converted to Markdown with source mapping.

- **Source Reference**: Points to the exact location where a value was found. Contains document ID, file path, location (page/line or char range), context snippet, and metadata (mtime, converter, checksum).

- **Candidate Value**: A potential value for a schema field extracted from the corpus. Contains the raw value, normalized form, confidence score (0-1), and list of source references where it was found.

- **Field Result**: Aggregation of all candidates for a single schema field. Contains field name, list of candidates (after deduplication), and optional consensus candidate if one is clearly preferred.

- **Extraction Result**: Complete output from the extraction phase. Contains field results for all schema fields, schema version, run ID, and timestamp.

- **Resolution**: User's decision for a field. Contains field name, chosen value, optional source document ID and location, and flag indicating if it was custom input.

- **Persisted Record**: Final saved record after user resolution. Contains record ID, resolved field values (arrays), provenance (source references per field), and audit information (who, when, version, config).

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Users can complete a full extraction workflow (upload schema + corpus, review candidates, save record) in under 10 minutes for a corpus of 10 documents with 5 schema fields
- **SC-002**: System processes and converts documents to Markdown at a rate of at least 5 documents per minute for standard formats (PDF, DOCX, HTML)
- **SC-003**: Extraction identifies at least 80% of schema fields present in documents (recall metric - fields that exist in docs are found)
- **SC-004**: 95% of saved records pass schema validation on first submission (users don't need to fix validation errors repeatedly)
- **SC-005**: Users can view source provenance for any candidate value within 2 seconds of clicking "View source"
- **SC-006**: System handles corpora of up to 100 documents without requiring pagination or batching in the review interface
- **SC-007**: Zero fabricated values in extraction results (100% of candidates have verifiable source locations)
- **SC-008**: Users successfully resolve disagreements (select among conflicting candidates) in 90% of cases without needing to re-run extraction
