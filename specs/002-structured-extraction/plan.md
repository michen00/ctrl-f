# Implementation Plan: Structured Extraction with OpenAI/Gemini API Integration

**Branch**: `002-structured-extraction` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-structured-extraction/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add a new extraction pipeline that uses OpenAI and Gemini's native structured output capabilities to extract data from documents. This provides an alternative to the existing langextract-based extraction, leveraging cloud APIs for potentially higher accuracy and better schema adherence. The implementation will integrate OpenAI and Gemini APIs with structured outputs, use fuzzy matching to locate extractions in source documents, generate JSONL files compatible with `langextract.visualize()`, and maintain non-interference with existing extraction logic.

## Technical Context

**Language/Version**: Python 3.12+ (per project requirements)
**Primary Dependencies**:

- pydantic>=2 (schema validation, data models)
- openai>=1.0.0 (OpenAI API client with structured outputs support)
- google-genai>=1.3.0 (already in dependencies, Gemini API client)
- langextract>=0.1.0 (visualization integration)
- thefuzz>=0.22.0 (fuzzy string matching for character alignment)
- structlog>=25.4.0 (structured logging)
- jsonschema>=4.0.0 (JSON Schema validation)

**Storage**: N/A (outputs JSONL files, no persistent storage required for this feature)
**Testing**: pytest (unit, integration, contract tests)
**Target Platform**: Cross-platform (Linux, macOS, Windows) - local desktop/web application
**Project Type**: Single project (Python application, extends existing codebase)
**Performance Goals**:

- Process documents at API rate limits (exponential backoff retry with 3 max retries)
- Character alignment within 2 seconds per document
- Generate JSONL files efficiently for large corpora

**Constraints**:

- Requires network access for API calls (different from existing local-only constraint)
- API key management and security (environment variables: OPENAI_API_KEY, GOOGLE_API_KEY)
- Rate limiting and cost considerations (retry logic, token usage logging)
- Token limits for large documents (detect limits, split documents if needed, skip with error if too large)
- Schema compatibility with API capabilities (validate before API calls, support nested structures, flatten for extraction)

**Scale/Scope**:

- Support same corpus sizes as existing extraction (up to 100 documents)
- Handle API rate limits gracefully
- Support both OpenAI and Gemini providers
- Maintain backward compatibility with existing extraction pipeline

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Test-First Development (NON-NEGOTIABLE)

✅ **COMPLIANT**: All modules will follow TDD. Tests written before implementation for:

- API client integration (mocked API calls)
- Character interval finding and fuzzy matching
- JSONL generation and validation
- Schema flattening logic
- Error handling and retry logic
- Visualization integration

### II. Type Safety & Static Analysis

✅ **COMPLIANT**:

- All Pydantic models provide type safety (ExtractionRecord, JSONLLine)
- All functions will have type hints
- mypy strict mode will be enforced
- ruff and pylint will validate code quality

### III. CLI Interface Standard

⚠️ **PARTIAL COMPLIANCE**:

- Primary interface is Gradio web UI (not CLI) - same as existing feature
- **Justification**: This feature extends the existing Gradio-based extraction pipeline. CLI mode deferred to v1 per existing roadmap.
- **Alternative considered**: Pure CLI with JSON I/O - rejected because this integrates with existing UI workflow
- **Mitigation**: Core extraction logic will be modular and testable; can be CLI-callable in future

### IV. Data Integrity & Validation

✅ **COMPLIANT**:

- Pydantic models validate all data structures (ExtractionRecord, JSONLLine)
- Schema validation before API calls
- JSONL format validation
- Type checking for all field values

### V. Observability & Logging

✅ **COMPLIANT**:

- structlog configured for structured logging
- Log levels: INFO (API calls, milestones), DEBUG (per-extraction), WARN (API errors, retries), ERROR (fatal)
- Contextual information in logs (document_id, provider, model, extraction_index)

## Project Structure

### Documentation (this feature)

```text
specs/002-structured-extraction/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── ctrlf/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── structured_extract.py    # New module (draft exists)
│       ├── server.py                # Existing Gradio app entrypoint
│       ├── ui.py                    # Existing Gradio interface (may extend)
│       ├── models.py                # Existing Pydantic models
│       ├── schema_io.py             # Existing schema validation
│       ├── ingest.py                # Existing document processing
│       ├── extract.py               # Existing langextract-based extraction
│       ├── aggregate.py             # Existing candidate aggregation
│       ├── storage.py               # Existing TinyDB adapter
│       └── logging_conf.py          # Existing structlog configuration

tests/
├── unit/
│   ├── test_structured_extract.py   # New tests for structured extraction
│   └── [existing test files]
├── integration/
│   ├── test_structured_extraction_e2e.py  # New E2E tests
│   └── [existing test files]
└── contract/
    └── [existing contract tests]
```

**Structure Decision**: Single project structure. The new `structured_extract.py` module extends the existing codebase without modifying existing extraction logic. It reuses existing modules (schema_io, ingest, models) and maintains separation from `extract.py`. Test structure follows existing pytest conventions with unit, integration, and contract test separation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations requiring justification. Partial compliance on CLI interface standard is consistent with existing feature (001-schema-corpus-extractor) and justified by Gradio UI requirements.

## Phase 0: Research Complete

All technology decisions documented in `research.md`. Key decisions:

- `openai>=1.0.0` for OpenAI API client with structured outputs
- `google-genai>=1.3.0` (already in dependencies) for Gemini API client
- Exponential backoff retry strategy (3 retries, handles rate limits and timeouts)
- Token limit detection and document splitting strategy
- Schema compatibility validation and flattening approach
- Environment variable-based API key management
- Token usage logging and optional cost estimation

No NEEDS CLARIFICATION items remain - all technical choices are resolved.

## Phase 1: Design Complete

### Data Model

Complete data model documented in `data-model.md` with:

- Core entities: ExtractionRecord, JSONLLine
- Supporting structures: API Configuration, Character Interval, Alignment Status
- Integration with existing models (CorpusDocument, schema models)
- Data flow from input to JSONL output
- Validation points and error handling

### Function Contracts

Function contracts documented in `contracts/function-contracts.md` covering:

- Character interval finding with fuzzy matching
- Structured extraction API calls (OpenAI and Gemini)
- Schema flattening for nested structures
- Main extraction orchestration
- JSONL file writing
- Visualization integration

### Quickstart Guide

User-facing quickstart guide created in `quickstart.md` with:

- Installation instructions
- API key configuration
- Basic usage examples (OpenAI and Gemini)
- Configuration options
- Output format documentation
- Troubleshooting guide

### Agent Context

Agent context will be updated with OpenAI and Gemini API integration technologies.

## Constitution Check (Post-Design)

Re-evaluated after Phase 1 design:

### I. Test-First Development (Post-Design)

✅ **COMPLIANT**: All modules have clear testable contracts. TDD approach defined for each component. API calls will be mocked in tests.

### II. Type Safety & Static Analysis (Post-Design)

✅ **COMPLIANT**: All function contracts include type hints. Pydantic models (ExtractionRecord, JSONLLine) provide runtime type safety.

### III. CLI Interface Standard (Post-Design)

⚠️ **PARTIAL COMPLIANCE**: Justified in Complexity Tracking. Core logic is modular and testable; can be CLI-callable in future.

### IV. Data Integrity & Validation (Post-Design)

✅ **COMPLIANT**: Validation points defined at every stage. Pydantic models enforce data integrity. Schema validation before API calls.

### V. Observability & Logging (Post-Design)

✅ **COMPLIANT**: structlog configuration defined. Logging points identified in function contracts (API calls, retries, errors, token usage).

## Next Steps

Ready for `/speckit.tasks` to generate implementation task breakdown.

**Generated Artifacts**:

- ✅ `research.md` - Technology decisions and patterns
- ✅ `data-model.md` - Complete data model specification
- ✅ `contracts/function-contracts.md` - Function API contracts
- ✅ `quickstart.md` - User guide
- ✅ Agent context updated (Cursor IDE context file updated)
