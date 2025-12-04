# Implementation Plan: Schema-Grounded Corpus Extractor

**Branch**: `001-schema-corpus-extractor` | **Date**: 2025-11-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-schema-corpus-extractor/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a schema-driven document extraction application that accepts JSON Schema or Pydantic models and a corpus of documents, extracts candidate values for each schema field using PydanticAI (with Ollama/OpenAI/Gemini) with full provenance tracking, presents candidates for user review and resolution via Gradio UI, and persists validated records to TinyDB with audit trails. The system enforces zero fabrication (all candidates must be grounded in source documents) and provides comprehensive source location tracking.

**Note**: Previously used langextract for extraction, but this has been replaced with PydanticAI. langextract is now only used for visualization.

## Technical Context

**Language/Version**: Python 3.12+ (per project requirements)
**Primary Dependencies**:

- pydantic>=2 (schema validation, data models)
- gradio (web UI framework)
- tinydb (local JSON-based database)
- markitdown (document to Markdown conversion)
- pydantic-ai (unified schema-based extraction with Ollama/OpenAI/Gemini)
- langextract (visualization only, not used for extraction)
- thefuzz (fuzzy string matching for deduplication, uses rapidfuzz under the hood)
- python-slugify (ID generation)
- structlog (structured logging)
- jsonschema (JSON Schema validation)

**Storage**: TinyDB (local JSON database) for persisted records; in-memory for processing
**Testing**: pytest (unit, integration, contract tests)
**Target Platform**: Cross-platform (Linux, macOS, Windows) - local desktop/web application
**Project Type**: Single project (Python application with Gradio web UI)
**Performance Goals**:

- Process 5 documents/minute for conversion (SC-002)
- View source provenance within 2 seconds (SC-005)
- Complete full workflow in under 10 minutes for 10 docs with 5 fields (SC-001)

**Constraints**:

- Local processing only (no network transmission) - FR-019
- Single-user application (no authentication) - FR-021
- Flat schemas only in v0 (no nested objects/arrays) - FR-001
- Zero fabrication requirement (all candidates must have source spans) - FR-004

**Scale/Scope**:

- Support up to 100 documents without pagination (SC-006)
- Handle corpora of hundreds of documents with batch processing (clarification Q5)
- Single extraction run per session

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### I. Test-First Development (NON-NEGOTIABLE)

✅ **COMPLIANT**: All modules will follow TDD. Tests written before implementation for:

- Schema coercion and validation logic
- Extraction and aggregation algorithms
- Normalization and deduplication
- UI components and callbacks
- Storage operations

### II. Type Safety & Static Analysis

✅ **COMPLIANT**:

- All Pydantic models provide type safety
- All functions will have type hints
- mypy strict mode will be enforced
- ruff and pylint will validate code quality

### III. CLI Interface Standard

⚠️ **PARTIAL COMPLIANCE**:

- Primary interface is Gradio web UI (not CLI)
- **Justification**: Gradio provides interactive review interface essential for candidate selection and source viewing. CLI mode deferred to v1 per roadmap.
- **Alternative considered**: Pure CLI with JSON I/O - rejected because review/resolution workflow requires interactive UI for viewing sources and comparing candidates
- **Mitigation**: Core extraction logic will be CLI-callable; Gradio wraps it. Future v1 will add standalone CLI mode.

### IV. Data Integrity & Validation

✅ **COMPLIANT**:

- Pydantic models validate all data structures
- Schema validation before extraction
- Record validation before persistence
- Type checking for all field values

### V. Observability & Logging

✅ **COMPLIANT**:

- structlog configured for structured logging
- Log levels: INFO (milestones), DEBUG (per-candidate), WARN (skips), ERROR (fatal)
- Contextual information in logs (run_id, field_name, doc_id)

## Project Structure

### Documentation (this feature)

```text
specs/001-schema-corpus-extractor/
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
│       ├── server.py          # Gradio app entrypoint
│       ├── ui.py              # Gradio interface components
│       ├── models.py          # Pydantic data models
│       ├── schema_io.py       # Schema validation and conversion
│       ├── ingest.py          # markitdown wrapper and source mapping
│       ├── extract.py         # PydanticAI-based extraction (replaced langextract)
│       ├── aggregate.py       # Candidate clustering, normalization, consensus
│       ├── storage.py         # TinyDB adapter
│       └── logging_conf.py    # structlog configuration

tests/
├── unit/
│   ├── test_schema_io.py
│   ├── test_aggregate.py
│   ├── test_ingest.py
│   ├── test_extract.py
│   └── test_storage.py
├── integration/
│   └── test_end_to_end.py
└── contract/
    └── test_data_models.py
```

**Structure Decision**: Single project structure chosen. The application is a Python package (`ctrlf.app`) with modular components. Gradio UI serves as the primary interface, but core logic is separated into modules that could be CLI-callable in future versions. Test structure follows standard pytest conventions with unit, integration, and contract test separation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation                                    | Why Needed                                                                                                                                                                                  | Simpler Alternative Rejected Because                                                                                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gradio web UI instead of CLI (Principle III) | Interactive review interface essential for candidate selection, source viewing, and conflict resolution. Users need to see source context, compare candidates, and make informed decisions. | Pure CLI with JSON I/O rejected because review/resolution workflow requires visual comparison of candidates and source snippets. Batch mode without review would reduce accuracy and user trust. |

## Phase 0: Research Complete

All technology decisions documented in `research.md`. Key decisions:

- markitdown for document conversion
- PydanticAI for field extraction (replaced langextract)
- thefuzz for deduplication (clean API wrapper around rapidfuzz)
- TinyDB for persistence
- Gradio for UI
- Pydantic v2 + jsonschema for validation

No NEEDS CLARIFICATION items remain - all technical choices are resolved.

## Phase 1: Design Complete

### Data Model

Complete data model documented in `data-model.md` with:

- Core entities: SourceRef, Candidate, FieldResult, ExtractionResult, Resolution, PersistedRecord
- Schema extension pattern
- Data flow and validation points

### Function Contracts

Function contracts documented in `contracts/function-contracts.md` covering:

- Schema I/O operations
- Document ingestion
- Field extraction
- Candidate aggregation
- Storage operations
- UI components

### Quickstart Guide

User-facing quickstart guide created in `quickstart.md` with:

- Installation instructions
- Basic usage workflow
- Configuration options
- Troubleshooting guide

### Agent Context

Agent context updated with Python 3.12+ and TinyDB technologies.

## Constitution Check (Post-Design)

Re-evaluated after Phase 1 design:

### I. Test-First Development (Post-Design)

✅ **COMPLIANT**: All modules have clear testable contracts. TDD approach defined for each component.

### II. Type Safety & Static Analysis (Post-Design)

✅ **COMPLIANT**: All function contracts include type hints. Pydantic models provide runtime type safety.

### III. CLI Interface Standard (Post-Design)

⚠️ **PARTIAL COMPLIANCE**: Justified in Complexity Tracking. Core logic is modular and CLI-callable; Gradio wraps it.

### IV. Data Integrity & Validation (Post-Design)

✅ **COMPLIANT**: Validation points defined at every stage. Pydantic models enforce data integrity.

### V. Observability & Logging (Post-Design)

✅ **COMPLIANT**: structlog configuration defined. Logging points identified in function contracts.

## Next Steps

Ready for `/speckit.tasks` to generate implementation task breakdown.

**Generated Artifacts**:

- ✅ `research.md` - Technology decisions and patterns
- ✅ `data-model.md` - Complete data model specification
- ✅ `contracts/function-contracts.md` - Function API contracts
- ✅ `quickstart.md` - User guide
- ✅ Agent context updated
