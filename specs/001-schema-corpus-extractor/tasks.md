---
description: "Task list template for feature implementation"
---

# Tasks: Schema-Grounded Corpus Extractor

**Input**: Design documents from `/specs/001-schema-corpus-extractor/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included per TDD requirement (Constitution Principle I). All tests MUST be written first and verified to fail before implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths use `src/ctrlf/app/` structure per plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per implementation plan (src/ctrlf/app/, tests/unit/, tests/integration/, tests/contract/)
- [x] T002 Update pyproject.toml with dependencies: pydantic>=2, gradio, tinydb, markitdown, langextract, thefuzz, python-slugify, structlog, jsonschema
- [x] T003 [P] Configure linting and formatting tools (ruff, pylint, mypy) in pyproject.toml
- [x] T004 [P] Setup pytest configuration in pyproject.toml with coverage settings

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 [P] Create structlog configuration in src/ctrlf/app/logging_conf.py
- [x] T006 [P] Create base Pydantic models in src/ctrlf/app/models.py (SourceRef, Candidate, FieldResult, ExtractionResult, Resolution, PersistedRecord)
- [x] T007 Create error handling utilities in src/ctrlf/app/errors.py for graceful degradation
- [x] T008 [P] Create TinyDB storage adapter skeleton in src/ctrlf/app/storage.py
- [x] T009 [P] Create schema I/O module skeleton in src/ctrlf/app/schema_io.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Extract Structured Data from Documents (Priority: P1) 🎯 MVP

**Goal**: Core extraction workflow - accept schema and corpus, extract candidates, present for review, save validated record with provenance

**Independent Test**: Provide simple schema (name, email, date) and 2-3 PDF documents. System extracts candidates, allows selection, saves validated record with source tracking.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Unit test for SourceRef model validation in tests/unit/test_models.py
- [x] T011 [P] [US1] Unit test for Candidate model validation in tests/unit/test_models.py
- [x] T012 [P] [US1] Unit test for FieldResult model validation in tests/unit/test_models.py
- [x] T013 [P] [US1] Unit test for ExtractionResult model validation in tests/unit/test_models.py
- [x] T014 [P] [US1] Unit test for document conversion in tests/unit/test_ingest.py
- [x] T015 [P] [US1] Unit test for field extraction in tests/unit/test_extract.py
- [x] T016 [P] [US1] Unit test for candidate aggregation in tests/unit/test_aggregate.py
- [x] T017 [P] [US1] Unit test for storage operations in tests/unit/test_storage.py
- [x] T018 [US1] Integration test for full extraction workflow in tests/integration/test_end_to_end.py

### Implementation for User Story 1

- [x] T019 [P] [US1] Implement SourceRef model in src/ctrlf/app/models.py
- [x] T020 [P] [US1] Implement Candidate model in src/ctrlf/app/models.py
- [x] T021 [P] [US1] Implement FieldResult model in src/ctrlf/app/models.py
- [x] T022 [P] [US1] Implement ExtractionResult model in src/ctrlf/app/models.py
- [x] T023 [P] [US1] Implement Resolution model in src/ctrlf/app/models.py
- [x] T024 [P] [US1] Implement PersistedRecord model in src/ctrlf/app/models.py
- [x] T025 [US1] Implement convert_document_to_markdown function in src/ctrlf/app/ingest.py (depends on T005)
- [x] T026 [US1] Implement process_corpus function in src/ctrlf/app/ingest.py (depends on T025)
- [x] T027 [US1] Implement extract_field_candidates function in src/ctrlf/app/extract.py (depends on T019, T020)
- [x] T028 [US1] Implement run_extraction function in src/ctrlf/app/extract.py (depends on T027)
- [x] T029 [US1] Implement normalize_value function in src/ctrlf/app/aggregate.py (depends on T020)
- [x] T030 [US1] Implement deduplicate_candidates function in src/ctrlf/app/aggregate.py (depends on T020, T029)
- [x] T031 [US1] Implement detect_consensus function in src/ctrlf/app/aggregate.py (depends on T020)
- [x] T032 [US1] Implement aggregate_field_results function in src/ctrlf/app/aggregate.py (depends on T021, T029, T030, T031)
- [x] T033 [US1] Implement save_record function in src/ctrlf/app/storage.py (depends on T024)
- [x] T034 [US1] Implement export_record function in src/ctrlf/app/storage.py (depends on T033)
- [x] T035 [US1] Create upload interface in src/ctrlf/app/ui.py (depends on T026, T028)
- [x] T036 [US1] Create review interface in src/ctrlf/app/ui.py (depends on T032, T035)
- [x] T037 [US1] Implement show_source_context function in src/ctrlf/app/ui.py (depends on T019)
- [x] T038 [US1] Create main server entrypoint in src/ctrlf/app/server.py (depends on T035, T036)
- [x] T039 [US1] Add error handling and progress indicators to UI (depends on T007, T035, T036)
- [x] T040 [US1] Add validation before saving records (depends on T032, T033)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Handle Schema Variations and Custom Values (Priority: P2)

**Goal**: Support both JSON Schema and Pydantic model input, allow custom value entry during review, validate custom inputs

**Independent Test**: Provide Pydantic model instead of JSON Schema, enter custom value in "Other" field. System accepts both formats and validates custom values.

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T041 [P] [US2] Unit test for JSON Schema validation in tests/unit/test_schema_io.py
- [x] T042 [P] [US2] Unit test for JSON Schema to Pydantic conversion in tests/unit/test_schema_io.py
- [x] T043 [P] [US2] Unit test for Pydantic model import in tests/unit/test_schema_io.py
- [x] T044 [P] [US2] Unit test for schema extension (array coercion) in tests/unit/test_schema_io.py
- [x] T045 [P] [US2] Unit test for nested schema rejection in tests/unit/test_schema_io.py
- [x] T046 [US2] Integration test for Pydantic model workflow in tests/integration/test_end_to_end.py

### Implementation for User Story 2

- [x] T047 [US2] Implement validate_json_schema function in src/ctrlf/app/schema_io.py (depends on T009)
- [x] T048 [US2] Implement convert_json_schema_to_pydantic function in src/ctrlf/app/schema_io.py (depends on T047)
- [x] T049 [US2] Implement import_pydantic_model function in src/ctrlf/app/schema_io.py (depends on T009)
- [x] T050 [US2] Implement extend_schema function in src/ctrlf/app/schema_io.py (depends on T048, T049)
- [x] T051 [US2] Add schema format detection to upload interface in src/ctrlf/app/ui.py (depends on T035, T047, T049)
- [x] T052 [US2] Add "Other" text input option to review interface in src/ctrlf/app/ui.py (depends on T036)
- [x] T053 [US2] Add custom value validation in review interface in src/ctrlf/app/ui.py (depends on T052, T050)
- [x] T054 [US2] Add error messages for invalid schemas in src/ctrlf/app/ui.py (depends on T051, T007)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Resolve Disagreements and View Provenance (Priority: P3)

**Goal**: Display all candidates with confidence scores, highlight disagreements, enable source comparison, show source context on demand

**Independent Test**: Provide corpus with conflicting values. System shows all candidates, marks disagreements, allows source comparison, saves chosen resolution.

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T055 [P] [US3] Unit test for disagreement detection in tests/unit/test_aggregate.py
- [x] T056 [P] [US3] Unit test for confidence score computation in tests/unit/test_aggregate.py
- [x] T057 [US3] Integration test for disagreement resolution workflow in tests/integration/test_end_to_end.py

### Implementation for User Story 3

- [x] T058 [US3] Enhance detect_consensus to flag disagreements in src/ctrlf/app/aggregate.py (depends on T031)
- [x] T059 [US3] Add confidence score display to review interface in src/ctrlf/app/ui.py (depends on T036)
- [x] T060 [US3] Add visual flagging for fields with disagreements in src/ctrlf/app/ui.py (depends on T036, T058)
- [x] T061 [US3] Enhance show_source_context to support side-by-side comparison in src/ctrlf/app/ui.py (depends on T037)
- [x] T062 [US3] Add "View source" button functionality for each candidate in src/ctrlf/app/ui.py (depends on T037, T059)
- [x] T063 [US3] Add filter/search functionality for fields in review interface in src/ctrlf/app/ui.py (depends on T036)
- [x] T064 [US3] Ensure no pre-selection for fields with disagreements in src/ctrlf/app/ui.py (depends on T036, T058)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T065 [P] Add progress indicators with cancellation support in src/ctrlf/app/ui.py
- [x] T066 [P] Add error summary display at end of processing in src/ctrlf/app/ui.py
- [x] T067 [P] Add null policy configuration option in src/ctrlf/app/ui.py
- [x] T068 [P] Add JSON export functionality for saved records in src/ctrlf/app/ui.py (depends on T034)
- [x] T069 [P] Add field filtering (unresolved/flagged) to review interface in src/ctrlf/app/ui.py
- [ ] T070 [P] Update README.md with installation and usage instructions
- [x] T071 [P] Add docstrings to all public functions and classes
- [ ] T072 Code cleanup and refactoring
- [ ] T073 [P] Performance optimization across all modules (target SC-001: <10min workflow, SC-002: 5 docs/min, SC-005: <2s source view)
- [x] T074 [P] Additional unit tests for edge cases in tests/unit/ (multiple occurrences per document, missing location fallback, special characters/encoding/images/tables, SC-003 recall metrics, SC-004 validation pass rate)
- [ ] T075 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 UI components but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Enhances US1 UI components but independently testable

### Within Each User Story

- Tests (included per TDD) MUST be written and FAIL before implementation
- Models before services/functions
- Core functions before UI integration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all model tests together:
Task: "Unit test for SourceRef model validation in tests/unit/test_models.py"
Task: "Unit test for Candidate model validation in tests/unit/test_models.py"
Task: "Unit test for FieldResult model validation in tests/unit/test_models.py"
Task: "Unit test for ExtractionResult model validation in tests/unit/test_models.py"

# Launch all model implementations together:
Task: "Implement SourceRef model in src/ctrlf/app/models.py"
Task: "Implement Candidate model in src/ctrlf/app/models.py"
Task: "Implement FieldResult model in src/ctrlf/app/models.py"
Task: "Implement ExtractionResult model in src/ctrlf/app/models.py"
Task: "Implement Resolution model in src/ctrlf/app/models.py"
Task: "Implement PersistedRecord model in src/ctrlf/app/models.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD requirement)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
