---
description: "Task list for Structured Extraction with OpenAI/Gemini API Integration"
---

# Tasks: Structured Extraction with OpenAI/Gemini API Integration

**Input**: Design documents from `/specs/002-structured-extraction/`
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

**Purpose**: Project initialization and dependency setup

- [ ] T001 Add openai>=1.0.0 dependency to pyproject.toml
- [ ] T002 [P] Verify google-genai>=1.3.0 is in dependencies (already present)
- [ ] T003 [P] Verify langextract>=0.1.0 is in dependencies (already present)
- [ ] T004 [P] Verify thefuzz>=0.22.0 is in dependencies (already present)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 [P] Verify ExtractionRecord model exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T006 [P] Verify JSONLLine model exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T007 [P] Verify find_char_interval function exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T008 [P] Verify _flatten_extractions function exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T009 [P] Verify write_jsonl function exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T010 [P] Verify visualize_extractions function exists in src/ctrlf/app/structured_extract.py (draft exists, validate and enhance)
- [ ] T011 Create API key validation utility in src/ctrlf/app/structured_extract.py for checking OPENAI_API_KEY and GOOGLE_API_KEY environment variables
- [ ] T012 Create retry logic helper function with exponential backoff in src/ctrlf/app/structured_extract.py (max_retries=3, handles 429, 5xx, timeouts)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Extract Using OpenAI Structured Outputs (Priority: P1) 🎯 MVP

**Goal**: Core extraction workflow - accept schema and corpus, call OpenAI API with structured outputs, generate JSONL file with character intervals

**Independent Test**: Provide simple schema (character, emotion, relationship) and 2-3 documents. System calls OpenAI API with response_format containing schema, processes response, generates valid JSONL lines with character intervals and alignment status.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Unit test for ExtractionRecord model validation in tests/unit/test_structured_extract.py
- [ ] T014 [P] [US1] Unit test for JSONLLine model validation in tests/unit/test_structured_extract.py
- [ ] T015 [P] [US1] Unit test for find_char_interval with exact match in tests/unit/test_structured_extract.py
- [ ] T016 [P] [US1] Unit test for find_char_interval with fuzzy match in tests/unit/test_structured_extract.py
- [ ] T017 [P] [US1] Unit test for find_char_interval with no match in tests/unit/test_structured_extract.py
- [ ] T018 [P] [US1] Unit test for _flatten_extractions with flat schema in tests/unit/test_structured_extract.py
- [ ] T019 [P] [US1] Unit test for _flatten_extractions with nested objects in tests/unit/test_structured_extract.py
- [ ] T020 [P] [US1] Unit test for _flatten_extractions with arrays in tests/unit/test_structured_extract.py
- [ ] T021 [P] [US1] Unit test for _call_structured_extraction_api with OpenAI (mocked) in tests/unit/test_structured_extract.py
- [ ] T022 [P] [US1] Unit test for _call_structured_extraction_api error handling (mocked) in tests/unit/test_structured_extract.py
- [ ] T023 [P] [US1] Unit test for write_jsonl function in tests/unit/test_structured_extract.py
- [ ] T024 [US1] Integration test for OpenAI extraction workflow in tests/integration/test_structured_extraction_e2e.py (mocked API calls)

### Implementation for User Story 1

- [ ] T025 [US1] Implement _call_structured_extraction_api for OpenAI provider in src/ctrlf/app/structured_extract.py (depends on T011, T012)
- [ ] T026 [US1] Add OpenAI API client initialization with API key from environment in src/ctrlf/app/structured_extract.py (depends on T011)
- [ ] T027 [US1] Add structured output format configuration for OpenAI (response_format with json_schema) in src/ctrlf/app/structured_extract.py (depends on T025)
- [ ] T028 [US1] Add error handling for OpenAI API errors (rate limits, timeouts, invalid responses) in src/ctrlf/app/structured_extract.py (depends on T012, T025)
- [ ] T029 [US1] Implement run_structured_extraction with OpenAI provider support in src/ctrlf/app/structured_extract.py (depends on T025, T007, T008)
- [ ] T030 [US1] Add token limit detection and handling for OpenAI in src/ctrlf/app/structured_extract.py (depends on T025)
- [ ] T031 [US1] Add logging for OpenAI API calls (token usage, response times) in src/ctrlf/app/structured_extract.py (depends on T025)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Extract Using Gemini Structured Outputs (Priority: P1)

**Goal**: Support Gemini API as alternative provider with same functionality as OpenAI

**Independent Test**: Switch provider to "gemini" with same schema and corpus. System calls Gemini API with response_schema in generation config, processes response, generates same JSONL format as OpenAI extractions.

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T032 [P] [US2] Unit test for _call_structured_extraction_api with Gemini (mocked) in tests/unit/test_structured_extract.py
- [ ] T033 [P] [US2] Unit test for Gemini API error handling (mocked) in tests/unit/test_structured_extract.py
- [ ] T034 [US2] Integration test for Gemini extraction workflow in tests/integration/test_structured_extraction_e2e.py (mocked API calls)

### Implementation for User Story 2

- [ ] T035 [US2] Add Gemini provider support to _call_structured_extraction_api in src/ctrlf/app/structured_extract.py (depends on T025, T011)
- [ ] T036 [US2] Add Gemini API client initialization with API key from environment in src/ctrlf/app/structured_extract.py (depends on T011)
- [ ] T037 [US2] Add structured output format configuration for Gemini (response_schema in GenerationConfig) in src/ctrlf/app/structured_extract.py (depends on T035)
- [ ] T038 [US2] Add error handling for Gemini API errors (rate limits, timeouts, invalid responses) in src/ctrlf/app/structured_extract.py (depends on T012, T035)
- [ ] T039 [US2] Add provider selection logic to run_structured_extraction in src/ctrlf/app/structured_extract.py (depends on T029, T035)
- [ ] T040 [US2] Add token limit detection and handling for Gemini in src/ctrlf/app/structured_extract.py (depends on T035)
- [ ] T041 [US2] Add logging for Gemini API calls (token usage, response times) in src/ctrlf/app/structured_extract.py (depends on T035)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Visualize and Export JSONL Results (Priority: P2)

**Goal**: Generate HTML visualizations and ensure JSONL format compatibility with langextract.visualize()

**Independent Test**: Generate JSONL file from extraction results. System generates HTML using langextract.visualize(), saves to file, and JSONL format matches langextract expectations with correct structure.

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T042 [P] [US3] Unit test for visualize_extractions function in tests/unit/test_structured_extract.py
- [ ] T043 [P] [US3] Unit test for JSONL format validation in tests/unit/test_structured_extract.py
- [ ] T044 [US3] Integration test for visualization workflow in tests/integration/test_structured_extraction_e2e.py

### Implementation for User Story 3

- [ ] T045 [US3] Enhance visualize_extractions to handle langextract.visualize() return types in src/ctrlf/app/structured_extract.py (depends on T010)
- [ ] T046 [US3] Add JSONL format validation before visualization in src/ctrlf/app/structured_extract.py (depends on T009)
- [ ] T047 [US3] Add error handling for visualization failures in src/ctrlf/app/structured_extract.py (depends on T045)
- [ ] T048 [US3] Verify JSONL output format matches langextract.visualize() expectations in src/ctrlf/app/structured_extract.py (depends on T009)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T049 [P] Add comprehensive error messages for API key missing/invalid scenarios in src/ctrlf/app/structured_extract.py
- [ ] T050 [P] Add schema validation before API calls in src/ctrlf/app/structured_extract.py
- [ ] T051 [P] Add cost estimation helpers (optional, log token usage) in src/ctrlf/app/structured_extract.py
- [ ] T052 [P] Add documentation strings to all functions in src/ctrlf/app/structured_extract.py
- [ ] T053 [P] Update module docstring in src/ctrlf/app/structured_extract.py
- [ ] T054 [P] Add type hints to all functions in src/ctrlf/app/structured_extract.py
- [ ] T055 [P] Run make check to ensure all linting and type checking passes
- [ ] T056 [P] Update README.md with structured extraction usage examples
- [ ] T057 [P] Run quickstart.md validation to ensure examples work

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
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Shares infrastructure with US1 but independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Depends on JSONL generation from US1/US2 but independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- API integration before orchestration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, User Stories 1 and 2 can start in parallel (both P1)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all model validation tests together:
Task: "Unit test for ExtractionRecord model validation in tests/unit/test_structured_extract.py"
Task: "Unit test for JSONLLine model validation in tests/unit/test_structured_extract.py"

# Launch all function tests together:
Task: "Unit test for find_char_interval with exact match in tests/unit/test_structured_extract.py"
Task: "Unit test for find_char_interval with fuzzy match in tests/unit/test_structured_extract.py"
Task: "Unit test for find_char_interval with no match in tests/unit/test_structured_extract.py"
Task: "Unit test for _flatten_extractions with flat schema in tests/unit/test_structured_extract.py"
Task: "Unit test for _flatten_extractions with nested objects in tests/unit/test_structured_extract.py"
Task: "Unit test for _flatten_extractions with arrays in tests/unit/test_structured_extract.py"
```

---

## Parallel Example: User Stories 1 and 2

```bash
# Once Foundational phase is complete, both P1 stories can proceed in parallel:

# Developer A: User Story 1 (OpenAI)
Task: "Implement _call_structured_extraction_api for OpenAI provider in src/ctrlf/app/structured_extract.py"
Task: "Add OpenAI API client initialization with API key from environment in src/ctrlf/app/structured_extract.py"

# Developer B: User Story 2 (Gemini)
Task: "Add Gemini provider support to _call_structured_extraction_api in src/ctrlf/app/structured_extract.py"
Task: "Add Gemini API client initialization with API key from environment in src/ctrlf/app/structured_extract.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (OpenAI extraction)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (OpenAI) → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 (Gemini) → Test independently → Deploy/Demo
4. Add User Story 3 (Visualization) → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (OpenAI)
   - Developer B: User Story 2 (Gemini)
   - Developer C: User Story 3 (Visualization) - can start after US1/US2
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Draft implementation exists in src/ctrlf/app/structured_extract.py - tasks focus on completing API integration
- API calls should be mocked in tests to avoid actual API usage during testing
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
