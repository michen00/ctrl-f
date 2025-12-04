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

- [x] T001 Add pydantic-ai>=0.0.14 dependency to pyproject.toml ✅ **COMPLETE**
- [x] T002 [P] Verify google-genai>=1.3.0 is in dependencies (already present) ✅ **VERIFIED**
- [x] T003 [P] Verify langextract>=0.1.0 is in dependencies (already present) ✅ **VERIFIED**
- [x] T004 [P] Verify thefuzz>=0.22.0 is in dependencies (already present) ✅ **VERIFIED**

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 [P] Verify ExtractionRecord model exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T006 [P] Verify JSONLLine model exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T007 [P] Verify find_char_interval function exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T008 [P] Verify _flatten_extractions function exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T009 [P] Verify write_jsonl function exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T010 [P] Verify visualize_extractions function exists in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [ ] T011 Create API key validation utility in src/ctrlf/app/structured_extract.py for checking OPENAI_API_KEY and GOOGLE_API_KEY environment variables (Ollama doesn't need keys)
- [ ] T012 Create retry logic helper function with exponential backoff in src/ctrlf/app/structured_extract.py (max_retries=3, handles 429, 5xx, timeouts)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Extract Using Structured Outputs (Priority: P1) 🎯 MVP

**Goal**: Core extraction workflow - accept schema and corpus, call PydanticAI Agent with Ollama (default), OpenAI, or Gemini, generate JSONL file with character intervals

**Independent Test**: Provide simple schema (character, emotion, relationship) and 2-3 documents. System calls PydanticAI Agent with provider="ollama" (or "openai"/"gemini"), processes response, generates valid JSONL lines with character intervals and alignment status.

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
- [ ] T021 [P] [US1] Unit test for _call_structured_extraction_api with Ollama/OpenAI/Gemini (mocked PydanticAI Agent) in tests/unit/test_structured_extract.py
- [ ] T022 [P] [US1] Unit test for _call_structured_extraction_api error handling (mocked) in tests/unit/test_structured_extract.py
- [ ] T023 [P] [US1] Unit test for write_jsonl function in tests/unit/test_structured_extract.py
- [ ] T024 [US1] Integration test for Ollama/OpenAI/Gemini extraction workflow in tests/integration/test_structured_extraction_e2e.py (mocked PydanticAI Agent calls)

### Implementation for User Story 1

- [x] T025 [US1] Implement _call_structured_extraction_api using PydanticAI Agent for Ollama/OpenAI/Gemini providers in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Uses PydanticAI Agent with unified interface
- [x] T026 [US1] Add provider model string configuration (ollama:model, openai:model, google-gla:model) in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Model strings configured per provider
- [x] T027 [US1] Add Pydantic model as output_type for structured outputs in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - PydanticAI Agent accepts schema_model as output_type
- [ ] T028 [US1] Add error handling for API errors (rate limits, timeouts, invalid responses) in src/ctrlf/app/structured_extract.py (depends on T012, T025)
- [x] T029 [US1] Implement run_structured_extraction with provider support in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Supports Ollama (default), OpenAI, Gemini
- [ ] T030 [US1] Add token limit detection and handling in src/ctrlf/app/structured_extract.py (depends on T025)
- [ ] T031 [US1] Add logging for API calls (token usage, response times) in src/ctrlf/app/structured_extract.py (depends on T025)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Provider Selection and Configuration (Priority: P1)

**Goal**: Support Ollama (default), OpenAI, and Gemini providers with unified functionality through PydanticAI

**Independent Test**: Switch provider to "ollama", "openai", or "gemini" with same schema and corpus. System calls PydanticAI Agent with correct provider model string, processes response, generates same JSONL format regardless of provider.

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T032 [P] [US2] Unit test for _call_structured_extraction_api with Ollama/OpenAI/Gemini providers (mocked PydanticAI Agent) in tests/unit/test_structured_extract.py
- [ ] T033 [P] [US2] Unit test for provider-specific error handling (mocked) in tests/unit/test_structured_extract.py
- [ ] T034 [US2] Integration test for multi-provider extraction workflow in tests/integration/test_structured_extraction_e2e.py (mocked PydanticAI Agent calls)

### Implementation for User Story 2

- [x] T035 [US2] Add Ollama/OpenAI/Gemini provider support to _call_structured_extraction_api in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - All providers supported via PydanticAI
- [x] T036 [US2] Add provider model string configuration (ollama:llama3, openai:gpt-4o, google-gla:gemini-2.5-flash) in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Model strings configured
- [x] T037 [US2] Add Pydantic model output_type configuration for all providers in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Unified via PydanticAI Agent
- [ ] T038 [US2] Add error handling for provider-specific API errors (rate limits, timeouts, invalid responses) in src/ctrlf/app/structured_extract.py (depends on T012, T035)
- [x] T039 [US2] Add provider selection logic to run_structured_extraction in src/ctrlf/app/structured_extract.py ✅ **COMPLETE** - Provider parameter with Ollama default
- [ ] T040 [US2] Add token limit detection and handling for all providers in src/ctrlf/app/structured_extract.py (depends on T035)
- [ ] T041 [US2] Add logging for provider API calls (token usage, response times) in src/ctrlf/app/structured_extract.py (depends on T035)

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

## Phase 6: UI Integration & Backward Compatibility

**Purpose**: Integration with existing Gradio UI and ensuring backward compatibility

- [ ] T058 [P] Integrate structured extraction into existing Gradio UI as primary extraction option in src/ctrlf/app/ui.py (FR-014)
- [ ] T059 [P] Ensure backward compatibility with existing extract.py logic (can coexist, new pipeline is primary) (FR-015)
- [ ] T060 [P] Verify deduplication logic works with structured extraction results (uses existing aggregate.py) (FR-016)
- [ ] T061 [P] Verify plurality vote suggestion mechanism works with structured extraction results (uses existing aggregate.py) (FR-017)

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T049 [P] Add comprehensive error messages for API key missing/invalid scenarios in src/ctrlf/app/structured_extract.py (Ollama doesn't need keys)
- [ ] T050 [P] Add schema validation before API calls in src/ctrlf/app/structured_extract.py
- [ ] T051 [P] Add cost estimation helpers (optional, log token usage) in src/ctrlf/app/structured_extract.py
- [x] T052 [P] Add documentation strings to all functions in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T053 [P] Update module docstring in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T054 [P] Add type hints to all functions in src/ctrlf/app/structured_extract.py ✅ **COMPLETE**
- [x] T055 [P] Run make check to ensure all linting and type checking passes ✅ **COMPLETE**
- [x] T056 [P] Update README.md with structured extraction usage examples ✅ **COMPLETE**
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
- **Implementation Status**: PydanticAI integration is complete in src/ctrlf/app/structured_extract.py - supports Ollama (default), OpenAI, and Gemini via unified PydanticAI Agent interface
- **Completed Tasks**: Core extraction functionality (T025-T029, T035-T039) is complete. Remaining work focuses on error handling, logging, UI integration, and testing
- API calls should be mocked in tests to avoid actual API usage during testing (mock PydanticAI Agent, not individual API clients)
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
