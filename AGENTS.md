# Agent Instructions for this Repository

## 1. Repository Identity & Purpose

**ctrlf** is a **Schema-Grounded Corpus Extractor** - a Python application that extracts structured data from unstructured documents with full provenance tracking and zero fabrication guarantees.

### Core Purpose

- Extract structured data from document corpora (PDF, DOCX, HTML, TXT) based on user-defined schemas (JSON Schema or Pydantic models)
- Use AI-powered extraction with confidence scoring, deduplication, and consensus detection
- Provide an interactive Gradio web interface for review and resolution
- Ensure all extracted values are grounded in source documents with complete provenance tracking
- Persist validated records to TinyDB with complete audit trails

### Key Features

- **Schema-Driven**: Accepts JSON Schema or Pydantic model definitions
- **Multi-Format Support**: Processes PDF, DOCX, HTML, TXT from directories or archives (ZIP, TAR, TAR.GZ)
- **AI-Powered Extraction**: Uses `langextract` with Google GenAI to extract candidate values with confidence scores
- **Intelligent Deduplication**: Groups near-duplicate candidates using fuzzy string matching (`thefuzz`)
- **Consensus Detection**: Automatically identifies high-confidence values (confidence ≥0.75, margin ≥0.20)
- **Full Provenance**: Every value includes source document, location (page/line or char-range), and context snippet
- **Zero Fabrication**: All candidates must be grounded in actual document content
- **Interactive Review**: Gradio web UI with filtering, search, source viewing, and side-by-side comparison
- **Persistent Storage**: Saves validated records to TinyDB with complete audit trails

### Technology Stack

- **Python 3.12+** required
- **Pydantic v2** for data validation and type safety
- **Gradio** for web UI
- **TinyDB** for local JSON database storage
- **markitdown** for document to Markdown conversion
- **langextract** for field extraction
- **thefuzz** for fuzzy string matching
- **structlog** for structured logging
- Standard Python tooling (`uv`, `ruff`, `mypy`, `pytest`)

## 2. Development Workflow

### Environment Setup

- The project uses `uv` for dependency management. `uv` automatically creates and manages a virtual environment (`.venv`) in the project root.
- **Execution:** Prefer `uv run` over direct `python` calls (e.g., use `uv run python -m mymodule` instead of `python -m mymodule`) to ensure the correct environment is used.
- Python version: >=3.12 (defined in `.python-version` and `pyproject.toml`).

### Git Configuration

- **Git LFS:** This project uses Git LFS. The `make develop` target handles `git lfs install --local` and `git lfs pull`.
- **Blame Ignore:** A `.git-blame-ignore-revs` file is present to exclude large style changes from `git blame`. Configure it locally via `git config blame.ignoreRevsFile .git-blame-ignore-revs` (handled by `make develop`).

### Common Commands (Makefile)

The `Makefile` is the primary entry point for development tasks.

- **Run Demo/Server:** `make demo` (starts Gradio server on <http://localhost:7860>)
- **Install Dependencies:** `make develop` (installs dev deps and git hooks)
- **Run All Checks (CI):** `make check` (runs format-all and test)
- **Run Tests:** `make test` (runs pytest with coverage, supports PARALLEL={true|false})
- **Linting:** `make lint` (ruff check --fix)
- **Formatting:** `make format` (lint + ruff format) or `make format-all` (pre-commit + format-unsafe)
- **Clean:** `make clean` (removes artifacts and caches)
- **Git Hooks:** `make enable-git-hooks` / `make disable-git-hooks`
- **Publishing:** `make push-test` / `make push-prod`

### Testing

- **Test-Driven Development (TDD):** Strongly encouraged. Write tests _before_ implementation to ensure requirements are met and code is robust.
- **Test Categories (Purpose):**
  - **Type 1: Executable Documentation:** Treat these tests like code snippets in tutorials. They should be **DAMP** (Descriptive And Meaningful Phrases) and well-commented. Place them "above the fold" in test files. These are meant for reading and understanding how to use the code.
  - **Type 2: Coverage & Reliability:** Focus on edge cases, path coverage, and confidence. These can be more **DRY** (Don't Repeat Yourself) but must remain readable. Ensure tests never fail opaquely.
- **Testing Paradigms (Scope):**
  - **Unit Tests:** Focus on individual components in isolation.
  - **Integration Tests:** Verify interactions between components or external systems.
  - **Smoke Tests:** Quick checks to ensure critical paths work.
  - **Functional/E2E Tests:** Validate full workflows from a user's perspective.
  - **Regression Tests:** Add specific tests when worthy bugs are found to prevent recurrence.
- **What to Test:**
  - **Public Interfaces:** Test user-facing code thoroughly.
  - **Private Implementation:** Avoid testing private methods or classes directly; test them through the public interface.
- **Pytest Best Practices:**
  - **Fixtures:** Use `pytest` fixtures for setup/teardown (prefer `yield` fixtures). Use `conftest.py` for shared fixtures.
  - **Parametrization:** Use `@pytest.mark.parametrize` for data-driven testing to cover multiple scenarios efficiently.
  - **Markers:** Use markers (e.g., `@pytest.mark.slow`, `@pytest.mark.integration`) to categorize tests.
  - **Style:** Prefer functional tests over class-based tests.
- Tests are located in `tests/`.
- Uses `pytest`.

## 3. Project Structure

- `src/ctrlf/`: Source code for the package.
  - `src/ctrlf/app/`: Main application modules:
    - `models.py`: Pydantic data models (SourceRef, Candidate, FieldResult, ExtractionResult, Resolution, PersistedRecord)
    - `schema_io.py`: Schema validation and conversion (JSON Schema ↔ Pydantic)
    - `ingest.py`: Document conversion to Markdown (markitdown wrapper)
    - `extract.py`: Field extraction using langextract
    - `aggregate.py`: Candidate deduplication, normalization, consensus detection
    - `storage.py`: TinyDB adapter for persistent storage
    - `ui.py`: Gradio interface components
    - `server.py`: Application entrypoint (launches Gradio server)
    - `errors.py`: Error handling and error summary collection
    - `logging_conf.py`: Structured logging configuration
  - `src/ctrlf/bin/`: Scripts and utilities
- `tests/`: Test suite.
  - `tests/unit/`: Unit tests for individual modules
  - `tests/integration/`: Integration tests for workflows
  - `tests/contract/`: Contract tests
- `specs/`: Feature specifications (e.g., `001-schema-corpus-extractor/`)
- `.github/`: GitHub Actions workflows.
- `pyproject.toml`: Project configuration (build, deps, tools).
- `uv.lock`: Locked dependencies.

### Configuration Files

- **Linting & Formatting:**
  - `.ruff.toml`: Configuration for `ruff` (linting and formatting).
  - `.pylintrc`: Configuration for `pylint`.
  - `.bandit.yml`: Security linting configuration.
  - `.codespellrc`: Spell checking configuration.
  - `.markdownlint.yml`: Markdown linting configuration.
  - `.pre-commit-config.yaml`: Pre-commit hook definitions.
- **Testing:**
  - `pyproject.toml`: Contains `pytest` configuration (under `[tool.pytest.ini_options]`).
  - `.coveragerc`: Coverage configuration.
- **Editor & Environment:**
  - `.editorconfig`: Universal editor configuration.
  - `.vscode/settings.json`: VS Code workspace settings.
  - `.vscode/extensions.json`: Recommended VS Code extensions.
  - `.python-version`: Specifies the Python version.
- **Git:**
  - `.gitignore`: Files to ignore.
  - `.gitattributes`: Git attribute settings.
  - `.git-blame-ignore-revs`: Revisions to ignore in blame.
- **Documentation & Release:**
  - `cliff.toml`: Configuration for `git cliff` (changelog generation).
  - `.readthedocs.yaml`: Read the Docs configuration.

## 4. Contribution Guidelines

- **Commit Messages:**
  - Follow Conventional Commits (e.g., `feat(core): add new utility`).
  - **Atomic Commits:** Keep commits atomic (one logical change per commit).
  - **Summary Length:** Keep the summary (first line) under 50 characters.
- **Backward Compatibility:**
  - Maintain backward compatibility for public APIs, especially in minor and patch releases.
  - If breaking changes are necessary, they must be documented and ideally deprecated first with warnings.
- **Changelog:** Managed via `git cliff`.
- **Code Style:** Enforced by `ruff` and `mypy`.
- **Verification:** Run `make check` to perform final checks (linting, types, tests) after making code changes.
- **Onboarding:** The project includes a `make demo` target that starts the Gradio server for quick testing.
- **Library Preferences:**
  - **CLI:** Use `typer` with `rich-argparse` for command-line interfaces.
  - **Data Processing:** Prefer `polars` over `pandas` for data manipulation.
  - **Logging:** Use structured logging (via `structlog`).
  - **JSON:** Use `yapic-json` for high-performance JSON processing.
  - **Progress Indicators:** Use `tqdm` (or `rich.progress`) for long-running tasks where performance allows.
- **CLI Guidelines:**
  - **Help:** Ensure all CLI tools have a helpful `--help` message.
  - **Colors:** Use colors where helpful.
  - **Docstrings:** Use a heredoc (multi-line string) for the "docstring" or usage text.
- **Python Idioms:**
  - **Concurrency:**
    - **Preference:** Use `concurrent.futures` for parallelism or `asyncio` for async I/O.
    - **Optionality:** Concurrency should be optional (controlled by a flag/argument). Always include a serial/synchronous implementation.
  - **`__slots__`:** Use `__slots__` for classes where memory optimization is beneficial or to prevent dynamic attribute creation.
  - **Return Values:** Use `NamedTuple` (or `dataclass`) for return values with more than 3 elements, instead of plain tuples.
  - **Enums:** Use `StrEnum` (Python 3.11+) and `auto()` for string-based enumerations.
  - **Path Manipulation:** Prefer `pathlib.Path` over `os.path` or string manipulation for file system paths.
  - **Membership Checks:** Use `set` for membership checks where appropriate, but be mindful of the overhead of casting iterables to sets.
  - **Loop Optimization:** Avoid repeated dot notation access in tight loops. Cache methods or attributes (e.g., `append_to_my_list = my_list.append`) outside the loop if they are static accessors.
  - **Module Exports (`__all__`):** Define `__all__` as a tuple (not list) to control exports. Omit parentheses if it fits on one line. Place it near the top of the file, immediately below the docstring (or `from __future__` imports).
- **Type Hints:**

  - Use modern type hints (Python 3.11+) for all function signatures (e.g., `list[str]`, `str | None`).
  - Annotate class attributes and significant variables.
  - **Pydantic Types:** Utilize numeric convenience types from Pydantic (e.g., `NegativeInt`, `NonPositiveFloat`) where appropriate to enforce constraints at the type level.
  - **Annotated Constraints:** Use `typing.Annotated` with `pydantic.Field` to define reusable constrained types.

    ```python
    # Examples of useful patterns (not included by default):
    FinitePositiveFloat = Annotated[FiniteFloat, Field(gt=0.0)]
    Probability = Annotated[NonNegativeFloat, Field(le=1.0)]
    NonEmptyStr = Annotated[str, Field(min_length=1)]
    ```

- **Documentation:**
  - **Syncing:** Ensure `AGENTS.md`, `.github/copilot-instructions.md`, `.github/instructions/CI.instructions.md`, and `README.md` are kept updated and synced with code changes.
  - Follow **Google-style** docstrings.
  - **Modules:** Include docstrings at the module level.
  - **Classes:** All classes must have docstrings.
  - **Constants & Enums:** Document constants and Enum members.
  - **Functions & Methods:**
    - **`Args:` Section:** Use judiciously. Do not include an `Args:` section just to list argument names.
    - **When to use `Args:`:** Only if it adds material value (e.g., critical code, complex/opaque logic, special gotchas, or non-obvious parameters).
    - **No Redundancy:** If you do include `Args:`, do not repeat information already clear from the signature (like types or self-explanatory names).

## 5. Project-Specific Guidelines

### Data Flow Architecture

The application follows a clear data flow:

1. **Ingestion** (`ingest.py`): Corpus files → Markdown conversion → `CorpusDocument` with source mapping
2. **Extraction** (`extract.py`): Markdown + Schema → Candidate generation → `Candidate` with `SourceRef`
3. **Aggregation** (`aggregate.py`): Candidates → Deduplication → Normalization → Consensus detection → `FieldResult`
4. **Review** (`ui.py`): `FieldResult` → User selection → `Resolution`
5. **Persistence** (`storage.py`): Resolutions → Validation → `PersistedRecord` → TinyDB

### Key Data Models

- **`SourceRef`**: Document location (doc_id, path, location, snippet, metadata)
- **`Candidate`**: Extracted value with confidence score and source references
- **`FieldResult`**: Aggregated candidates for a field with optional consensus
- **`ExtractionResult`**: Complete extraction output for all fields
- **`Resolution`**: User's decision for a field (candidate or custom value)
- **`PersistedRecord`**: Final saved record with provenance and audit trail

### Schema Handling

- Supports both JSON Schema and Pydantic models
- Schema conversion handled in `schema_io.py`
- Currently supports flat schemas only (primitive types or arrays of primitives)
- Nested objects/arrays are not supported in v0

### Document Processing

- Uses `markitdown` for document conversion (PDF, DOCX, HTML, TXT → Markdown)
- Supports directories and archives (ZIP, TAR, TAR.GZ)
- All processing happens locally (no network transmission)
- Errors are handled gracefully with comprehensive error summaries

### Consensus Detection

- **Confidence threshold**: 0.75 (minimum confidence for consensus, adjustable in UI)
- **Margin threshold**: 0.20 (minimum margin over next candidate, fixed)
- Consensus candidates are automatically pre-selected in the UI
- Disagreements (similar confidence scores) are flagged for manual review

### Storage

- Records saved to TinyDB in `~/.ctrlf/db/` (default)
- One table per schema (keyed by schema hash)
- Records keyed by `record_id`
- Each record includes: resolved values, provenance, audit trail

### UI Components

- Gradio-based web interface (`ui.py`)
- Field filtering and search capabilities
- Visual indicators for disagreements and consensus
- Individual source viewing for each candidate
- Side-by-side source comparison
- Progress tracking with cancellation support

## 6. Specific Task Instructions

### Adding a Dependency

1. Add it to `pyproject.toml` (dependencies or dev-dependencies).
2. Run `uv lock` (or `make develop` which usually handles sync).

### Modifying CI

- Edit `.github/workflows/CI.yml`.

### Running the Application

- **Development**: `make demo` starts the Gradio server on <http://localhost:7860>
- **Production**: `uv run python -m ctrlf.app.server` (or use the installed package entrypoint)

### Testing Extraction Workflows

- Unit tests cover individual modules (ingest, extract, aggregate, storage, schema_io)
- Integration tests verify end-to-end workflows
- Contract tests ensure API contracts are maintained
- Use `make test` to run all tests with coverage

### Working with Schemas

- JSON Schema files should follow draft-07 specification
- Pydantic models should use Pydantic v2 syntax
- Schema validation happens in `schema_io.py`
- Test schema loading with `tests/unit/test_schema_io.py`
