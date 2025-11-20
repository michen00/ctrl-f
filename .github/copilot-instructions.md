# Repository Instructions for GitHub Copilot

## 1) High‑Level Details

**ctrlf** is a **Schema-Grounded Corpus Extractor** - a Python application that extracts structured data from unstructured documents with full provenance tracking and zero fabrication guarantees.

### Core Purpose

- Extract structured data from document corpora (PDF, DOCX, HTML, TXT) based on user-defined schemas
- Use AI-powered extraction with confidence scoring, deduplication, and consensus detection
- Provide an interactive Gradio web interface for review and resolution
- Ensure all extracted values are grounded in source documents with complete provenance tracking

### Key Features

- **Schema-Driven**: Accepts JSON Schema or Pydantic model definitions
- **Multi-Format Support**: Processes PDF, DOCX, HTML, TXT from directories or archives (ZIP, TAR, TAR.GZ)
- **AI-Powered Extraction**: Uses `langextract` to extract candidate values with Google GenAI to generate synthetic examples and extract structured data from the input corpus
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
- **uv** (recommended) or `pip` for dependency management

## 2) Build and Validation Information

### Installation

- **Production**: `make install` (installs dependencies via `uv sync`)
- **Development**: `make develop` (includes dev dependencies and git hooks)
  - Optional: `make develop WITH_HOOKS=false` to skip git hooks
- **Demo**: `make demo` (installs and runs the server on `http://localhost:7860`)

### Build System

- Uses `uv` as the package manager (auto-installed if missing)
- Build backend: `uv_build` (configured in `pyproject.toml`)
- Python version managed via `.python-version` file
- Build markers in `build/` directory track dependency versions

### Code Quality & Testing

- **Format & Lint**: `make format-all` (runs pre-commit hooks + ruff format)
  - Individual: `make format` (lint + format), `make lint` (ruff check --fix)
- **Type Checking**: `mypy` configured with strict mode, Pydantic plugin enabled
- **Testing**: `make test` (runs pytest with coverage)
  - Parallel: `make test PARALLEL=true`
  - Coverage reports: `--cov=src --cov-report=term-missing`
- **Full Check**: `make check` (runs `format-all` + `test`)

### Pre-commit Hooks

- Managed via `pre-commit` (or `prek` if available)
- Install: `make enable-git-hooks` (or automatically with `make develop`)
- Run manually: `make run-pre-commit`
- Hooks configured in `.pre-commit-config.yaml` (not shown but referenced)

### Development Workflow

1. `make develop` - Install with dev dependencies
2. `make format-all` - Format and lint code
3. `make test` - Run tests
4. `make check` - Run all checks (format + test)

### Environment Variables

- `DEBUG=true` - Enable debug output in Makefile
- `VERBOSE=true` - Enable verbose output
- `PARALLEL=true|false` - Control parallel test execution (default: true)
- `WITH_HOOKS=true|false` - Control git hooks installation (default: true)
- `CACHE=true|false` - Control build cache (default: true)

## 3) Project Layout and Architecture

### Directory Structure

```text
ctrlf/
├── src/ctrlf/          # Source code
│   ├── app/            # Application modules
│   │   ├── models.py          # Pydantic data models (SourceRef, Candidate, FieldResult, etc.)
│   │   ├── schema_io.py       # Schema validation and conversion (JSON Schema ↔ Pydantic)
│   │   ├── ingest.py          # Document conversion to Markdown (markitdown wrapper)
│   │   ├── extract.py         # Field extraction using langextract
│   │   ├── aggregate.py       # Candidate deduplication, normalization, consensus detection
│   │   ├── storage.py         # TinyDB adapter for persistent storage
│   │   ├── ui.py              # Gradio interface components
│   │   ├── server.py          # Application entrypoint
│   │   ├── errors.py          # Error handling and error summary collection
│   │   └── logging_conf.py    # Structured logging configuration
│   └── bin/            # Scripts
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── contract/       # Contract tests
├── specs/              # Feature specifications
│   └── 001-schema-corpus-extractor/  # Main spec with requirements, user stories, etc.
└── docs/               # Documentation
```

### Data Flow

1. **Ingestion**: Corpus files → Markdown conversion → `CorpusDocument` with source mapping
2. **Extraction**: Markdown + Schema → Candidate generation → `Candidate` with `SourceRef`
3. **Aggregation**: Candidates → Deduplication → Normalization → Consensus detection → `FieldResult`
4. **Review**: `FieldResult` → User selection → `Resolution`
5. **Persistence**: Resolutions → Validation → `PersistedRecord` → TinyDB

### Key Data Models (`src/ctrlf/app/models.py`)

- **`SourceRef`**: Document location (doc_id, path, location, snippet, metadata)
- **`Candidate`**: Extracted value with confidence score and source references
- **`FieldResult`**: Aggregated candidates for a field with optional consensus
- **`ExtractionResult`**: Complete extraction output for all fields
- **`Resolution`**: User's decision for a field (candidate or custom value)
- **`PersistedRecord`**: Final saved record with provenance and audit trail

### Application Entry Point

- **Server**: `src/ctrlf/app/server.py` - `main()` function launches Gradio app
- **UI**: `src/ctrlf/app/ui.py` - `create_upload_interface()` creates the Gradio interface
- **Run**: `make demo` or `uv run python -m ctrlf.app.server`
- **Port**: Default `http://localhost:7860`

### Storage

- **Location**: `~/.ctrlf/db/` (default)
- **Database**: TinyDB (JSON-based)
- **Structure**: One table per schema (keyed by schema hash), records keyed by `record_id`
- **Record Contents**: resolved values, provenance, audit trail (run ID, timestamp, app version, config)

### Configuration

- **Consensus Threshold**: 0.75 (minimum confidence, adjustable in UI)
- **Margin Threshold**: 0.20 (minimum margin over next candidate, fixed)
- **Null Policy**: "Empty List" (default) or "Explicit Null" for fields with no candidates

### Limitations (v0)

- Flat schemas only (no nested objects/arrays)
- Single-user application (no authentication)
- Local processing only (no cloud/network features)
- Batch processing (no incremental streaming)
- Basic normalization (limited for emails, dates, URLs)

## 4) Conventional Commits and contribution workflow

- Commit message format: `<type>(<scope>): <subject>`

  - Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`
  - Useful scopes for this repo:

    - `app` - Application code changes
    - `ui` - User interface changes
    - `extract` - Extraction logic
    - `ingest` - Document ingestion
    - `aggregate` - Candidate aggregation/deduplication
    - `storage` - Database/storage layer
    - `schema` - Schema handling/validation
    - `models` - Data models
    - `server` - Server/entrypoint
    - `test` - Test code
    - `docs` - Documentation
    - `build` - Build system
    - `deps` - Dependencies

  - Examples:

    - `feat(extract): add support for array field extraction`
    - `fix(ui): resolve source viewing bug in review interface`
    - `docs(readme): update installation instructions`
    - `test(aggregate): add tests for consensus detection`
    - `refactor(models): simplify SourceRef validation`
    - `chore(deps): update pydantic to 2.12.2`

- Recommended loop before commit/PR:

  - `make check` (or run `make develop` → `make format-all` → `make test` in that order)
  - Keep CHANGELOG via conventional commits; `cliff.toml` is included for changelog tooling if you choose to generate release notes.

---

Note to GitHub Copilot: Please trust these instructions and only perform additional searches if the information provided is incomplete or found to be in error.
