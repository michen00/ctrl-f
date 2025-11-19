# ctrlf

**Schema-Grounded Corpus Extractor** - Extract structured data from unstructured documents with full provenance tracking and zero fabrication guarantees.

## Overview

`ctrlf` is a Python application that extracts structured data from document corpora based on user-defined schemas. It uses AI-powered extraction with confidence scoring, deduplication, and consensus detection to identify candidate values, then provides an interactive web interface for review and resolution. All extracted values are grounded in source documents with complete provenance tracking.

## Features

### Core Capabilities

- **Schema-Driven Extraction**: Accept JSON Schema or Pydantic model definitions to specify the structure of data to extract
- **Multi-Format Document Support**: Process PDF, DOCX, HTML, and TXT files from directories or archives (ZIP, TAR, TAR.GZ)
- **AI-Powered Field Extraction**: Uses `langextract` to extract candidate values for each schema field with confidence scores
- **Intelligent Deduplication**: Groups near-duplicate candidates using fuzzy string matching (`thefuzz`)
- **Consensus Detection**: Automatically identifies high-confidence values when one candidate significantly outperforms others
- **Disagreement Detection**: Flags fields where multiple candidates have similar confidence scores, requiring manual review
- **Full Provenance Tracking**: Every extracted value includes source document, location (page/line or char-range), and context snippet
- **Zero Fabrication Guarantee**: All candidates must be grounded in actual document content with verifiable source locations
- **Interactive Review Interface**: Gradio web UI with:
  - Field filtering and search capabilities
  - Visual indicators for disagreements and consensus
  - Individual source viewing for each candidate
  - Side-by-side source comparison
  - Progress tracking with cancellation support
- **Export Capabilities**: Export validated records as JSON files with full provenance
- **Persistent Storage**: Saves validated records to TinyDB with complete audit trails

### Technical Highlights

- **Type-Safe**: Built with Pydantic v2 for runtime validation and type safety
- **Structured Logging**: Uses `structlog` for contextual, structured logging throughout
- **Graceful Error Handling**: Continues processing on individual document/field errors, provides comprehensive error summaries
- **Local Processing**: All processing happens locally - no network transmission of documents or data
- **Extensible Architecture**: Modular design with clear separation of concerns (ingest, extract, aggregate, storage)

## Installation

### Requirements

- Python 3.12+
- `uv` (recommended) or `pip`
- `make` (for using Makefile shortcuts)

### Install with Makefile (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ctrlf

# Install the project
make install

# Or install for development (includes dev dependencies and git hooks)
make develop
```

The Makefile will automatically:

- Check for and install `uv` if needed
- Install Python version from `.python-version`
- Install all dependencies
- Set up git hooks (if using `make develop`)

### Install with uv directly

```bash
# Clone the repository
git clone <repository-url>
cd ctrlf

# Install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install with pip

```bash
pip install -e .
```

## Quick Start

### 1. Install and Start the Application

```bash
# Install the project (if not already done)
make install

# Start the application
python -m ctrlf.app.server
```

The application will start on `http://localhost:7860`.

**Note**: If you're developing, use `make develop` instead of `make install` to get development dependencies and git hooks.

### 2. Define Your Schema

Create a JSON Schema file (`schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Person's full name"
    },
    "email": {
      "type": "string",
      "format": "email",
      "description": "Email address"
    },
    "date": {
      "type": "string",
      "format": "date",
      "description": "Date in YYYY-MM-DD format"
    }
  },
  "required": ["name", "email"]
}
```

Or create a Pydantic model (`schema.py`):

```python
from pydantic import BaseModel, EmailStr
from datetime import date

class Person(BaseModel):
    name: str
    email: EmailStr
    date: date | None = None
```

**Note**: v0 supports only flat schemas (primitive types or arrays of primitives). Nested objects/arrays are not supported.

### 3. Prepare Your Corpus

Organize your documents in a directory:

```text
corpus/
├── document1.pdf
├── document2.docx
├── document3.html
└── document4.txt
```

Or create a ZIP/TAR archive of documents.

### 4. Run Extraction

1. Open `http://localhost:7860` in your browser
2. **Upload Schema**: Upload your schema file (JSON or Python)
3. **Select Schema Type**: Choose "JSON Schema" or "Pydantic Model" (auto-detected from file extension)
4. **Upload Corpus**: Either:
   - Upload a ZIP/TAR archive file, or
   - Enter a directory path containing your documents
5. **Configure Options** (optional):
   - **Null policy**: How to handle empty fields
     - "Empty List" (default): `[]` for fields with no candidates
     - "Explicit Null": `[null]` for fields with no candidates
   - **Confidence threshold**: Consensus detection threshold (default: 0.75)
     - Higher values require more confidence for automatic consensus
6. Click **"Run Extraction"**
7. **Monitor Progress**:
   - Progress bar shows current stage (schema loading, corpus processing, extraction)
   - Progress messages display detailed status
   - You can cancel the operation at any time using the cancel button

### 5. Review and Resolve

The review interface provides comprehensive tools for reviewing and resolving extracted candidates:

1. **Filter and Search Fields**:

   - Use the search box to filter fields by name
   - Filter by type:
     - **All**: Show all fields
     - **Unresolved**: Show only fields without consensus
     - **Flagged (Disagreements)**: Show only fields with conflicting candidates

2. **Review Candidates**: For each field, you'll see:

   - **Consensus Status**:
     - ✅ Consensus detected (with confidence percentage) - value is pre-selected
     - 🔴 Disagreement detected - multiple candidates with similar confidence, manual selection required
     - ⚠️ No consensus - manual selection required
   - **Candidate List**: All candidate values with confidence scores displayed as percentages
   - **Individual "View Source" Buttons**: One button per candidate to view its source context
   - **"View Source for Selected Candidate" Button**: View source for the currently selected candidate
   - **"Other" Text Input**: Option to enter a custom value (validated against field type)

3. **View Source Context**: Click any "View source" button to see:

   - Document filename and path
   - Location information (page number or character range)
   - Context snippet showing the surrounding text
   - Side-by-side comparison when multiple sources exist for a candidate

4. **Select Values**:

   - Choose a candidate from the radio button list, or
   - Enter a custom value in the "Other" field (automatically validated)
   - Custom values are type-checked against the schema field type

5. **Save or Export**:
   - **Save Record**: Click "Save Record" to validate and persist to TinyDB
     - Shows success message with record ID
   - **Export as JSON**: Click "Export as JSON" to download the record as a JSON file
     - Includes resolved values, provenance, and audit trail

## Architecture

### Module Structure

```text
src/ctrlf/app/
├── models.py          # Pydantic data models (SourceRef, Candidate, FieldResult, etc.)
├── schema_io.py       # Schema validation and conversion (JSON Schema ↔ Pydantic)
├── ingest.py          # Document conversion to Markdown (markitdown wrapper)
├── extract.py         # Field extraction using langextract
├── aggregate.py       # Candidate deduplication, normalization, consensus detection
├── storage.py         # TinyDB adapter for persistent storage
├── ui.py              # Gradio interface components
├── server.py          # Application entrypoint
├── errors.py          # Error handling and error summary collection
└── logging_conf.py    # Structured logging configuration
```

### Data Flow

1. **Ingestion**: Corpus files → Markdown conversion → `CorpusDocument` with source mapping
2. **Extraction**: Markdown + Schema → Candidate generation → `Candidate` with `SourceRef`
3. **Aggregation**: Candidates → Deduplication → Normalization → Consensus detection → `FieldResult`
4. **Review**: `FieldResult` → User selection → `Resolution`
5. **Persistence**: Resolutions → Validation → `PersistedRecord` → TinyDB

### Key Data Models

- **`SourceRef`**: Document location (doc_id, path, location, snippet, metadata)
- **`Candidate`**: Extracted value with confidence score and source references
- **`FieldResult`**: Aggregated candidates for a field with optional consensus
- **`ExtractionResult`**: Complete extraction output for all fields
- **`Resolution`**: User's decision for a field (candidate or custom value)
- **`PersistedRecord`**: Final saved record with provenance and audit trail

## Configuration

### Storage Location

Records are saved to TinyDB in:

- Default: `~/.ctrlf/db/`
- One table per schema (keyed by schema hash)
- Records keyed by `record_id`
- Each record includes:
  - `resolved`: Final field values (as arrays per Extended Schema)
  - `provenance`: Source references for each field
  - `audit`: Run ID, timestamp, app version, configuration

### Consensus Thresholds

- **Confidence threshold**: 0.75 (minimum confidence for consensus)
  - Adjustable in UI (0.0 to 1.0, step 0.05)
  - Higher values require more confidence for automatic consensus
- **Margin threshold**: 0.20 (minimum margin over next candidate)
  - Fixed threshold for consensus detection
  - Ensures consensus candidate is significantly better than alternatives

### Null Policy

Controls how empty fields are handled:

- **Empty list** (default): `[]` for fields with no candidates
- **Explicit null**: `[null]` for fields with no candidates

**Note**: In v0, the null policy setting is accepted but not yet fully implemented in the extraction logic. All fields are currently stored as arrays per the Extended Schema pattern.

## Error Handling

The system handles errors gracefully:

- **Document conversion failures**: Continues processing other documents, shows warning, includes in error summary
- **Extraction errors**: Logs and continues, produces partial results
- **Validation errors**: Shows field-level messages, keeps user on review screen
- **Cancellation**: Operations can be cancelled at any time; partial results are preserved

**Error Summary**: After extraction completes, check the error output section (if visible) for a comprehensive summary of any issues encountered during processing. The summary includes:

- Document-level errors (conversion failures, unsupported formats)
- Field-level errors (extraction failures, validation issues)
- Error counts and affected documents/fields

## Development

### Setup

```bash
# Install for development (includes dev dependencies and git hooks)
make develop

# Or install without git hooks
make develop WITH_HOOKS=false
```

### Running Tests

```bash
# Run all tests with coverage
make test
# or
make check

# Run tests in parallel (faster)
make test PARALLEL=true

# Run specific test file (using pytest directly)
uv run pytest tests/unit/test_schema_io.py
```

### Code Quality

```bash
# Format and lint (runs all code quality checks)
make format-all

# Format code only
make format

# Lint code only (with auto-fix)
make lint

# Run pre-commit checks manually
make run-pre-commit
```

### Other Development Commands

```bash
# Clean build artifacts and caches
make clean

# Reinstall for development
make reinstall-dev

# View all available Makefile targets
make help
```

### Project Structure

```text
ctrlf/
├── src/ctrlf/          # Source code
│   ├── app/            # Application modules
│   └── bin/            # Scripts
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── contract/       # Contract tests
├── specs/              # Feature specifications
└── docs/               # Documentation
```

## Limitations (v0)

- **Flat schemas only**: No nested objects/arrays (only primitive types or arrays of primitives)
- **Single-user application**: No authentication or multi-user support
- **Local processing only**: No cloud/network features, all processing happens locally
- **Basic normalization**: Limited normalization for emails, dates, URLs (can be extended)
- **Batch processing**: No incremental streaming, processes entire corpus at once
- **Null policy**: Setting is accepted but not yet fully implemented in extraction logic
- **Progress tracking**: High-level progress for extraction phase (detailed per-field progress not yet implemented)

## Dependencies

### Core

- `pydantic>=2.12.2` - Data validation and models
- `gradio>=4.0.0` - Web UI framework
- `tinydb>=4.8.0` - Local JSON database
- `markitdown>=0.0.1` - Document to Markdown conversion
- `langextract>=0.1.0` - Field extraction from documents
- `thefuzz>=0.22.0` - Fuzzy string matching for deduplication
- `jsonschema>=4.0.0` - JSON Schema validation
- `structlog>=25.4.0` - Structured logging
- `python-slugify>=8.0.0` - ID generation

### Development Dependencies

- `pytest>=8.4.0` - Testing framework
- `mypy>=1.18.2` - Type checking
- `ruff>=0.12.4` - Linting
- `pylint>=3.3.2` - Code analysis

## License

See [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for code of conduct.

## Support

For issues, questions, or contributions, please open an issue on the repository.
