# Quickstart Guide

**Feature**: Schema-Grounded Corpus Extractor
**Date**: 2025-11-12

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Running the Application

```bash
# Start Gradio server
python -m ctrlf.app.server

# Or using the CLI entrypoint (when implemented)
ctrlf-server
```

The application will start on `http://localhost:7860` (default Gradio port).

## Basic Usage

### 1. Define Your Schema

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

**Note**: v0 supports only flat schemas (no nested objects/arrays).

### 2. Prepare Your Corpus

Organize your documents in a directory:

```text
corpus/
├── document1.pdf
├── document2.docx
├── document3.html
└── document4.txt
```

Or create a zip/tar archive of documents.

**Supported formats**: PDF, DOCX, HTML, TXT

### 3. Run Extraction

1. Open the application in your browser: `http://localhost:7860`
2. **Upload Schema**: Upload your `schema.json` or `schema.py` file
3. **Upload Corpus**: Upload your corpus directory or archive
4. **Configure Options** (optional):
   - Null policy: Empty list vs explicit null
   - Chunk size: Document processing chunk size
   - Confidence threshold: Consensus detection threshold
   - Dedupe sensitivity: Similarity threshold for deduplication
5. Click **"Run Extraction"**
6. Wait for processing to complete (progress indicator shows status)

### 4. Review and Resolve

1. **Review Candidates**: For each field, you'll see:

   - List of candidate values with confidence scores
   - Consensus candidate (if detected) - pre-selected
   - "View source" button next to each candidate
   - "Other" option for custom input

2. **View Source Context**: Click "View source" to see:

   - Document filename and location (page/line or char range)
   - Context snippet with highlighted span
   - Document metadata (mtime, checksum, etc.)

3. **Select Values**:

   - Choose a candidate from the list, or
   - Select "Other" and enter a custom value
   - Custom values are validated against field type

4. **Filter and Search**: Use top bar to:

   - Filter by unresolved/flagged fields
   - Search for specific field names

5. **Submit**: Click "Submit" to validate and save

### 5. View Results

After submission, you'll see:

- Summary: Number of fields resolved, nulls, disagreements
- Record ID: Unique identifier for the saved record
- Download JSON: Export the record as JSON file

## Example Workflow

```python
# 1. Schema (schema.json)
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string"},
    "amount": {"type": "number"},
    "date": {"type": "string", "format": "date"}
  }
}

# 2. Corpus: invoices/ directory with PDF invoices

# 3. Run extraction via UI

# 4. Review candidates:
#    - invoice_number: ["INV-001", "INV-002"] (consensus: INV-001)
#    - amount: [100.50, 100.5] (deduplicated, consensus: 100.50)
#    - date: ["2024-01-15", "2024-01-16"] (disagreement - no consensus)

# 5. Resolve:
#    - Select INV-001 for invoice_number
#    - Select 100.50 for amount
#    - Select "2024-01-15" for date (after viewing sources)

# 6. Submit and save to TinyDB
```

## Configuration

### Null Policy

Controls how empty fields are handled:

- **Empty list** (default): `[]` for fields with no candidates
- **Explicit null**: `[null]` for fields with no candidates

### Consensus Thresholds

- **Confidence threshold**: 0.75 (minimum confidence for consensus)
- **Margin threshold**: 0.20 (minimum margin over next candidate)

These can be adjusted in the UI before running extraction.

## Error Handling

The system handles errors gracefully:

- **Document conversion failures**: Continues processing other documents, shows warning, includes in error summary
- **Extraction errors**: Logs and continues, produces partial results
- **Validation errors**: Shows field-level messages, keeps user on review screen

Check the error summary at the end of extraction for details.

## Data Storage

Records are saved to TinyDB (local JSON database) in:

- `~/.ctrlf/db/` (default location)
- One table per schema (keyed by schema hash)
- Records keyed by `record_id`

## Exporting Data

After saving a record:

1. Click "Download JSON" button
2. JSON file contains:
   - `resolved`: Final field values (arrays)
   - `provenance`: Source references per field
   - `audit`: Run ID, timestamp, configuration

## Troubleshooting

### Schema Validation Errors

- **Nested objects/arrays**: v0 only supports flat schemas. Flatten your schema.
- **Invalid JSON Schema**: Check JSON syntax and schema format.
- **Invalid Pydantic model**: Ensure model inherits from BaseModel, no nested structures.

### Document Conversion Errors

- **Unsupported format**: Check that format is PDF, DOCX, HTML, or TXT.
- **Corrupted file**: File may be damaged. Try re-saving or use a different file.
- **Large files**: Very large documents may take longer. Check progress indicator.

### Extraction Issues

- **No candidates found**:
  - Check that documents contain the target fields
  - Verify field names match schema
  - Try adjusting extraction parameters
- **Low confidence scores**:
  - Values may be ambiguous or poorly formatted
  - Review source context to verify
  - Consider manual entry via "Other"

## Next Steps

- Review saved records in TinyDB
- Export records for external processing
- Run multiple extractions with different schemas
- Adjust consensus thresholds for your use case

## Limitations (v0)

- Flat schemas only (no nested objects/arrays)
- Single-user application (no authentication)
- Local processing only (no cloud/network)
- Basic normalization (emails, dates, URLs)
- Batch processing (no incremental streaming)

See roadmap in research.md for planned v1 features.
