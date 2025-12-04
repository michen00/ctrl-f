# Quickstart Guide: Structured Extraction with OpenAI/Gemini

**Feature**: Structured Extraction with OpenAI/Gemini API Integration
**Date**: 2025-01-27

## Overview

This feature provides the primary extraction pipeline that uses PydanticAI with Ollama (default), OpenAI, or Gemini to extract data from documents. It generates JSONL files compatible with `langextract.visualize()` for visualization.

**Why not langextract**: langextract requires in-context examples (few-shot learning) and cannot condition extraction directly on the schema like modern APIs. PydanticAI allows us to pass the schema directly as a Pydantic model, eliminating the need for example generation and providing better schema adherence.

## Prerequisites

1. **Python 3.12+** installed
2. **API Keys**:
   - OpenAI API key (for OpenAI provider): Set `OPENAI_API_KEY` environment variable
   - Google API key (for Gemini provider): Set `GOOGLE_API_KEY` environment variable
3. **Dependencies**: Install project dependencies (see main README.md)

## Installation

1. Clone the repository (if not already done)
2. Install dependencies:

   ```bash
   make develop
   ```

3. Install additional dependencies (if not already in project):

   ```bash
   uv add openai  # For OpenAI support
   # google-genai is already in dependencies
   ```

## Basic Usage

### 1. Set Up API Keys

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Gemini (optional, if using Gemini provider)
export GOOGLE_API_KEY="your-google-api-key"
```

### 2. Prepare Your Schema

Create a JSON Schema file or Pydantic model:

**JSON Schema example** (`schema.json`):

```json
{
  "type": "object",
  "properties": {
    "character": { "type": "string" },
    "emotion": { "type": "string" },
    "relationship": { "type": "string" }
  }
}
```

**Pydantic model example** (`schema.py`):

```python
from pydantic import BaseModel

class ExtractionSchema(BaseModel):
    character: str
    emotion: str
    relationship: str
```

### 3. Prepare Your Corpus

Place your documents in a directory or archive (ZIP, TAR, TAR.GZ):

- Supported formats: PDF, DOCX, HTML, TXT (via markitdown)
- Can be a directory of files or an archive

### 4. Run Structured Extraction

**Python code example**:

```python
from ctrlf.app.structured_extract import (
    run_structured_extraction,
    write_jsonl,
    visualize_extractions,
)
from ctrlf.app.ingest import process_corpus
from ctrlf.app.schema_io import convert_json_schema_to_pydantic
import json

# Load schema
with open("schema.json") as f:
    schema_dict = json.load(f)
schema = convert_json_schema_to_pydantic(schema_dict)

# Process corpus
corpus_docs = process_corpus("path/to/corpus")

# Run structured extraction with Ollama (default, no API key needed)
jsonl_lines = run_structured_extraction(
    schema=schema,
    corpus_docs=corpus_docs,
    provider="ollama",  # Can omit this parameter to use default
    model="llama3",  # Optional, uses default if omitted
    fuzzy_threshold=80,
)

# Write JSONL file
write_jsonl(jsonl_lines, "extraction_results.jsonl")

# Visualize
html_content = visualize_extractions(
    "extraction_results.jsonl",
    output_html_path="visualization.html",
)
```

**Using OpenAI instead**:

```python
jsonl_lines = run_structured_extraction(
    schema=schema,
    corpus_docs=corpus_docs,
    provider="openai",
    model="gpt-4o",  # Optional, uses default if omitted
    fuzzy_threshold=80,
)
```

**Using Gemini instead**:

```python
jsonl_lines = run_structured_extraction(
    schema=schema,
    corpus_docs=corpus_docs,
    provider="gemini",
    model="gemini-2.5-flash",  # Optional, uses default if omitted
    fuzzy_threshold=80,
)
```

## Configuration Options

### Provider Selection

- **Ollama** (default): Use `provider="ollama"` or omit provider parameter
  - Models: `"llama3"` (default), `"llama3.2"`, `"mistral"`, etc.
  - No API key required (local only)
  - Best for local development and testing

- **OpenAI**: Use `provider="openai"`
  - Models: `"gpt-4o"` (default), `"gpt-4-turbo"`, `"gpt-3.5-turbo"`
  - Requires `OPENAI_API_KEY` environment variable

- **Gemini**: Use `provider="gemini"`
  - Models: `"gemini-2.5-flash"` (default), `"gemini-2.0-flash-exp"`, `"gemini-1.5-pro"`, `"gemini-1.5-flash"`
  - Requires `GOOGLE_API_KEY` environment variable

### Fuzzy Matching Threshold

- `fuzzy_threshold: int` (default: 80)
- Range: 0-100
- Higher values require more exact matches
- Lower values allow more fuzzy matches
- Recommended: 80 for balanced accuracy

### Model Selection

- **OpenAI**: `"gpt-4o"` (recommended), `"gpt-4-turbo"`, `"gpt-3.5-turbo"`
- **Gemini**: `"gemini-2.0-flash-exp"` (recommended), `"gemini-1.5-pro"`, `"gemini-1.5-flash"`

## Output Format

### JSONL File Structure

Each line in the JSONL file represents one document:

```json
{
  "extractions": [
    {
      "extraction_class": "character",
      "extraction_text": "Lady Juliet",
      "char_interval": { "start_pos": 0, "end_pos": 11 },
      "alignment_status": "match_exact",
      "extraction_index": 1,
      "group_index": 0,
      "description": null,
      "attributes": null
    }
  ],
  "text": "Lady Juliet gazed longingly at the stars...",
  "document_id": "doc_bffedd4b"
}
```

### Alignment Status Values

- `"match_exact"`: Exact match found in document (case-sensitive or case-insensitive)
- `"match_fuzzy"`: Fuzzy match found (similarity >= threshold)
- `"no_match"`: No match found (char_interval will be {0, 0})

## Visualization

The generated JSONL file can be visualized using `langextract.visualize()`:

```python
from ctrlf.app.structured_extract import visualize_extractions

# Generate HTML visualization
html_content = visualize_extractions(
    "extraction_results.jsonl",
    output_html_path="visualization.html",
)

# Open visualization.html in a web browser
```

## Troubleshooting

### API Key Errors

**Error**: `ValueError: API key not found`

**Solution**: Set the appropriate environment variable:

```bash
export OPENAI_API_KEY="your-key"  # For OpenAI
export GOOGLE_API_KEY="your-key"  # For Gemini
```

### Rate Limit Errors

**Error**: `429 Too Many Requests`

**Solution**: The system automatically retries with exponential backoff. If errors persist:

- Reduce corpus size or process in batches
- Use a model with higher rate limits
- Wait and retry later

### Token Limit Errors

**Error**: `400 Bad Request: Token limit exceeded`

**Solution**:

- Split large documents into smaller chunks
- Use a model with larger context window (e.g., `gpt-4o` or `gemini-2.0-flash-exp`)
- Pre-process documents to remove unnecessary content

### No Matches Found

**Issue**: Many extractions have `"alignment_status": "no_match"`

**Solution**:

- Lower the `fuzzy_threshold` (e.g., 70 instead of 80)
- Check if extraction text matches document text (may be API formatting issue)
- Review API response to ensure correct extraction

### Schema Validation Errors

**Error**: `ValueError: Invalid schema`

**Solution**:

- Ensure schema is valid JSON Schema format
- Check that schema matches API capabilities (both OpenAI and Gemini support nested structures)
- Validate schema using `schema_io.validate_json_schema()`

## Advanced Usage

### Custom Retry Logic

The system uses exponential backoff with 3 retries by default. To customize (future enhancement):

- Configure `max_retries` parameter
- Adjust retry delays
- Customize error handling

### Cost Tracking

Token usage is logged for each API call. To estimate costs:

- Check logs for token counts
- Use published pricing: OpenAI ~$0.01-0.03 per 1k tokens, Gemini ~$0.00025-0.002 per 1k tokens
- Monitor API usage dashboard

### Parallel Processing

Currently, documents are processed sequentially. For large corpora:

- Process in batches manually
- Future versions will support parallel processing with rate limit throttling

## Integration with Existing Pipeline

This feature is designed to be non-interfering with the existing `extract.py` pipeline:

- Uses separate module: `structured_extract.py`
- Different output format: JSONL instead of `ExtractionResult`
- Can be used alongside existing extraction
- No modifications to existing code

## Next Steps

1. Review generated JSONL file
2. Visualize extractions using `visualize_extractions()`
3. Integrate with downstream processing tools
4. Consider adding to Gradio UI for interactive use (future enhancement)

## Support

For issues or questions:

- Check error logs for detailed error messages
- Review function contracts in `contracts/function-contracts.md`
- Consult data model documentation in `data-model.md`
