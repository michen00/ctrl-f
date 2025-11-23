# Structured Extraction Draft Implementation

This is a draft implementation of a new extraction approach using structured outputs from OpenAI/Gemini.

## Overview

The `structured_extract.py` module provides an alternative extraction pipeline that:

1. **Uses Structured Outputs**: Leverages OpenAI and Gemini's native structured output capabilities with JSON Schema
2. **Per-Document Processing**: Processes each document individually with the schema passed to the API
3. **Fuzzy Character Alignment**: Uses fuzzy regex matching to find where each extraction appears in the document
4. **JSONL Output**: Creates a JSONL file with a format compatible with `langextract.visualize()`
5. **Visualization**: Can generate HTML visualizations of the extractions

## Key Components

### `ExtractionRecord`

Represents a single extraction with:

- `extraction_class`: Field name/type
- `extraction_text`: Extracted value
- `char_interval`: Character positions `{"start_pos": int, "end_pos": int}`
- `alignment_status`: `"match_exact"`, `"match_fuzzy"`, or `"no_match"`
- `extraction_index`: Sequential index
- `group_index`: Grouping index for related extractions
- `attributes`: Optional additional metadata

### `JSONLLine`

Represents one line in the JSONL output:

- `extractions`: List of `ExtractionRecord` objects
- `text`: Full document text
- `document_id`: Document identifier

### Functions

- **`find_char_interval()`**: Uses fuzzy matching (thefuzz) to locate extracted text in the document
- **`run_structured_extraction()`**: Main extraction function that processes all documents
- **`write_jsonl()`**: Writes results to JSONL file
- **`visualize_extractions()`**: Generates HTML visualization using `langextract.visualize()`

## Current Status

### ✅ Implemented

- Data models matching the JSONL format
- Fuzzy character interval finding
- JSONL file writing
- Visualization wrapper (calls `langextract.visualize()`)
- Schema flattening for nested structures

### ⚠️ TODO (Placeholder)

- **API Integration**: `_call_structured_extraction_api()` is a placeholder
  - Needs OpenAI client integration with `response_format={"type": "json_schema"}`
  - Needs Gemini client integration with `response_schema`
  - Requires API keys and proper error handling

## Usage Example

```python
from ctrlf.app.structured_extract import (
    run_structured_extraction,
    write_jsonl,
    visualize_extractions,
)
from ctrlf.app.ingest import process_corpus
from ctrlf.app.schema_io import convert_json_schema_to_pydantic
import json

# Load schema and corpus
schema_json = '''
{
  "type": "object",
  "properties": {
    "character": {"type": "string"},
    "emotion": {"type": "string"},
    "relationship": {"type": "string"}
  }
}
'''
schema = convert_json_schema_to_pydantic(json.loads(schema_json))
corpus_docs = process_corpus("path/to/corpus")

# Run structured extraction
jsonl_lines = run_structured_extraction(
    schema=schema,
    corpus_docs=corpus_docs,
    provider="openai",  # or "gemini"
    model="gpt-4o",     # or "gemini-2.0-flash-exp"
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

## JSONL Format

Each line follows this structure:

```json
{
  "extractions": [
    {
      "extraction_class": "character",
      "extraction_text": "Lady Juliet",
      "char_interval": {"start_pos": 0, "end_pos": 11},
      "alignment_status": "match_exact",
      "extraction_index": 1,
      "group_index": 0,
      "description": null,
      "attributes": {"emotional_state": "longing"}
    }
  ],
  "text": "Lady Juliet gazed longingly at the stars...",
  "document_id": "doc_bffedd4b"
}
```

## Integration Notes

This module is designed to be **non-interfering** with existing extraction logic:

- Uses separate functions and models
- Doesn't modify existing `extract.py` or `aggregate.py`
- Can be used alongside or instead of the current extraction pipeline
- No changes to existing imports or dependencies

## Next Steps

1. Implement actual API calls in `_call_structured_extraction_api()`
2. Add proper error handling and retry logic
3. Add configuration for API keys and model selection
4. Consider adding disambiguation/consensus logic similar to existing pipeline
5. Add tests for fuzzy matching and JSONL generation
6. Integrate with main application if desired
