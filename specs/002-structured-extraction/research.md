# Research & Technology Decisions

**Feature**: Structured Extraction with OpenAI/Gemini API Integration
**Date**: 2025-01-27

## Technology Stack Decisions

### OpenAI API Client

**Decision**: Use `openai` Python package (official OpenAI SDK) for structured outputs.

**Rationale**:

- Official SDK maintained by OpenAI with best support for latest features
- Native support for structured outputs via `response_format` parameter
- Handles authentication, retries, and error handling
- Version 1.0+ supports structured outputs with JSON Schema

**Alternatives considered**:

- `openai-python` (older package name) - deprecated in favor of `openai`
- Direct HTTP requests - more complex, requires manual handling of auth, retries, rate limits
- Third-party wrappers - unnecessary abstraction layer

**Implementation Details**:

- Minimum version: `openai>=1.0.0` (structured outputs support)
- Authentication: API key via environment variable `OPENAI_API_KEY` or client initialization
- Structured outputs: Use `response_format={"type": "json_schema", "json_schema": {"schema": schema, "strict": True}}`
- Model support: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` (structured outputs supported)

### Gemini API Client

**Decision**: Use `google-genai` package (already in dependencies) for structured outputs.

**Rationale**:

- Already included in project dependencies (`google-genai>=1.3.0`)
- Official Google SDK with native structured output support
- Consistent with existing codebase dependencies
- Supports `response_schema` in generation config

**Alternatives considered**:

- `google-generativeai` (alternative package) - same package, different import name
- Direct HTTP requests - more complex, requires manual handling

**Implementation Details**:

- Use existing `google-genai>=1.3.0` dependency
- Authentication: API key via environment variable `GOOGLE_API_KEY` or client configuration
- Structured outputs: Use `response_schema` in `GenerationConfig` with `response_mime_type="application/json"`
- Model support: `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`

### API Rate Limiting and Retry Strategy

**Decision**: Implement exponential backoff retry logic with configurable max retries.

**Rationale**:

- Both OpenAI and Gemini APIs have rate limits that can be hit during batch processing
- Exponential backoff prevents overwhelming the API and respects rate limits
- Configurable retries allow users to balance between reliability and speed
- Standard practice for API integrations

**Implementation Details**:

- Use exponential backoff: initial delay 1s, multiplier 2, max delay 60s
- Default max retries: 3 attempts
- Retry on: rate limit errors (429), timeout errors, temporary server errors (5xx)
- Don't retry on: authentication errors (401), invalid request errors (400), schema validation errors
- Log retry attempts with structured logging

**Alternatives considered**:

- Fixed delay retries - less efficient, doesn't adapt to API conditions
- No retries - poor user experience when rate limits are hit
- Third-party retry libraries (tenacity, backoff) - additional dependency, but could be considered if complexity grows

### Token Limit Handling

**Decision**: Detect token limits before API calls, split documents if necessary, or skip with clear error.

**Rationale**:

- Both APIs have token limits (context window sizes)
- Large documents may exceed limits, causing API errors
- Proactive detection prevents wasted API calls
- Document splitting allows processing of large documents

**Implementation Details**:

- Estimate tokens: ~4 characters per token (rough estimate, can use `tiktoken` for OpenAI if needed)
- OpenAI limits: `gpt-4o` ~128k tokens, `gpt-3.5-turbo` ~16k tokens
- Gemini limits: `gemini-2.0-flash-exp` ~1M tokens, `gemini-1.5-pro` ~2M tokens
- Strategy: If document exceeds limit, split into chunks with overlap, process separately, merge results
- If splitting not feasible (too large), skip with clear error message
- Log token estimates and splitting decisions

**Alternatives considered**:

- Truncate documents - loses information, not acceptable
- Always split - unnecessary overhead for small documents
- Fail silently - poor user experience

### Schema Compatibility and Limitations

**Decision**: Validate schema compatibility before API calls, support nested objects and arrays, flatten for extraction.

**Rationale**:

- Both APIs support nested JSON Schema structures
- Existing codebase already handles schema flattening in `structured_extract.py`
- Validation prevents API errors and provides clear feedback
- Flattening allows consistent extraction format regardless of schema complexity

**Implementation Details**:

- Validate JSON Schema format before API calls
- Support nested objects and arrays (both APIs support this)
- Flatten nested structures for consistent extraction format (already implemented in draft)
- Handle array fields: extract all items, create separate extraction records
- Handle nested objects: flatten with dot notation (e.g., `address.city`)
- Log schema validation results

**Alternatives considered**:

- Reject nested schemas - too restrictive, APIs support them
- No validation - poor error messages, wasted API calls
- Different handling per provider - inconsistent user experience

### API Key Management

**Decision**: Support environment variables as primary method, with optional config file support.

**Rationale**:

- Environment variables are standard practice for API keys
- Secure (not committed to version control)
- Easy to configure in different environments
- Config file provides alternative for users who prefer it

**Implementation Details**:

- Environment variables: `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- Check for keys at initialization, provide clear error if missing
- Optional config file: `~/.ctrlf/config.toml` or similar (future enhancement)
- UI input: Consider adding to Gradio UI for user convenience (future enhancement)
- Never log or expose API keys in logs or error messages

**Alternatives considered**:

- Only environment variables - less flexible
- Only config file - less standard, harder to use in automation
- Hardcoded keys - security risk, not acceptable

### Cost Tracking and Estimation

**Decision**: Log token usage and provide cost estimation helpers (optional, not required for v0).

**Rationale**:

- API calls have costs that users should be aware of
- Token counting enables cost estimation
- Logging provides audit trail
- Estimation helpers improve user experience

**Implementation Details**:

- Log token counts (input and output) for each API call
- Provide helper functions for cost estimation (optional)
- Use published pricing: OpenAI ~$0.01-0.03 per 1k tokens, Gemini ~$0.00025-0.002 per 1k tokens
- Don't block functionality if cost tracking fails
- Consider adding cost warnings for large corpora (future enhancement)

**Alternatives considered**:

- No cost tracking - poor user experience, unexpected bills
- Required cost tracking - adds complexity, may not be needed for all users
- Real-time cost limits - too complex for v0

### Parallel Processing

**Decision**: Support optional parallel processing with rate limit awareness (future enhancement, not required for v0).

**Rationale**:

- Large corpora benefit from parallel processing
- Rate limits must be respected
- Complexity may not be justified for v0

**Implementation Details**:

- v0: Sequential processing (simpler, respects rate limits automatically)
- Future: Add parallel processing with rate limit throttling
- Use `concurrent.futures.ThreadPoolExecutor` or `asyncio` for parallelization
- Implement rate limit throttling to prevent API errors

**Alternatives considered**:

- Always parallel - may hit rate limits, complex error handling
- No parallel support - slower for large corpora, but acceptable for v0

## Integration Patterns

### Non-Interference with Existing Extraction

**Decision**: Keep `structured_extract.py` as separate module, don't modify `extract.py`.

**Rationale**:

- Maintains backward compatibility
- Allows both extraction methods to coexist
- Easier to test and maintain
- Users can choose which method to use

**Implementation Details**:

- `structured_extract.py` is standalone module
- Reuses existing modules: `schema_io.py`, `ingest.py`, `models.py`
- No modifications to `extract.py` or `aggregate.py`
- Can be called independently or integrated into UI as alternative option

### Visualization Integration

**Decision**: Use `langextract.visualize()` for HTML visualization (already implemented in draft).

**Rationale**:

- `langextract` already in dependencies
- Provides proven visualization capabilities
- JSONL format matches `langextract` expectations
- No need to build custom visualization

**Implementation Details**:

- Generate JSONL files in format compatible with `langextract.visualize()`
- Call `lx.visualize(jsonl_path)` to generate HTML
- Handle different return types (string vs object with `.data` attribute)
- Save HTML to file or return as string

## Resolved Clarifications

All NEEDS CLARIFICATION items from Technical Context have been resolved:

1. ✅ OpenAI client version and authentication: Use `openai>=1.0.0`, environment variable `OPENAI_API_KEY`
2. ✅ Rate limits and retry strategies: Exponential backoff, 3 retries, handle 429/5xx errors
3. ✅ Token limit handling: Detect limits, split documents if needed, skip with error if too large
4. ✅ Schema compatibility: Support nested structures, validate before API calls, flatten for extraction
5. ✅ API key management: Environment variables primary, optional config file future enhancement
6. ✅ Cost tracking: Log token usage, optional cost estimation helpers
7. ✅ Parallel processing: Sequential for v0, parallel as future enhancement
