"""Structured extraction module using PydanticAI with Ollama/OpenAI/Gemini.

This module provides the primary extraction approach that:
1. Uses PydanticAI Agent with Pydantic models for structured outputs
2. Supports Ollama (default), OpenAI, and Gemini providers
3. Processes each document individually with the Extended Schema model
4. Uses fuzzy regex to find character positions in documents
5. Outputs results in JSONL format for visualization (via langextract.visualize())

Note: langextract is now only used for visualization, not for extraction.
PydanticAI unifies schema handling across all LLM providers.

Example usage:

    from ctrlf.app.structured_extract import (
        run_structured_extraction,
        write_jsonl,
        visualize_extractions,
    )
    from ctrlf.app.ingest import process_corpus
    from ctrlf.app.schema_io import convert_json_schema_to_pydantic

    # Load schema and corpus
    schema_json = '{"type": "object", "properties": {"character": {"type": "string"}}}'
    schema = convert_json_schema_to_pydantic(json.loads(schema_json))
    corpus_docs = process_corpus("path/to/corpus")

    # Run structured extraction (Ollama is default)
    jsonl_lines = run_structured_extraction(
        schema=schema,
        corpus_docs=corpus_docs,
        provider="ollama",
        model="llama3",
    )

    # Write JSONL file
    write_jsonl(jsonl_lines, "extraction_results.jsonl")

    # Visualize
    html_content = visualize_extractions(
        "extraction_results.jsonl",
        output_html_path="visualization.html",
    )
"""

from __future__ import annotations

__all__ = (
    "FlattenedExtraction",
    "_call_structured_extraction_api",
    "_extraction_record_to_candidate",
    "check_ollama_setup",
    "find_char_interval",
    "run_structured_extraction",
    "visualize_extractions",
    "write_jsonl",
)

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from pydantic import BaseModel
from thefuzz import fuzz

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import Candidate, SourceRef
from ctrlf.app.schema_io import validate_json_schema

if TYPE_CHECKING:
    from collections.abc import Callable

    from ctrlf.app.ingest import CorpusDocument

logger = get_logger(__name__)

T = TypeVar("T")


def estimate_cost(
    tokens_input: int,
    tokens_output: int,
    provider: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Estimate cost for API call based on token usage.

    Provides cost estimation for OpenAI and Gemini providers based on published
    pricing (as of 2024). Ollama is free (local only).

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        provider: API provider ("ollama", "openai", or "gemini")
        model: Model name (optional, for more accurate pricing)

    Returns:
        Dictionary with cost estimation:
        - total_tokens: Total token count
        - input_cost: Cost for input tokens (USD)
        - output_cost: Cost for output tokens (USD)
        - total_cost: Total cost (USD)
        - provider: Provider name
        - model: Model name used
        - note: Additional notes about pricing

    Note:
        Pricing is approximate and may vary. Check provider pricing pages for
        current rates. Ollama returns zero cost (local only).
    """
    total_tokens = tokens_input + tokens_output

    # Pricing per 1M tokens (as of 2024, approximate)
    # OpenAI: https://openai.com/pricing
    # Gemini: https://ai.google.dev/pricing
    pricing: dict[str, dict[str, dict[str, float]]] = {
        "openai": {
            "gpt-4o": {"input": 2.50, "output": 10.00},  # $2.50/$10 per 1M tokens
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "default": {"input": 2.50, "output": 10.00},
        },
        "gemini": {
            "gemini-2.5-flash": {
                "input": 0.075,
                "output": 0.30,
            },  # $0.075/$0.30 per 1M tokens
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "default": {"input": 0.075, "output": 0.30},
        },
        "ollama": {
            "default": {"input": 0.0, "output": 0.0},  # Free (local only)
        },
    }

    if provider == "ollama":
        return {
            "total_tokens": total_tokens,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "provider": provider,
            "model": model or "llama3",
            "note": "Ollama is free (local only, no API costs)",
        }

    provider_pricing = pricing.get(provider, {})
    model_name = model or (
        "gpt-4o"
        if provider == "openai"
        else ("gemini-2.5-flash" if provider == "gemini" else "default")
    )
    model_pricing = provider_pricing.get(
        model_name, provider_pricing.get("default", {"input": 0.0, "output": 0.0})
    )

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (tokens_input / 1_000_000) * model_pricing["input"]
    output_cost = (tokens_output / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "total_tokens": total_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "provider": provider,
        "model": model_name,
        "note": (
            "Pricing is approximate. Check provider pricing pages for current rates."
        ),
    }


# not sure why we're doing this in Python
def check_ollama_setup(model: str | None = None) -> None:
    """Check if Ollama is running and the specified model is available.

    Args:
        model: Model name to check (defaults to "llama3" if not specified)

    Raises:
        RuntimeError: If Ollama is not running or model is not available
    """
    import shutil  # noqa: PLC0415  # nosec import_subprocess
    import subprocess  # noqa: PLC0415  # nosec import_subprocess

    model_name = model or "llama3"

    # Find ollama executable
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        msg = (
            "Ollama is not installed. "
            "Please install Ollama from https://ollama.ai. "
            f"After installation, start Ollama and pull the model with: "
            f"ollama pull {model_name}"
        )
        raise RuntimeError(msg)

    # Check if Ollama is running
    # Note: ollama_path is from shutil.which() which validates the executable exists
    # We're only calling "ollama list" which is safe
    try:
        result = subprocess.run(  # noqa: S603
            [ollama_path, "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            msg = (
                f"Ollama is not running or not installed. "
                f"Please start Ollama or install it from https://ollama.ai. "
                f"Error: {result.stderr}"
            )
            raise RuntimeError(msg)
    except subprocess.TimeoutExpired:
        msg = (
            "Ollama is not responding. "
            "Please ensure Ollama is running. "
            "Start it with: ollama serve"
        )
        raise RuntimeError(msg) from None

    # Check if the model is available
    if model_name not in result.stdout:
        msg = (
            f"Ollama model '{model_name}' is not available. "
            f"Available models: {result.stdout.strip() or '(none)'}. "
            f"Pull the model with: ollama pull {model_name}"
        )
        raise RuntimeError(msg)


def validate_api_key(provider: str) -> None:
    """Validate API key for the given provider.

    Checks for required API keys based on provider:
    - Ollama: No API key required (local only), but checks if Ollama is running
    - OpenAI: Requires OPENAI_API_KEY environment variable
    - Gemini: Requires GOOGLE_API_KEY environment variable

    Args:
        provider: API provider ("ollama", "openai", or "gemini")

    Raises:
        ValueError: If provider is invalid or API key is missing/invalid
        RuntimeError: If Ollama is not running or model is not available
    """
    import os  # noqa: PLC0415

    if provider == "ollama":
        # Ollama doesn't need API keys (local only), but check if it's running
        # We'll check for the default model when extraction is called
        return

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            msg = (
                "OPENAI_API_KEY environment variable is required for OpenAI provider. "
                "Set it with: export OPENAI_API_KEY='your-key'"
            )
            raise ValueError(msg)
        # Basic validation: OpenAI keys typically start with "sk-"
        if not api_key.startswith("sk-"):
            logger.warning(
                "OPENAI_API_KEY does not start with 'sk-'. "
                "This may indicate an invalid key format."
            )

    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            msg = (
                "GOOGLE_API_KEY environment variable is required for Gemini provider. "
                "Set it with: export GOOGLE_API_KEY='your-key'"
            )
            raise ValueError(msg)
        # Basic validation: Google API keys are typically long alphanumeric strings
        if len(api_key) < 20:
            logger.warning(
                "GOOGLE_API_KEY appears to be too short. "
                "This may indicate an invalid key format."
            )

    else:
        msg = f"Unsupported provider: {provider}. Supported: ollama, openai, gemini"
        raise ValueError(msg)


def _retry_with_exponential_backoff[T](  # noqa: PLR0913
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    retryable_errors: tuple[type[Exception], ...] | None = None,
) -> T:
    """Retry a function with exponential backoff.

    Handles retryable errors (rate limits, timeouts, server errors) with
    exponential backoff. Non-retryable errors (authentication, invalid requests)
    are raised immediately.

    Args:
        func: Function to retry (callable)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        multiplier: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        retryable_errors: Tuple of exception types to retry on. If None, uses default
            retryable errors (RateLimitError, TimeoutError, ServerError)

    Returns:
        Return value from func()

    Raises:
        Exception: Last exception raised by func() if all retries exhausted
    """
    import time  # noqa: PLC0415

    # Default retryable errors: rate limits (429), timeouts, server errors (5xx)
    # Note: PydanticAI may wrap these in RuntimeError, so we'll check error messages
    if retryable_errors is None:
        retryable_errors = (TimeoutError,)

    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if error is retryable
            is_retryable = isinstance(e, retryable_errors)

            # Also check error messages for common retryable patterns
            error_msg = str(e).lower()
            if not is_retryable:
                is_retryable = any(
                    pattern in error_msg
                    for pattern in [
                        "rate limit",
                        "429",
                        "timeout",
                        "timed out",
                        "server error",
                        "500",
                        "502",
                        "503",
                        "504",
                    ]
                )

            # Don't retry on authentication errors (401) or invalid requests (400)
            if any(
                pattern in error_msg
                for pattern in [
                    "authentication",
                    "401",
                    "unauthorized",
                    "invalid request",
                    "400",
                    "bad request",
                ]
            ):
                logger.exception("Non-retryable error (authentication/invalid request)")
                raise

            # If not retryable or out of retries, raise
            if not is_retryable or attempt >= max_retries:
                if attempt >= max_retries:
                    logger.exception(
                        "Max retries (%d) exceeded for %s",
                        max_retries,
                        func.__name__ if hasattr(func, "__name__") else "function",
                    )
                raise

            # Log retry attempt
            logger.warning(
                "Retryable error (attempt %d/%d): %s. Retrying in %.1f seconds...",
                attempt + 1,
                max_retries,
                e,
                delay,
            )

            # Wait before retry
            time.sleep(delay)

            # Calculate next delay with exponential backoff
            delay = min(delay * multiplier, max_delay)

    # Should never reach here, but handle just in case
    if last_exception:
        raise last_exception
    msg = "Retry logic failed unexpectedly"
    raise RuntimeError(msg)


class FlattenedExtraction(NamedTuple):
    """Represents a flattened extraction from structured data.

    Attributes:
        field_name: Full field name (may include prefix for nested fields)
        value: Extracted value as string
        attributes: Optional attributes dictionary (e.g., index for array items)
    """

    field_name: str
    value: str
    attributes: dict[str, Any] | None


class ExtractionRecord(BaseModel):
    """Represents a single extraction with character interval information.

    Attributes:
        extraction_class: The class/type of extraction (field name)
        extraction_text: The extracted text value
        char_interval: Character position interval in the document
        alignment_status: Status of alignment (match_exact, match_fuzzy, no_match)
        extraction_index: Index of this extraction in the sequence
        group_index: Group index for related extractions
        description: Optional description
        attributes: Optional additional attributes
    """

    extraction_class: str
    extraction_text: str
    char_interval: dict[str, int]  # {"start_pos": int, "end_pos": int}
    alignment_status: str
    extraction_index: int
    group_index: int
    description: str | None = None
    attributes: dict[str, Any] | None = None


class JSONLLine(BaseModel):
    """Represents a single line in the JSONL output.

    Attributes:
        extractions: List of extractions for this document
        text: The full document text
        document_id: Unique identifier for the document
    """

    extractions: list[ExtractionRecord]
    text: str
    document_id: str


def find_char_interval(
    text: str,
    extraction_text: str,
    fuzzy_threshold: int = 80,
) -> tuple[dict[str, int], str]:
    """Find character interval for an extraction using fuzzy regex matching.

    Args:
        text: Full document text to search in
        extraction_text: The extracted text to locate
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching

    Returns:
        Tuple of (char_interval dict, alignment_status)
        char_interval: {"start_pos": int, "end_pos": int}
        alignment_status: "match_exact", "match_fuzzy", or "no_match"
    """
    # Handle empty extraction text
    if not extraction_text:
        return ({"start_pos": 0, "end_pos": 0}, "no_match")

    # First try exact match
    exact_match = text.find(extraction_text)
    if exact_match != -1:
        return (
            {"start_pos": exact_match, "end_pos": exact_match + len(extraction_text)},
            "match_exact",
        )

    # Try case-insensitive exact match
    text_lower = text.lower()
    extraction_lower = extraction_text.lower()
    case_insensitive_match = text_lower.find(extraction_lower)
    if case_insensitive_match != -1:
        return (
            {
                "start_pos": case_insensitive_match,
                "end_pos": case_insensitive_match + len(extraction_text),
            },
            "match_exact",
        )

    # Try fuzzy matching using sliding window
    extraction_len = len(extraction_text)
    best_match_pos = -1
    best_match_score = 0
    window_size = min(extraction_len * 2, len(text))

    # Slide window through text
    for i in range(len(text) - window_size + 1):
        window = text[i : i + window_size]
        score = fuzz.partial_ratio(extraction_text, window)
        if score > best_match_score:
            best_match_score = score
            best_match_pos = i

    # If we found a good fuzzy match, try to find exact boundaries
    if best_match_score >= fuzzy_threshold and best_match_pos != -1:
        # Find the best substring match within the window
        window = text[best_match_pos : best_match_pos + window_size]
        # Use token-based matching to find better boundaries
        extraction_tokens = extraction_text.split()
        window_tokens = window.split()

        # Find best token alignment
        # Handle case where window has fewer tokens than extraction
        num_tokens_to_match = min(len(window_tokens), len(extraction_tokens))
        best_token_start = 0
        best_token_score = 0

        # Only iterate if we have enough tokens to slide
        if len(window_tokens) >= len(extraction_tokens):
            # Normal case: window has enough tokens, slide through
            for i in range(len(window_tokens) - len(extraction_tokens) + 1):
                window_subset = " ".join(window_tokens[i : i + len(extraction_tokens)])
                score = fuzz.ratio(extraction_text, window_subset)
                if score > best_token_score:
                    best_token_score = score
                    best_token_start = i
        else:
            # Edge case: window has fewer tokens than extraction
            # Use all available tokens and find best alignment
            window_subset = " ".join(window_tokens)
            best_token_score = fuzz.ratio(extraction_text, window_subset)
            best_token_start = 0

        # Calculate character positions from token positions
        token_start_char = len(" ".join(window_tokens[:best_token_start]))
        if best_token_start > 0:
            token_start_char += 1  # Account for space before first token

        # Use actual number of matched tokens (may be less than extraction_tokens)
        matched_text = " ".join(
            window_tokens[best_token_start : best_token_start + num_tokens_to_match]
        )
        start_pos = best_match_pos + token_start_char
        end_pos = start_pos + len(matched_text)

        return (
            {"start_pos": start_pos, "end_pos": end_pos},
            "match_fuzzy",
        )

    # No match found
    return ({"start_pos": 0, "end_pos": 0}, "no_match")


def _call_structured_extraction_api(  # noqa: PLR0915
    text: str,
    schema_model: type[BaseModel],
    provider: str = "ollama",
    model: str | None = None,
) -> dict[str, Any]:
    """Call LLM API with structured outputs using PydanticAI.

    Uses PydanticAI Agent to extract structured data from text according to the schema.
    Supports Ollama (default), OpenAI, and Gemini providers.
    Includes API key validation, schema validation, and retry logic with
    exponential backoff.

    Args:
        text: Document text to extract from
        schema_model: Pydantic model (Extended Schema) defining the output structure
        provider: API provider ("ollama", "openai", or "gemini", default: "ollama")
        model: Model name (e.g., "llama3", "gpt-4o", "gemini-2.5-flash")
            For Ollama, defaults to "llama3" if not specified
            For Gemini, defaults to "gemini-2.5-flash" if not specified

    Returns:
        Extracted data as dict matching the schema (validated Pydantic model instance)

    Raises:
        ValueError: If provider is not supported, API key is missing/invalid,
            or schema is invalid
        RuntimeError: If extraction fails after retries
    """
    # Validate API key before proceeding
    validate_api_key(provider)

    # For Ollama, check if it's running and model is available
    if provider == "ollama":
        check_ollama_setup(model)

    # Validate schema before API call
    try:
        schema_dict = schema_model.model_json_schema()
        # Validate JSON Schema format (basic validation - Pydantic model
        # already validates structure)
        schema_json_str = json.dumps(schema_dict)
        validate_json_schema(schema_json_str)
        logger.debug("Schema validation passed for %s", provider)
    except Exception as e:
        logger.exception("Schema validation failed")
        msg = f"Invalid schema: {e}. Please ensure schema is a valid Pydantic model."
        raise ValueError(msg) from e

    try:
        from pydantic_ai import Agent  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "pydantic-ai is required for structured extraction. "
            "Install with: uv add pydantic-ai (or pip install pydantic-ai). "
            "If using pydantic-ai-slim, ensure google extras are installed: "
            'pip install "pydantic-ai-slim[google]"'
        )
        raise ImportError(msg) from e

    # Determine model string based on provider
    # Format: "provider:model-name" where provider matches PydanticAI's provider
    # identifiers. Note: google-gla is the correct provider identifier (not
    # google-genai). See: https://ai.pydantic.dev/api/models/google/
    if provider == "ollama":
        model_str = f"ollama:{model or 'llama3'}"
    elif provider == "openai":
        model_str = f"openai:{model or 'gpt-4o'}"
    elif provider == "gemini":
        # google-gla is the PydanticAI provider identifier for Generative Language API
        # (package name is google-genai, but provider identifier is google-gla)
        model_str = f"google-gla:{model or 'gemini-2.5-flash'}"
    else:
        msg = f"Unsupported provider: {provider}. Supported: ollama, openai, gemini"
        raise ValueError(msg)

    def _execute_extraction() -> dict[str, Any]:
        """Execute the extraction API call (wrapped for retry logic)."""
        import time  # noqa: PLC0415

        # Estimate token count (rough estimate: ~4 characters per token)
        estimated_tokens = len(text) // 4
        estimated_schema_tokens = len(str(schema_model.model_json_schema())) // 4
        total_estimated_tokens = estimated_tokens + estimated_schema_tokens

        # Token limits by provider/model (approximate)
        token_limits: dict[str, dict[str, int]] = {
            "ollama": {"llama3": 128000, "default": 128000},
            "openai": {
                "gpt-4o": 128000,
                "gpt-4-turbo": 128000,
                "gpt-3.5-turbo": 16385,
                "default": 128000,
            },
            "gemini": {
                "gemini-2.5-flash": 1000000,
                "gemini-1.5-pro": 2000000,
                "gemini-1.5-flash": 1000000,
                "default": 1000000,
            },
        }

        # Check token limits
        provider_limits = token_limits.get(provider, {})
        model_name = model or (
            "llama3"
            if provider == "ollama"
            else ("gpt-4o" if provider == "openai" else "gemini-2.5-flash")
        )
        token_limit = provider_limits.get(
            model_name, provider_limits.get("default", 128000)
        )

        if total_estimated_tokens > token_limit:
            logger.warning(
                "Estimated token count (%d) exceeds limit (%d) for %s. "
                "Document may be truncated or fail.",
                total_estimated_tokens,
                token_limit,
                model_str,
            )

        # Log API call start
        start_time = time.time()
        logger.info(
            "Starting API call with %s (estimated tokens: %d, limit: %d)",
            model_str,
            total_estimated_tokens,
            token_limit,
        )

        try:
            # Create PydanticAI Agent with the Extended Schema as output_type
            agent = Agent(
                model_str,
                output_type=schema_model,
                system_prompt=(
                    "Extract structured data from the provided document text. "
                    "Return only the extracted data matching the specified schema. "
                    "For array fields, return all matching values found in "
                    "the document."
                ),
            )

            # Run extraction synchronously
            result = agent.run_sync(text)

            # Calculate response time
            response_time = time.time() - start_time

            # Extract token usage if available from PydanticAI result
            # PydanticAI may provide token usage in result.usage or similar
            tokens_input = (
                getattr(result, "usage", {}).get("input_tokens", estimated_tokens)
                if hasattr(result, "usage")
                else estimated_tokens
            )
            tokens_output = (
                getattr(result, "usage", {}).get(
                    "output_tokens", estimated_schema_tokens
                )
                if hasattr(result, "usage")
                else estimated_schema_tokens
            )
            total_tokens = tokens_input + tokens_output

            # Estimate cost (optional, for logging)
            cost_estimate = estimate_cost(tokens_input, tokens_output, provider, model)

            # Log API call success with token usage, response time, and cost estimate
            logger.info(
                "API call completed successfully with %s: %d fields extracted, "
                "tokens: %d input + %d output = %d total, response time: %.2fs, "
                "estimated cost: $%.6f",
                model_str,
                len(result.output.model_dump()),
                tokens_input,
                tokens_output,
                total_tokens,
                response_time,
                cost_estimate["total_cost"],
            )

            # Log detailed cost breakdown at debug level
            logger.debug(
                "Cost breakdown: input=$%.6f, output=$%.6f, total=$%.6f (%s)",
                cost_estimate["input_cost"],
                cost_estimate["output_cost"],
                cost_estimate["total_cost"],
                cost_estimate["note"],
            )

            # Convert Pydantic model instance to dict
            extracted_data = result.output.model_dump()

            logger.debug(
                "Extracted data for document using %s: %s fields extracted",
                model_str,
                len(extracted_data),
            )
        except Exception:
            # Calculate response time even on error
            response_time = time.time() - start_time
            logger.exception(
                "API call failed with %s after %.2fs",
                model_str,
                response_time,
            )
            raise
        else:
            return extracted_data

    # Execute with retry logic
    try:
        return _retry_with_exponential_backoff(
            _execute_extraction,
            max_retries=3,
            initial_delay=1.0,
            multiplier=2.0,
            max_delay=60.0,
        )
    except Exception as e:
        logger.exception(
            "Structured extraction failed with %s after retries", model_str, exc_info=e
        )
        # Provide helpful error messages for common issues
        error_str = str(e).lower()
        if provider == "ollama":
            if "404" in error_str or "not found" in error_str or "model" in error_str:
                model_name = model or "llama3"
                msg = (
                    f"Ollama model '{model_name}' is not available. "
                    f"Error: {e}. "
                    f"Pull the model with: ollama pull {model_name} "
                    f"Or check available models with: ollama list"
                )
            elif "connection" in error_str or "refused" in error_str:
                msg = (
                    f"Ollama is not running. "
                    f"Error: {e}. "
                    f"Start Ollama with: ollama serve"
                )
            else:
                msg = f"Extraction failed: {e}"
        else:
            msg = f"Extraction failed: {e}"
        raise RuntimeError(msg) from e


def _extraction_record_to_candidate(
    extraction_record: ExtractionRecord,
    doc_id: str,
    doc_markdown: str,
    source_map: dict[str, Any],
) -> Candidate:
    """Convert ExtractionRecord to Candidate format for existing pipeline.

    Args:
        extraction_record: ExtractionRecord from structured extraction
        doc_id: Document identifier
        doc_markdown: Full document markdown text
        source_map: Source mapping from document

    Returns:
        Candidate object compatible with existing extraction pipeline
    """
    # Map alignment_status to confidence score
    # match_exact = high confidence (0.9)
    # match_fuzzy = medium (0.7)
    # no_match = low (0.5)
    confidence_map = {
        "match_exact": 0.9,
        "match_fuzzy": 0.7,
        "no_match": 0.5,
    }
    confidence = confidence_map.get(extraction_record.alignment_status, 0.5)

    # Extract character positions
    start_pos = extraction_record.char_interval.get("start_pos", 0)
    end_pos = extraction_record.char_interval.get("end_pos", start_pos)

    # Create location descriptor
    location = f"char-range [{start_pos}:{end_pos}]"
    if "pages" in source_map:
        pages = source_map.get("pages", {})
        if isinstance(pages, dict):
            for page_num, page_info in pages.items():
                if isinstance(page_info, dict):
                    page_start = page_info.get("start", 0)
                    page_end = page_info.get("end", float("inf"))
                    if page_start <= start_pos <= page_end:
                        location = f"page {page_num}"
                        break

    # Extract snippet (similar to extract.py logic)
    context = 50
    snippet_start = max(0, start_pos - context)
    snippet_end = min(len(doc_markdown), end_pos + context)
    snippet = doc_markdown[snippet_start:snippet_end]
    if len(snippet) < 7:  # MIN_SNIPPET_LENGTH
        snippet = doc_markdown[:100] if len(doc_markdown) > 100 else doc_markdown

    # Create SourceRef
    source_ref = SourceRef(
        doc_id=doc_id,
        path=source_map.get("file_path", source_map.get("file_name", "unknown")),
        location=location,
        snippet=snippet,
        metadata={
            "span_start": start_pos,
            "span_end": end_pos,
            "alignment_status": extraction_record.alignment_status,
            **source_map,
        },
    )

    # Create Candidate
    return Candidate(
        value=extraction_record.extraction_text,
        normalized=None,  # Will be normalized in aggregate.py
        confidence=confidence,
        sources=[source_ref],
    )


def _flatten_extractions(
    data: dict[str, Any],
    schema: dict[str, Any],
    prefix: str = "",
) -> list[FlattenedExtraction]:
    """Flatten extracted data into FlattenedExtraction objects.

    Handles nested objects and arrays according to the schema.

    Args:
        data: Extracted data dict
        schema: JSON Schema definition
        prefix: Prefix for nested field names

    Returns:
        List of FlattenedExtraction objects
    """
    extractions: list[FlattenedExtraction] = []

    if "properties" not in schema:
        return extractions

    for field_name, field_schema in schema["properties"].items():
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name
        field_value = data.get(field_name)

        if field_value is None:
            continue

        field_type = field_schema.get("type")

        if field_type == "array":
            # Handle array of values
            items_schema = field_schema.get("items", {})
            if isinstance(field_value, list):
                for idx, item in enumerate(field_value):
                    if isinstance(item, str):
                        extractions.append(
                            FlattenedExtraction(full_field_name, item, {"index": idx})
                        )
                    elif isinstance(item, dict):
                        # Nested object in array
                        nested = _flatten_extractions(
                            item, items_schema, f"{full_field_name}[{idx}]"
                        )
                        extractions.extend(nested)
        elif field_type == "object":
            # Handle nested object
            nested = _flatten_extractions(field_value, field_schema, full_field_name)
            extractions.extend(nested)
        elif isinstance(field_value, str):
            # Primitive string value
            extractions.append(FlattenedExtraction(full_field_name, field_value, None))
        elif field_value is not None:
            # Convert non-string primitives to string
            extractions.append(
                FlattenedExtraction(full_field_name, str(field_value), None)
            )

    return extractions


def run_structured_extraction(
    schema: type[BaseModel],
    corpus_docs: list[CorpusDocument],
    provider: str = "ollama",
    model: str | None = None,
    fuzzy_threshold: int = 80,
) -> list[JSONLLine]:
    """Run structured extraction on corpus documents using PydanticAI.

    Args:
        schema: Pydantic model class (Extended Schema) defining the output structure
        corpus_docs: List of corpus documents
        provider: API provider ("ollama", "openai", or "gemini", default: "ollama")
        model: Model name (optional, uses provider default)
            For Ollama: defaults to "llama3"
            For OpenAI: defaults to "gpt-4o"
            For Gemini: defaults to "gemini-2.5-flash"
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)

    Returns:
        List of JSONLLine objects, one per document

    Raises:
        ValueError: If schema is not a Pydantic model
        RuntimeError: If extraction fails
    """
    # Schema validation: function signature enforces type[BaseModel], so no runtime
    # check needed. The type system guarantees schema is a BaseModel subclass.

    # Get JSON Schema dict for flattening
    schema_dict = schema.model_json_schema()

    jsonl_lines: list[JSONLLine] = []

    for doc in corpus_docs:
        try:
            # Call structured extraction API with PydanticAI
            extracted_data = _call_structured_extraction_api(
                doc.markdown, schema, provider=provider, model=model
            )

            # Flatten extractions
            flat_extractions = _flatten_extractions(extracted_data, schema_dict)

            # Find character intervals and create ExtractionRecord objects
            extraction_records: list[ExtractionRecord] = []
            for idx, extraction in enumerate(flat_extractions):
                char_interval, alignment_status = find_char_interval(
                    doc.markdown, extraction.value, fuzzy_threshold=fuzzy_threshold
                )

                extraction_record = ExtractionRecord(
                    extraction_class=extraction.field_name,
                    extraction_text=extraction.value,
                    char_interval=char_interval,
                    alignment_status=alignment_status,
                    extraction_index=idx + 1,
                    group_index=idx,  # Simple grouping - can be enhanced
                    description=None,
                    attributes=extraction.attributes,
                )
                extraction_records.append(extraction_record)

            # Create JSONL line
            jsonl_line = JSONLLine(
                extractions=extraction_records,
                text=doc.markdown,
                document_id=doc.doc_id,
            )
            jsonl_lines.append(jsonl_line)

        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Structured extraction failed for document %s: %s", doc.doc_id, e
            )
            # Create empty extraction record for failed documents
            jsonl_line = JSONLLine(
                extractions=[],
                text=doc.markdown,
                document_id=doc.doc_id,
            )
            jsonl_lines.append(jsonl_line)

    return jsonl_lines


def write_jsonl(
    jsonl_lines: list[JSONLLine],
    output_path: str | Path,
) -> None:
    """Write JSONL lines to file.

    Args:
        jsonl_lines: List of JSONLLine objects
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for line in jsonl_lines:
            # Convert to dict and write as JSON line
            line_dict = line.model_dump()
            f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

    logger.info("Wrote %d lines to %s", len(jsonl_lines), output_path)


def _validate_jsonl_format(jsonl_path: Path) -> None:
    """Validate JSONL file format before visualization.

    Checks that the JSONL file:
    1. Exists and is readable
    2. Contains valid JSON on each line
    3. Each line has required fields (extractions, text, document_id)
    4. Format matches langextract.visualize() expectations

    Args:
        jsonl_path: Path to JSONL file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not jsonl_path.exists():
        msg = f"JSONL file not found: {jsonl_path}"
        raise FileNotFoundError(msg)

    if not jsonl_path.is_file():
        msg = f"Path is not a file: {jsonl_path}"
        raise ValueError(msg)

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        msg = f"Failed to read JSONL file {jsonl_path}: {e}"
        raise ValueError(msg) from e

    if not lines:
        logger.warning("JSONL file is empty: %s", jsonl_path)
        return

    # Validate each line
    for line_num, raw_line in enumerate(lines, start=1):
        line_content = raw_line.strip()
        if not line_content:
            continue  # Skip empty lines

        try:
            data = json.loads(line_content)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON on line {line_num} of {jsonl_path}: {e}"
            raise ValueError(msg) from e

        # Check required fields for langextract.visualize() compatibility
        required_fields = ["extractions", "text", "document_id"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            msg = (
                f"Missing required fields on line {line_num} of {jsonl_path}: "
                f"{missing_fields}. Required: {required_fields}"
            )
            raise ValueError(msg)

        # Validate field types
        if not isinstance(data.get("extractions"), list):
            msg = f"'extractions' must be a list on line {line_num} of {jsonl_path}"
            raise TypeError(msg)

        if not isinstance(data.get("text"), str):
            msg = f"'text' must be a string on line {line_num} of {jsonl_path}"
            raise TypeError(msg)

        if not isinstance(data.get("document_id"), str):
            msg = f"'document_id' must be a string on line {line_num} of {jsonl_path}"
            raise TypeError(msg)

    logger.debug(
        "JSONL format validation passed for %s (%d lines)", jsonl_path, len(lines)
    )


def visualize_extractions(
    jsonl_path: str | Path,
    output_html_path: str | Path | None = None,
) -> str:
    """Visualize extractions from JSONL file using langextract.

    Validates JSONL format before visualization and handles various return types
    from langextract.visualize(). Includes comprehensive error handling for
    visualization failures.

    Args:
        jsonl_path: Path to input JSONL file
        output_html_path: Optional path to save HTML visualization
            (if None, returns HTML as string)

    Returns:
        HTML content as string

    Raises:
        ImportError: If langextract is not available
        FileNotFoundError: If jsonl_path doesn't exist
        ValueError: If JSONL format is invalid or visualization fails
        PermissionError: If output file cannot be written
    """
    jsonl_path = Path(jsonl_path)

    # Validate JSONL format before visualization
    try:
        _validate_jsonl_format(jsonl_path)
    except (FileNotFoundError, ValueError, TypeError):
        logger.exception("JSONL validation failed")
        raise

    try:
        import langextract as lx  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "langextract is required for visualization. "
            "Install with: uv add langextract (or pip install langextract)"
        )
        raise ImportError(msg) from e

    try:
        # Use langextract's visualize function
        logger.info("Generating visualization from %s", jsonl_path)
        html_content = lx.visualize(str(jsonl_path))

        # Handle different return types from langextract
        # langextract may return:
        # 1. A string directly
        # 2. A Jupyter display object with .data attribute
        # 3. An object with HTML content in various attributes
        html_str: str
        if isinstance(html_content, str):
            html_str = html_content
        elif hasattr(html_content, "data"):
            html_str = str(html_content.data)
        elif hasattr(html_content, "html"):
            html_str = str(html_content.html)
        else:
            # Fallback: convert to string
            html_str = str(html_content)

        def _validate_html_content(content: str) -> None:
            """Validate HTML content is not empty."""
            if not content or not content.strip():
                msg = "langextract.visualize() returned empty HTML content"
                raise ValueError(msg)  # noqa: TRY301

        _validate_html_content(html_str)

        logger.info(
            "Visualization generated successfully (%d characters)", len(html_str)
        )

    except Exception as e:
        logger.exception("Visualization failed for %s", jsonl_path)
        msg = f"Visualization failed: {e}"
        raise ValueError(msg) from e

    # Write to file if output path provided
    if output_html_path:
        try:
            output_path = Path(output_html_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(html_str)
            logger.info("Wrote visualization to %s", output_path)
        except PermissionError:
            logger.exception("Permission denied writing to %s", output_path)
            raise
        except OSError as e:
            logger.exception("Failed to write visualization to %s", output_path)
            msg = f"Failed to write visualization file: {e}"
            raise ValueError(msg) from e

    return html_str
