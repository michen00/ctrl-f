"""Gradio UI components for schema-grounded corpus extractor."""

from __future__ import annotations

__all__ = (
    "ExtractionWorkflowResult",
    "create_review_interface",
    "create_upload_interface",
    "show_source_context",
)

import json
import shutil
import tarfile
import tempfile
import zipfile
from inspect import cleandoc
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import gradio as gr
from slugify import slugify

from ctrlf.app.aggregate import has_disagreement
from ctrlf.app.errors import ErrorSummary, collect_errors
from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import process_corpus
from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import (
    Candidate,
    ExtractionResult,
    PersistedRecord,
    PrePromptInstrumentation,
    Resolution,
    SourceRef,
)
from ctrlf.app.schema_io import (
    convert_json_schema_to_pydantic,
    extend_schema,
    import_pydantic_model,
    validate_json_schema,
)
from ctrlf.app.storage import save_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel


logger = get_logger(__name__)


class NoOpProgress:
    """No-op progress tracker when Gradio progress is not available."""

    def __call__(self, *args: object, **kwargs: object) -> None:
        """No-op progress update."""

    @property
    def cancelled(self) -> bool:
        """Always return False for cancellation check."""
        return False


class ExtractionWorkflowResult(NamedTuple):
    """Result of extraction workflow execution.

    Attributes:
        progress_message: Progress status message
        error_message: Error message (empty if no errors)
        extraction_result: Extraction result or None if failed
        error_visibility: Gradio update for error visibility
    """

    progress_message: str
    error_message: str
    extraction_result: ExtractionResult | None
    error_visibility: Any


def _safe_snippet(snippet: str) -> str:
    """Escape triple backticks in snippet to prevent markdown fence breakage.

    Returns:
        Snippet with triple backticks escaped
    """
    return snippet.replace("```", "``` ")


class _CandidateSelection(NamedTuple):
    """Result of parsing a candidate selection string.

    Attributes:
        value: The chosen candidate value
        source_doc_id: Document ID of the first source
        source_location: Location of the first source
        provenance: List of source references
    """

    value: object
    source_doc_id: str | None
    source_location: str | None
    provenance: list[SourceRef]


def _parse_candidate_selection(
    candidate_value: str, candidates: list[Candidate]
) -> _CandidateSelection | None:
    """Parse a candidate selection string and extract candidate data.

    The candidate_value format is expected to be "idx:description" where idx is
    the index into the candidates list.

    Args:
        candidate_value: Selection string in format "idx:description"
        candidates: List of available candidates

    Returns:
        _CandidateSelection with extracted data, or None if parsing fails
    """
    try:
        idx_str = candidate_value.split(":")[0]
        idx = int(idx_str)
        if 0 <= idx < len(candidates):
            candidate = candidates[idx]
            source_doc_id = None
            source_location = None
            if candidate.sources:
                source_doc_id = candidate.sources[0].doc_id
                source_location = candidate.sources[0].location
            return _CandidateSelection(
                value=candidate.value,
                source_doc_id=source_doc_id,
                source_location=source_location,
                provenance=candidate.sources,
            )
    except (ValueError, IndexError):
        pass
    return None


def show_source_context(sources: list[SourceRef], *, side_by_side: bool = False) -> str:
    """Generate formatted source context display.

    Args:
        sources: Source references to display
        side_by_side: If True, format for side-by-side comparison

    Returns:
        Formatted Markdown string with snippets and metadata
    """
    if not sources:
        return "No sources available."

    if side_by_side and len(sources) >= 2:
        # Format for side-by-side comparison (using HTML table in Markdown)
        rows = []
        for i, source in enumerate(sources, 1):
            rows.append(
                f"| **Source {i}** | `{source.path}` | {source.location} |\n"
                f"| Context | `{_safe_snippet(source.snippet)}` | |"
            )
        header = "| Source | Document | Location |\n|--------|----------|----------|"
        return f"{header}\n" + "\n".join(rows)

    # Standard vertical format
    return "\n\n".join(
        cleandoc(f"""
                ### Source {i}
                **Document**: `{source.path}`
                **Location**: {source.location}

                **Context**:
                ```
                {_safe_snippet(source.snippet)}
                ```""")
        for i, source in enumerate(sources, 1)
    )


def _infer_schema_type(schema_file_path: str) -> str:
    """Infer schema type from file extension.

    Returns:
        "JSON Schema" for .json files, "Pydantic Model" for .py files

    Raises:
        ValueError: If file extension is not supported
    """
    schema_path = Path(schema_file_path)
    suffix = schema_path.suffix.lower()
    if suffix == ".json":
        return "JSON Schema"
    if suffix == ".py":
        return "Pydantic Model"
    msg = f"Unsupported schema file extension: {suffix}. Expected .json or .py"
    raise ValueError(msg)


def _load_schema(
    schema_file_path: str,
) -> type[BaseModel]:
    """Load schema from file and return Pydantic model class.

    Schema type is automatically inferred from the file extension:
    - .json files are treated as JSON Schema
    - .py files are treated as Pydantic models

    Returns:
        Pydantic model class

    Raises:
        ValueError: If schema loading fails or file extension is unsupported
    """
    schema_path = Path(schema_file_path)
    schema_type = _infer_schema_type(schema_file_path)

    if schema_type == "JSON Schema":
        # Load JSON Schema
        with schema_path.open() as f:
            schema_json = f.read()
        try:
            schema_dict = validate_json_schema(schema_json)
            return convert_json_schema_to_pydantic(schema_dict)
        except Exception as e:
            msg = f"Failed to load JSON Schema: {e}"
            raise ValueError(msg) from e
    else:
        # Load Pydantic model
        with schema_path.open() as f:
            code = f.read()
        try:
            return import_pydantic_model(code)
        except Exception as e:
            msg = f"Failed to load Pydantic model: {e}"
            raise ValueError(msg) from e


def _extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract archive file to destination directory.

    Raises:
        ValueError: If archive format is unsupported or archive contains unsafe paths
    """

    def _is_within_directory(base: Path, target: Path) -> bool:
        try:
            base = base.resolve(strict=False)
            target = target.resolve(strict=False)
            return str(target).startswith(str(base))
        except (OSError, RuntimeError):
            return False

    archive_lower = archive_path.lower()
    extract_to_path = Path(extract_to).resolve(strict=False)
    if archive_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                member_path = Path(member.filename)
                dest_path = (extract_to_path / member_path).resolve(strict=False)
                if not _is_within_directory(extract_to_path, dest_path):
                    msg = f"Unsafe path detected in zip archive: {member.filename}"
                    raise ValueError(msg)
                if member.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        zip_ref.open(member, "r") as source,
                        dest_path.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
    elif archive_lower.endswith((".tar", ".tar.gz")):
        mode = "r:gz" if archive_lower.endswith(".tar.gz") else "r"
        with tarfile.open(archive_path, mode) as tar_ref:  # type: ignore[call-overload]
            for member in tar_ref.getmembers():
                member_path = Path(member.name)
                dest_path = (extract_to_path / member_path).resolve(strict=False)
                if not _is_within_directory(extract_to_path, dest_path):
                    msg = f"Unsafe path detected in tar archive: {member.name}"
                    raise ValueError(msg)
                if member.isdir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    source = tar_ref.extractfile(member)
                    if source is not None:
                        with source, dest_path.open("wb") as target:
                            shutil.copyfileobj(source, target)
    else:
        msg = f"Unsupported archive format: {archive_path}"
        raise ValueError(msg)


def _check_cancellation(
    progress: gr.Progress | NoOpProgress,
    progress_messages: list[str],
) -> tuple[ExtractionWorkflowResult, PrePromptInstrumentation] | None:
    """Check if operation was cancelled and return cancellation result if so.

    Returns:
        Tuple of (ExtractionWorkflowResult, PrePromptInstrumentation) if cancelled,
        None otherwise
    """
    if progress.cancelled:  # type: ignore[union-attr]
        progress_messages.append("Operation cancelled by user.")
        return (
            ExtractionWorkflowResult(
                progress_message="\n".join(progress_messages),
                error_message="Operation cancelled by user.",
                extraction_result=None,
                error_visibility=gr.update(visible=True),
            ),
            PrePromptInstrumentation(interactions=[]),
        )
    return None


def _create_corpus_progress_callback(
    progress: gr.Progress | NoOpProgress,
    progress_messages: list[str],
    start_pct: float = 0.2,
    end_pct: float = 0.6,
) -> Callable[[int, int], None]:
    """Create progress callback for corpus processing.

    Returns:
        Progress callback function
    """

    def corpus_progress_callback(count: int, total: int) -> None:
        """Update progress during corpus processing."""
        # Handle empty corpus (total == 0) to avoid ZeroDivisionError
        if total == 0:
            # No documents to process - set progress to end_pct
            progress_pct = end_pct
            progress(
                progress_pct,
                desc="Processing corpus: 0/0 documents (empty corpus)",
            )
            progress_messages.append("Processed 0/0 documents (empty corpus)")
        else:
            # Map to start_pct-end_pct range
            progress_pct = start_pct + (count / total) * (end_pct - start_pct)
            progress(
                progress_pct,
                desc=f"Processing corpus: {count}/{total} documents",
            )
            progress_messages.append(f"Processed {count}/{total} documents")

        # Check for cancellation
        if progress.cancelled:  # type: ignore[union-attr]
            msg = "Operation cancelled by user"
            raise KeyboardInterrupt(msg)

    return corpus_progress_callback


def _get_directory_path(file_input: str | list[str] | None) -> str | None:
    """Extract directory path from file input.

    Handles both string paths and lists of file paths (from directory selection).
    When a list is provided, finds the common parent directory of all files.
    Works equally well for selected files or directories.

    Returns:
        Directory path as string, or None if input is None/empty
    """
    if file_input:
        if isinstance(file_input, str):
            path = Path(file_input)
            # If it's a file, return its parent directory
            # If it's a directory, return the directory itself
            return str(path.parent) if path.is_file() else str(path)
        # Handle list of file paths - find common parent directory
        if isinstance(file_input, list):
            if len(file_input) == 0:
                return None
            paths = [Path(f) for f in file_input]
            # Find common parent by comparing path parts
            common_parts: tuple[str, ...] | None = None
            for path in paths:
                parts = path.parent.parts if path.is_file() else path.parts
                if common_parts is None:
                    common_parts = parts
                else:
                    # Find the common prefix
                    common_parts = tuple(
                        p1
                        for p1, p2 in zip(common_parts, parts, strict=False)
                        if p1 == p2
                    )
            if common_parts:
                return str(Path(*common_parts))
            # Fallback: use parent of first file or the directory itself
            return str(paths[0].parent if paths[0].is_file() else paths[0])
    return None


def _resolve_schema_file_path(
    schema_file_path: str | None,
    schema_dir_path: str | None,
) -> str:
    """Resolve schema file path from file upload or text input.

    Args:
        schema_file_path: Path from file upload component
        schema_dir_path: Path from text input component

    Returns:
        Resolved schema file path

    Raises:
        ValueError: If no schema input is provided or path is invalid
    """
    if schema_file_path:
        return schema_file_path
    if schema_dir_path:
        resolved_path = _resolve_and_validate_path(schema_dir_path)
        # Ensure it's a file, not a directory
        path = Path(resolved_path)
        if not path.is_file():
            msg = f"Schema path must be a file, not a directory: {resolved_path}"
            raise ValueError(msg)
        return resolved_path

    msg = "Schema file or path is required"
    raise ValueError(msg)


def _load_and_extend_schema(
    schema_file_path: str,
    actual_progress: gr.Progress | NoOpProgress,
    progress_messages: list[str],
) -> type[BaseModel]:
    """Load and extend schema from file.

    Returns:
        Extended Pydantic model class

    Raises:
        ValueError: If schema file is invalid
        KeyboardInterrupt: If operation is cancelled
    """
    actual_progress(0, desc="Loading schema...")
    progress_messages.append("Loading schema...")

    # Check for cancellation
    cancel_result = _check_cancellation(actual_progress, progress_messages)
    if cancel_result:
        msg = "Operation cancelled by user"
        raise KeyboardInterrupt(msg)

    # Load schema (type is inferred from file extension)
    raw_model_class = _load_schema(schema_file_path)

    # Extend schema to support multiple values per field
    model_class = extend_schema(raw_model_class)

    actual_progress(0.2, desc="Schema loaded successfully.")
    progress_messages.append("Schema loaded successfully.")

    # Check for cancellation
    cancel_result = _check_cancellation(actual_progress, progress_messages)
    if cancel_result:
        msg = "Operation cancelled by user"
        raise KeyboardInterrupt(msg)

    return model_class


def _process_corpus_input(  # noqa: PLR0915
    corpus_file_path: str | None,
    corpus_dir_path: str | None,
    actual_progress: gr.Progress | NoOpProgress,
    progress_messages: list[str],
) -> list[Any]:
    """Process corpus from file upload and/or directory path.

    Both inputs can be provided simultaneously, and documents from both
    sources will be combined into a single corpus.

    Returns:
        List of corpus documents

    Raises:
        ValueError: If no corpus input is provided
        KeyboardInterrupt: If operation is cancelled
    """
    actual_progress(0.2, desc="Processing corpus...")
    progress_messages.append("Processing corpus...")

    corpus_docs: list[Any] = []
    docs_from_file: list[Any] = []
    docs_from_dir: list[Any] = []

    # Process file input if provided
    if corpus_file_path:
        # Check if it's an archive or a single document file
        corpus_path = Path(corpus_file_path)
        is_archive = corpus_path.suffix.lower() in (
            ".zip",
            ".tar",
        ) or corpus_path.name.lower().endswith(".tar.gz")

        if is_archive:
            # Extract archive to temp directory
            tmpdir = tempfile.mkdtemp()
            try:
                actual_progress(0.25, desc="Extracting archive...")
                progress_messages.append("Extracting archive...")
                _extract_archive(corpus_file_path, tmpdir)

                # Check for cancellation after extraction
                cancel_result = _check_cancellation(actual_progress, progress_messages)
                if cancel_result:
                    msg = "Operation cancelled by user"
                    raise KeyboardInterrupt(msg)

                # Calculate progress range for file processing
                # If both inputs provided, file gets 0.25-0.4, dir gets 0.4-0.6
                # If only file, it gets 0.25-0.6
                start_pct = 0.25
                end_pct = 0.4 if corpus_dir_path else 0.6
                corpus_progress_callback = _create_corpus_progress_callback(
                    actual_progress,
                    progress_messages,
                    start_pct=start_pct,
                    end_pct=end_pct,
                )

                docs_from_file = process_corpus(
                    tmpdir,
                    progress_callback=corpus_progress_callback,
                )
                progress_messages.append(
                    f"Processed {len(docs_from_file)} document(s) from file upload."
                )
            finally:
                # Clean up temp directory after processing
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            # Single document file - convert with markitdown
            actual_progress(0.25, desc="Converting document to markdown...")
            progress_messages.append("Converting document to markdown...")

            # Calculate progress range for file processing
            start_pct = 0.25
            end_pct = 0.4 if corpus_dir_path else 0.6
            corpus_progress_callback = _create_corpus_progress_callback(
                actual_progress,
                progress_messages,
                start_pct=start_pct,
                end_pct=end_pct,
            )

            docs_from_file = process_corpus(
                corpus_file_path,
                progress_callback=corpus_progress_callback,
            )
            progress_messages.append(
                f"Processed {len(docs_from_file)} document(s) from file upload."
            )

        # Check for cancellation after file processing
        cancel_result = _check_cancellation(actual_progress, progress_messages)
        if cancel_result:
            msg = "Operation cancelled by user"
            raise KeyboardInterrupt(msg)

    # Process directory input if provided
    if corpus_dir_path:
        # Resolve and validate text input path
        resolved_path = _resolve_and_validate_path(corpus_dir_path)

        # Calculate progress range for directory processing
        # If both inputs provided, dir gets 0.4-0.6
        # If only dir, it gets 0.25-0.6
        start_pct = 0.4 if corpus_file_path else 0.25
        end_pct = 0.6
        corpus_progress_callback = _create_corpus_progress_callback(
            actual_progress,
            progress_messages,
            start_pct=start_pct,
            end_pct=end_pct,
        )

        docs_from_dir = process_corpus(
            resolved_path,
            progress_callback=corpus_progress_callback,
        )
        progress_messages.append(
            f"Processed {len(docs_from_dir)} document(s) from directory path."
        )

        # Check for cancellation after directory processing
        cancel_result = _check_cancellation(actual_progress, progress_messages)
        if cancel_result:
            msg = "Operation cancelled by user"
            raise KeyboardInterrupt(msg)

    # Combine documents from both sources
    if not corpus_file_path and not corpus_dir_path:
        msg = "Corpus file or directory is required"
        raise ValueError(msg)

    corpus_docs = docs_from_file + docs_from_dir

    actual_progress(0.6, desc=f"Processed {len(corpus_docs)} total documents.")
    progress_messages.append(f"Processed {len(corpus_docs)} total documents.")

    # Check for cancellation
    cancel_result = _check_cancellation(actual_progress, progress_messages)
    if cancel_result:
        msg = "Operation cancelled by user"
        raise KeyboardInterrupt(msg)

    return corpus_docs


def _run_extraction_step(
    model_class: type[BaseModel],
    corpus_docs: list[Any],
    actual_progress: gr.Progress | NoOpProgress,
    progress_messages: list[str],
) -> tuple[ExtractionResult, PrePromptInstrumentation]:
    """Run the extraction step.

    Args:
        model_class: Extended Pydantic model class
        corpus_docs: List of corpus documents
        actual_progress: Progress tracker
        progress_messages: List to append progress messages to

    Returns:
        Tuple of (extraction_result, instrumentation)

    Raises:
        KeyboardInterrupt: If operation is cancelled
    """
    actual_progress(0.6, desc="Running extraction...")
    progress_messages.append("Running extraction...")

    # Check for cancellation before starting extraction
    cancel_result = _check_cancellation(actual_progress, progress_messages)
    if cancel_result:
        msg = "Operation cancelled by user"
        raise KeyboardInterrupt(msg)

    # Run extraction
    extraction_result, instrumentation = run_extraction(model_class, corpus_docs)

    actual_progress(0.95, desc="Extraction complete. Finalizing...")
    progress_messages.append("Extraction complete.")

    actual_progress(1.0, desc="Complete!")
    return extraction_result, instrumentation


def _resolve_and_validate_path(path_input: str | None) -> str:
    """Resolve and validate a text input path.

    Handles path resolution including:
    - Tilde expansion (~)
    - Spaces and special characters
    - File vs directory detection

    Returns:
        Resolved absolute path as string

    Raises:
        ValueError: If path is empty, doesn't exist, or is invalid
    """
    if not path_input or not path_input.strip():
        msg = "Path cannot be empty"
        raise ValueError(msg)

    # Strip whitespace
    path_str = path_input.strip()

    # Expand user home directory (~)
    path_str = str(Path(path_str).expanduser())

    # Resolve to absolute path (handles relative paths, .., etc.)
    # This also handles spaces and special characters naturally
    try:
        resolved_path = Path(path_str).resolve()
    except (OSError, RuntimeError) as e:
        msg = f"Invalid path: {path_str} - {e}"
        raise ValueError(msg) from e

    # Check if path exists
    if not resolved_path.exists():
        msg = f"Path does not exist: {resolved_path}"
        raise ValueError(msg)

    return str(resolved_path)


def create_upload_interface() -> gr.Blocks:  # noqa: C901, PLR0915
    """Create Gradio interface for schema and corpus upload.

    Returns:
        Gradio interface component
    """
    with gr.Blocks(title="Schema-Grounded Corpus Extractor") as interface:
        gr.Markdown("# Schema-Grounded Corpus Extractor")
        gr.Markdown(
            "Upload a schema (JSON Schema or Pydantic model) and a corpus of "
            "documents to extract structured data."
        )

        with gr.Row():
            with gr.Column():
                schema_file = gr.File(
                    label="Schema File",
                    file_types=[".json", ".py"],
                    type="filepath",
                )
                schema_dir = gr.Textbox(
                    label="Or Schema File Path",
                    placeholder=(
                        "Enter path to schema file "
                        "(e.g., ~/Documents/schema.json or ./schemas/model.py)"
                    ),
                    type="text",
                )

            with gr.Column():
                corpus_file = gr.File(
                    label="Corpus (Document or Archive)",
                    type="filepath",
                )
                corpus_dir = gr.Textbox(
                    label="Or Corpus Directory/File Path",
                    placeholder=(
                        "Enter path to directory or file "
                        "(e.g., ~/Documents/corpus or ./data/file.pdf)"
                    ),
                    type="text",
                )

        with gr.Row():
            null_policy = gr.Radio(
                choices=["Empty List", "Explicit Null"],
                value="Empty List",
                label="Null Policy",
            )
            confidence_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.75,
                step=0.05,
                label="Confidence Threshold",
            )

        run_button = gr.Button("Run Extraction", variant="primary")
        progress_output = gr.Textbox(
            label="Progress",
            lines=5,
            interactive=False,
        )
        error_output = gr.Textbox(
            label="Errors",
            lines=5,
            interactive=False,
            visible=False,
        )
        extraction_result_state = gr.State()
        instrumentation_state = gr.State()

        # Pre-prompt instrumentation display
        with gr.Accordion("Pre-Prompt Instrumentation", open=False):
            instrumentation_display = gr.Markdown(
                value="No instrumentation data available yet.",
                label="Pre-Prompt Interactions",
            )

        def run_extraction_workflow(
            schema_file_path: str | None,
            schema_dir_path: str | None,
            corpus_file_path: str | None,
            corpus_dir_path: str | None,
            _null_policy_str: str,
            _confidence: float,
            progress: gr.Progress | None = None,
        ) -> tuple[ExtractionWorkflowResult, PrePromptInstrumentation]:
            """Run the extraction workflow.

            Args:
                schema_file_path: Path to schema file from file upload
                schema_dir_path: Text input path to schema file
                corpus_file_path: Path to corpus archive
                corpus_dir_path: Text input path to corpus directory or file
                _null_policy_str: Null policy setting (unused in v0)
                _confidence: Confidence threshold (unused in v0)
                progress: Gradio progress tracker for progress bars and cancellation

            Returns:
                ExtractionWorkflowResult with progress, error, result, and visibility
            """
            error_summary = ErrorSummary()
            progress_messages: list[str] = []

            # Use no-op if progress is None
            actual_progress: gr.Progress | NoOpProgress = (
                NoOpProgress() if progress is None else progress
            )

            try:
                # Step 1: Load schema (0-20%)
                resolved_schema_path = _resolve_schema_file_path(
                    schema_file_path, schema_dir_path
                )
                model_class = _load_and_extend_schema(
                    resolved_schema_path, actual_progress, progress_messages
                )

                # Step 2: Process corpus (20-60%)
                corpus_docs = _process_corpus_input(
                    corpus_file_path,
                    corpus_dir_path,
                    actual_progress,
                    progress_messages,
                )

                # Step 3: Run extraction (60-100%)
                extraction_result, instrumentation = _run_extraction_step(
                    model_class, corpus_docs, actual_progress, progress_messages
                )

                progress_msg = "\n".join(progress_messages)
                workflow_result = ExtractionWorkflowResult(
                    progress_message=progress_msg,
                    error_message="",
                    extraction_result=extraction_result,
                    error_visibility=gr.update(visible=False),
                )
                return workflow_result, instrumentation  # noqa: TRY300

            except KeyboardInterrupt:
                # Handle cancellation
                progress_messages.append("Operation cancelled by user.")
                workflow_result = ExtractionWorkflowResult(
                    progress_message="\n".join(progress_messages),
                    error_message="Operation cancelled by user.",
                    extraction_result=None,
                    error_visibility=gr.update(visible=True),
                )
                return workflow_result, PrePromptInstrumentation(interactions=[])
            except Exception as e:
                collect_errors(
                    error_summary,
                    "ExtractionError",
                    "extraction_workflow",
                    e,
                )
                error_msg = error_summary.get_summary()
                progress_msg = "\n".join(progress_messages)
                logger.exception("Extraction workflow failed")
                workflow_result = ExtractionWorkflowResult(
                    progress_message=progress_msg,
                    error_message=error_msg,
                    extraction_result=None,
                    error_visibility=gr.update(visible=True),
                )
                return workflow_result, PrePromptInstrumentation(interactions=[])

        def format_instrumentation(
            instrumentation: PrePromptInstrumentation,
        ) -> str:
            """Format instrumentation data for display.

            Returns:
                Formatted markdown string
            """
            if not instrumentation.interactions:
                return "No pre-prompt interactions recorded."

            lines = ["## Pre-Prompt Interactions\n"]
            append_line = lines.append
            for i, interaction in enumerate(instrumentation.interactions, 1):
                # Truncate very long completions for display
                completion = interaction.completion
                completion_preview = completion
                if len(completion) > 2000:
                    completion_preview = completion[:2000] + "\n... (truncated)"

                # Build metadata section
                metadata_lines = []
                metadata_lines.append(f"**Model**: `{interaction.model}`")

                has_tokens = (
                    interaction.prompt_tokens is not None
                    or interaction.completion_tokens is not None
                )
                if has_tokens:
                    token_info = []
                    if interaction.prompt_tokens is not None:
                        token_info.append(
                            f"Prompt: {interaction.prompt_tokens:,} tokens"
                        )
                    if interaction.completion_tokens is not None:
                        token_info.append(
                            f"Completion: {interaction.completion_tokens:,} tokens"
                        )
                    if token_info:
                        metadata_lines.append(
                            f"**Token Usage**: {', '.join(token_info)}"
                        )

                if interaction.finish_reason:
                    metadata_lines.append(
                        f"**Finish Reason**: `{interaction.finish_reason}`"
                    )

                if interaction.response_metadata:
                    metadata_items = []
                    if "total_tokens" in interaction.response_metadata:
                        total = interaction.response_metadata["total_tokens"]
                        metadata_items.append(f"Total: {total:,} tokens")
                    if "safety_ratings" in interaction.response_metadata:
                        safety = interaction.response_metadata["safety_ratings"]
                        if isinstance(safety, list) and safety:
                            metadata_items.append(
                                f"Safety ratings: {len(safety)} categories checked"
                            )
                    if metadata_items:
                        metadata_lines.append(
                            f"**Additional Info**: {', '.join(metadata_items)}"
                        )

                append_line(
                    cleandoc(f"""
                        ### {i}. {interaction.step_name}

                        {chr(10).join(metadata_lines)}

                        #### Prompt:
                        ```
                        {interaction.prompt}
                        ```

                        #### Model Response:
                        ```
                        {completion_preview}
                        ```

                        **Response Length**: {len(completion):,} characters

                        ---
                        """)
                )

            return "\n".join(lines)

        def unpack_result(
            result: ExtractionWorkflowResult,
            instrumentation: PrePromptInstrumentation,
        ) -> tuple[str, Any, ExtractionResult | None, str, PrePromptInstrumentation]:
            """Unpack ExtractionWorkflowResult for Gradio outputs.

            Returns:
                Tuple of (progress_message, error_update, extraction_result,
                formatted instrumentation markdown, instrumentation)
            """
            # Combine error message and visibility into a single gr.update() object
            # Show error output if there's an error message, hide otherwise
            error_update = gr.update(
                value=result.error_message,
                visible=bool(result.error_message),
            )
            instrumentation_md = format_instrumentation(instrumentation)
            return (
                result.progress_message,
                error_update,
                result.extraction_result,
                instrumentation_md,
                instrumentation,
            )

        def run_extraction_with_progress(
            schema_file_path: str | None,
            schema_dir_path: str | None,
            corpus_file_path: str | None,
            corpus_dir_path: str | None,
            _null_policy_str: str,
            _confidence: float,
            progress: gr.Progress | None = None,
        ) -> tuple[str, Any, ExtractionResult | None, str, PrePromptInstrumentation]:
            """Wrapper to run extraction workflow and unpack result.

            Args:
                schema_file_path: Path to schema file from file upload
                schema_dir_path: Text input path to schema file
                corpus_file_path: Path to corpus archive
                corpus_dir_path: Text input path to corpus directory or file
                _null_policy_str: Null policy setting (unused in v0)
                _confidence: Confidence threshold (unused in v0)
                progress: Gradio progress tracker

            Returns:
                Tuple of (progress_message, error_update, extraction_result,
                formatted instrumentation markdown, instrumentation)
            """
            result, instrumentation = run_extraction_workflow(
                schema_file_path,
                schema_dir_path,
                corpus_file_path,
                corpus_dir_path,
                _null_policy_str,
                _confidence,
                progress,
            )
            return unpack_result(result, instrumentation)

        run_button.click(
            fn=run_extraction_with_progress,
            inputs=[
                schema_file,
                schema_dir,
                corpus_file,
                corpus_dir,
                null_policy,
                confidence_threshold,
            ],
            outputs=[
                progress_output,
                error_output,
                extraction_result_state,
                instrumentation_display,
                instrumentation_state,
            ],
        )

    return interface  # type: ignore[no-any-return]


def create_review_interface(  # noqa: PLR0915, C901
    extraction_result: ExtractionResult,
) -> gr.Blocks:
    """Create Gradio interface for reviewing and resolving candidates.

    Returns:
        Gradio interface component with field accordions
    """
    with gr.Blocks(title="Review Extraction Results") as interface:
        gr.Markdown("# Review Extraction Results")
        gr.Markdown(
            f"**Run ID**: `{extraction_result.run_id}`  \n"
            f"**Schema Version**: `{extraction_result.schema_version}`  \n"
            f"**Created**: {extraction_result.created_at}"
        )

        # Filter/search functionality (User Story 3 + Polish)
        with gr.Row():
            filter_text = gr.Textbox(
                label="Filter Fields",
                placeholder="Search by field name...",
                value="",
            )
            filter_type = gr.Radio(
                choices=["All", "Unresolved", "Flagged (Disagreements)"],
                value="All",
                label="Filter Type",
            )
        filter_status = gr.Markdown(visible=False)

        field_components: dict[str, Any] = {}
        field_accordions: dict[str, Any] = {}

        # Check for disagreements
        field_disagreements = {
            fr.field_name: has_disagreement(fr.candidates)
            for fr in extraction_result.results
        }

        for field_result in extraction_result.results:
            has_disag = field_disagreements.get(field_result.field_name, False)
            # Visual flagging for disagreements (User Story 3)
            accordion_label = f"Field: {field_result.field_name}"
            if has_disag:
                accordion_label = f"🔴 {accordion_label} (Disagreement)"

            with gr.Accordion(
                label=accordion_label,
                open=not field_result.consensus or has_disag,
            ) as accordion:
                field_accordions[field_result.field_name] = accordion

                # Show consensus status with enhanced confidence display
                if field_result.consensus:
                    gr.Markdown(
                        f"✅ **Consensus detected**: "
                        f"`{field_result.consensus.value}` "
                        f"**(confidence: {field_result.consensus.confidence:.2%})**"
                    )
                elif has_disag:
                    gr.Markdown(
                        "🔴 **⚠️ DISAGREEMENT DETECTED** - Multiple candidates "
                        "with similar confidence. Manual selection required."
                    )
                else:
                    gr.Markdown("⚠️ **No consensus** - manual selection required")

                # Show candidates with enhanced confidence display
                if field_result.candidates:
                    gr.Markdown("### Candidates")
                    # Store candidate choices with enhanced confidence display
                    candidate_choices = [
                        f"{i}: {c.value} (confidence: {c.confidence:.2%})"
                        for i, c in enumerate(field_result.candidates)
                    ]
                    # No pre-selection for fields with disagreements (User Story 3)
                    default_value = None
                    if field_result.consensus and not has_disag:
                        default_value = (
                            f"{field_result.candidates.index(field_result.consensus)}: "
                            f"{field_result.consensus.value} "
                            f"(confidence: {field_result.consensus.confidence:.2%})"
                        )

                    candidate_radio = gr.Radio(
                        choices=candidate_choices,
                        label="Select Candidate",
                        value=default_value,
                    )
                    field_components[f"{field_result.field_name}_candidate"] = (
                        candidate_radio
                    )

                    # Show source context output
                    source_context_output = gr.Markdown(
                        label="Source Context", visible=False
                    )

                    # Create individual "View source" buttons for each candidate
                    # (User Story 3)
                    with gr.Row():
                        for i, candidate in enumerate(field_result.candidates):
                            with gr.Column(scale=1):
                                view_source_btn = gr.Button(
                                    f"View Source {i + 1}",
                                    size="sm",
                                )

                                def make_show_source_fn_for_candidate(
                                    candidate_idx: int,
                                    candidates: list[Candidate],
                                ) -> Callable[[], tuple[str, Any]]:
                                    """Create function to show source for candidate.

                                    Returns:
                                        Function to show source context
                                    """

                                    def show_source() -> tuple[str, Any]:
                                        """Show source context for candidate.

                                        Returns:
                                            Tuple of (source_context_markdown,
                                            visibility_update)
                                        """
                                        if 0 <= candidate_idx < len(candidates):
                                            candidate = candidates[candidate_idx]
                                            # Use side-by-side if multiple sources
                                            side_by_side = len(candidate.sources) >= 2
                                            context = show_source_context(
                                                candidate.sources,
                                                side_by_side=side_by_side,
                                            )
                                            return context, gr.update(visible=True)
                                        return "", gr.update(visible=False)

                                    return show_source

                                view_source_btn.click(
                                    fn=make_show_source_fn_for_candidate(
                                        i, field_result.candidates
                                    ),
                                    inputs=[],
                                    outputs=[
                                        source_context_output,
                                        source_context_output,
                                    ],
                                )

                    # Also keep original "View Source Context" button
                    show_source_btn = gr.Button("View Source for Selected Candidate")

                    def make_show_source_for_selected_fn(
                        candidates: list[Candidate],
                    ) -> Callable[[str], tuple[str, Any]]:
                        """Create a function to show source for selected candidate.

                        Returns:
                            Function to show source context
                        """

                        def show_source_for_selected(selected: str) -> tuple[str, Any]:
                            """Show source context for selected candidate.

                            Args:
                                selected: Radio button selection in format
                                    "index: value (confidence: XX%)"

                            Returns:
                                Tuple of (source_context_markdown, visibility_update)
                            """
                            if not selected:
                                return "", gr.update(visible=False)

                            # Extract index from format "index: value (confidence: XX%)"
                            # Use partition to safely split on first colon
                            idx_str, _colon, _rest = selected.partition(":")
                            if not idx_str or not _colon:
                                return "", gr.update(visible=False)

                            try:
                                idx = int(idx_str.strip())
                            except ValueError:
                                return "", gr.update(visible=False)

                            if 0 <= idx < len(candidates):
                                candidate = candidates[idx]
                                side_by_side = len(candidate.sources) >= 2
                                context = show_source_context(
                                    candidate.sources, side_by_side=side_by_side
                                )
                                return context, gr.update(visible=True)

                            return "", gr.update(visible=False)

                        return show_source_for_selected

                    show_source_btn.click(
                        fn=make_show_source_for_selected_fn(field_result.candidates),
                        inputs=[candidate_radio],
                        outputs=[source_context_output, source_context_output],
                    )
                else:
                    gr.Markdown("*No candidates found*")

                # Custom value input
                gr.Markdown("### Or Enter Custom Value")
                custom_value = gr.Textbox(
                    label="Custom Value",
                    placeholder="Enter value manually...",
                )
                field_components[f"{field_result.field_name}_custom"] = custom_value

        # Filter functionality (User Story 3 + Polish T070)
        def filter_fields(filter_text: str, filter_type: str) -> list[Any]:
            """Filter fields based on search text and filter type.

            Args:
                filter_text: Search text
                filter_type: Filter type (All, Unresolved, Flagged)

            Returns:
                List of visibility updates for accordions
            """
            filter_lower = filter_text.lower() if filter_text else ""
            updates = []
            visible_count = 0

            for field_result in extraction_result.results:
                field_name = field_result.field_name
                has_disag = field_disagreements.get(field_name, False)

                # Apply filter type
                if filter_type == "Unresolved":
                    # Show only fields without consensus
                    type_match = field_result.consensus is None
                elif filter_type == "Flagged (Disagreements)":
                    # Show only fields with disagreements
                    type_match = has_disag
                else:  # "All"
                    type_match = True

                # Apply text filter
                text_match = not filter_lower or filter_lower in field_name.lower()

                visible = type_match and text_match
                if visible:
                    visible_count += 1
                updates.append(gr.update(visible=visible))

            return [
                *updates,
                gr.update(
                    visible=True,
                    value=f"Showing {visible_count} of {len(field_accordions)} fields",
                ),
            ]

        def update_filter(filter_text: str, filter_type: str) -> list[Any]:
            """Update filter when either text or type changes."""
            return filter_fields(filter_text, filter_type)

        filter_text.change(
            fn=update_filter,
            inputs=[filter_text, filter_type],
            outputs=[*field_accordions.values(), filter_status],
        )
        filter_type.change(
            fn=update_filter,
            inputs=[filter_text, filter_type],
            outputs=[*field_accordions.values(), filter_status],
        )

        # Save and Export buttons (Polish T069)
        with gr.Row():
            save_button = gr.Button("Save Record", variant="primary")
            export_json_button = gr.Button("Export as JSON", variant="secondary")
        save_status = gr.Textbox(label="Save Status", interactive=False)
        export_json_output = gr.File(label="Exported JSON", visible=False)

        def _build_resolved_record(  # noqa: PLR0912, PLR0915
            extraction_result: ExtractionResult,
            *field_values: str,
        ) -> PersistedRecord:
            """Build a PersistedRecord from extraction result and field values.

            This helper function ensures consistent record ID generation for both
            save and export operations.

            Args:
                extraction_result: Original extraction result
                *field_values: Field values from the form (alternating candidate/custom)

            Returns:
                PersistedRecord with consistent ID generation
            """
            resolutions: list[Resolution] = []
            resolved: dict[str, object] = {}
            provenance: dict[str, list[SourceRef]] = {}

            # Parse field values
            # Note: Fields with candidates contribute 2 values
            # (candidate_radio, custom_textbox)
            # Fields without candidates contribute 1 value (custom_textbox only)
            field_idx = 0
            for field_result in extraction_result.results:
                # Check if this field has candidates to determine
                # how many values it contributes
                has_candidates = bool(field_result.candidates)

                if has_candidates:
                    # Field with candidates: read both candidate_radio
                    # and custom_textbox
                    candidate_value = (
                        field_values[field_idx]
                        if field_idx < len(field_values)
                        else None
                    )
                    custom_value = (
                        field_values[field_idx + 1]
                        if field_idx + 1 < len(field_values)
                        else None
                    )
                    field_idx += 2
                else:
                    # Field without candidates: only read custom_textbox
                    candidate_value = None
                    custom_value = (
                        field_values[field_idx]
                        if field_idx < len(field_values)
                        else None
                    )
                    field_idx += 1

                # Determine which value to use
                chosen_value: object | None = None
                source_doc_id: str | None = None
                source_location: str | None = None
                custom_input = False
                field_provenance: list[SourceRef] = []

                if custom_value and custom_value.strip():
                    # Use custom value - validate it
                    custom_value_stripped = custom_value.strip()

                    # Basic validation: check if we can infer type from candidates
                    if field_result.candidates:
                        # Try to validate against candidate value type
                        candidate_val: Any = field_result.candidates[0].value

                        # Try to convert custom value to same type as candidates
                        try:
                            if isinstance(candidate_val, int):
                                chosen_value = int(custom_value_stripped)
                            elif isinstance(candidate_val, float):
                                chosen_value = float(custom_value_stripped)
                            elif isinstance(candidate_val, bool):
                                # Handle boolean strings
                                if custom_value_stripped.lower() in (
                                    "true",
                                    "1",
                                    "yes",
                                ):
                                    chosen_value = True
                                elif custom_value_stripped.lower() in (
                                    "false",
                                    "0",
                                    "no",
                                ):
                                    chosen_value = False
                                else:
                                    msg = (
                                        f"Invalid boolean value for field "
                                        f"'{field_result.field_name}': "
                                        f"{custom_value_stripped}"
                                    )
                                    raise ValueError(msg)  # noqa: TRY301
                            else:
                                # String or other types - use as-is
                                chosen_value = custom_value_stripped
                        except (ValueError, TypeError) as e:
                            msg = (
                                f"Invalid value type for field "
                                f"'{field_result.field_name}': {e}"
                            )
                            raise ValueError(msg) from e
                    else:
                        # No candidates to infer type from - use as string
                        chosen_value = custom_value_stripped

                    custom_input = True
                elif candidate_value:
                    # Extract candidate index and find the candidate
                    selection = _parse_candidate_selection(
                        candidate_value, field_result.candidates
                    )
                    if selection:
                        chosen_value = selection.value
                        source_doc_id = selection.source_doc_id
                        source_location = selection.source_location
                        field_provenance = selection.provenance

                if chosen_value is not None:
                    # Create resolution
                    resolution = Resolution(
                        field_name=field_result.field_name,
                        chosen_value=chosen_value,
                        source_doc_id=source_doc_id,
                        source_location=source_location,
                        custom_input=custom_input,
                    )
                    resolutions.append(resolution)

                    # Add to resolved (as array per Extended Schema)
                    resolved[field_result.field_name] = [chosen_value]
                    provenance[field_result.field_name] = field_provenance

            # Generate consistent record ID using single format
            # Use timestamp from extraction_result to ensure save/export use same ID
            timestamp = extraction_result.created_at
            record_id = slugify(f"record-{extraction_result.run_id}-{timestamp}")

            # Create PersistedRecord
            record = PersistedRecord(
                record_id=record_id,
                resolved=resolved,
                provenance=provenance,
                audit={
                    "run_id": extraction_result.run_id,
                    "app_version": "0.0.0",
                    "timestamp": timestamp,
                    "user": None,
                    "config": {},
                    "schema_version": extraction_result.schema_version,
                },
            )

            return record  # noqa: RET504

        def save_resolved_record(
            extraction_result: ExtractionResult,
            *field_values: str,
        ) -> str:
            """Save the resolved record.

            Args:
                extraction_result: Original extraction result
                *field_values: Field values from the form (alternating candidate/custom)

            Returns:
                Success message
            """
            try:
                # Build record using shared helper for consistent ID generation
                record = _build_resolved_record(extraction_result, *field_values)

                # Save record
                save_record(record)
            except Exception as e:
                logger.exception("Failed to save record")
                return f"Error saving record: {e}"
            else:
                return f"Record saved successfully! Record ID: {record.record_id}"

        def export_resolved_record_json(
            extraction_result: ExtractionResult,
            *field_values: str,
        ) -> tuple[str, Any]:
            """Export the resolved record as JSON file.

            Args:
                extraction_result: Original extraction result
                *field_values: Field values from the form (alternating candidate/custom)

            Returns:
                Tuple of (status_message, file_path_or_none)
            """
            try:
                # Build record using shared helper for consistent ID generation
                record = _build_resolved_record(extraction_result, *field_values)

                # Export to JSON file
                export_data = record.model_dump(mode="json")
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(export_data, f, indent=2)
                    export_path = f.name

                return (
                    f"Record exported successfully! File: {Path(export_path).name}",
                    gr.update(value=export_path, visible=True),
                )
            except Exception as e:
                logger.exception("Failed to export record")
                return (
                    f"Error exporting record: {e}",
                    gr.update(visible=False),
                )

        # Store extraction result in a state component for the save function
        # Initialize with the extraction_result parameter value since State components
        # only fire .change() events when written to by callbacks, not on initialization
        extraction_result_state_review = gr.State(value=extraction_result)

        save_button.click(
            fn=save_resolved_record,
            inputs=[extraction_result_state_review, *list(field_components.values())],
            outputs=[save_status],
        )

        export_json_button.click(
            fn=export_resolved_record_json,
            inputs=[extraction_result_state_review, *list(field_components.values())],
            outputs=[save_status, export_json_output],
        )

    return interface  # type: ignore[no-any-return]
