"""Gradio UI components for schema-grounded corpus extractor."""

from __future__ import annotations

__all__ = (
    "ExtractionWorkflowResult",
    "create_review_interface",
    "create_upload_interface",
    "show_source_context",
)

import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from inspect import cleandoc
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import gradio as gr
from slugify import slugify

from ctrlf.app.errors import ErrorSummary, collect_errors
from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import process_corpus
from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import Candidate, PersistedRecord, Resolution, SourceRef
from ctrlf.app.schema_io import import_pydantic_model
from ctrlf.app.storage import save_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from ctrlf.app.models import ExtractionResult


logger = get_logger(__name__)


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

    Args:
        snippet: Source snippet text that may contain code fences

    Returns:
        Snippet with triple backticks escaped
    """
    return snippet.replace("```", "``` ")


def show_source_context(sources: list[SourceRef]) -> str:
    """Generate formatted source context display.

    Args:
        sources: Source references to display

    Returns:
        Formatted Markdown string with snippets and metadata
    """
    return (
        "\n\n".join(
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
        if sources
        else "No sources available."
    )


def create_upload_interface() -> gr.Blocks:  # noqa: PLR0915
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
                schema_type = gr.Radio(
                    choices=["JSON Schema", "Pydantic Model"],
                    value="JSON Schema",
                    label="Schema Type",
                )

            with gr.Column():
                corpus_file = gr.File(
                    label="Corpus (Directory or Archive)",
                    file_types=[".zip", ".tar", ".tar.gz"],
                    type="filepath",
                )
                corpus_dir = gr.Textbox(
                    label="Or Corpus Directory Path",
                    placeholder="/path/to/corpus",
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

        def run_extraction_workflow(  # noqa: PLR0915
            schema_file_path: str | None,
            schema_type_str: str,
            corpus_file_path: str | None,
            corpus_dir_path: str | None,
            _null_policy_str: str,
            _confidence: float,
            _progress: gr.Progress,
        ) -> ExtractionWorkflowResult:
            """Run the extraction workflow.

            Args:
                schema_file_path: Path to schema file
                schema_type_str: Type of schema (JSON Schema or Pydantic Model)
                corpus_file_path: Path to corpus archive
                corpus_dir_path: Path to corpus directory
                _null_policy_str: Null policy setting (unused in v0)
                _confidence: Confidence threshold (unused in v0)
                _progress: Gradio progress tracker (unused)

            Returns:
                ExtractionWorkflowResult with progress, error, result, and visibility
            """
            error_summary = ErrorSummary()
            progress_messages: list[str] = []

            try:
                # Step 1: Load schema
                progress_messages.append("Loading schema...")
                if not schema_file_path:
                    msg = "Schema file is required"
                    raise ValueError(msg)  # noqa: TRY301

                # For v0, only Pydantic models are supported
                # JSON Schema conversion will be implemented in User Story 2
                if schema_type_str == "JSON Schema":
                    msg = (
                        "JSON Schema support not yet implemented. "
                        "Please use a Pydantic model file (.py) instead."
                    )
                    raise NotImplementedError(msg)  # noqa: TRY301

                # Read the Python file and import the model
                with Path(schema_file_path).open() as f:
                    code = f.read()
                model_class = import_pydantic_model(code)

                progress_messages.append("Schema loaded successfully.")

                # Step 2: Process corpus
                progress_messages.append("Processing corpus...")

                if corpus_file_path:
                    # Extract archive to temp directory
                    # Note: We need to keep the temp directory alive during processing
                    # So we'll extract first, then process
                    tmpdir = tempfile.mkdtemp()
                    try:
                        if corpus_file_path.endswith(".zip"):
                            with zipfile.ZipFile(corpus_file_path, "r") as zip_ref:
                                zip_ref.extractall(tmpdir)  # noqa: S202
                        elif corpus_file_path.endswith((".tar", ".tar.gz")):
                            mode = (
                                "r:gz" if corpus_file_path.endswith(".tar.gz") else "r"
                            )
                            with tarfile.open(corpus_file_path, mode) as tar_ref:  # type: ignore[call-overload]
                                tar_ref.extractall(tmpdir)  # noqa: S202
                        else:
                            msg = f"Unsupported archive format: {corpus_file_path}"
                            raise ValueError(msg)

                        corpus_docs = process_corpus(
                            tmpdir,
                            progress_callback=lambda count,
                            total: progress_messages.append(
                                f"Processed {count}/{total} documents"
                            ),
                        )
                    finally:
                        # Clean up temp directory after processing
                        shutil.rmtree(tmpdir, ignore_errors=True)
                elif corpus_dir_path:
                    if not Path(corpus_dir_path).exists():
                        msg = f"Corpus directory does not exist: {corpus_dir_path}"
                        raise ValueError(msg)  # noqa: TRY301
                    corpus_docs = process_corpus(
                        corpus_dir_path,
                        progress_callback=lambda count, total: progress_messages.append(
                            f"Processed {count}/{total} documents"
                        ),
                    )
                else:
                    msg = "Corpus file or directory is required"
                    raise ValueError(msg)  # noqa: TRY301

                progress_messages.append(f"Processed {len(corpus_docs)} documents.")

                # Step 3: Run extraction
                progress_messages.append("Running extraction...")
                extraction_result = run_extraction(model_class, corpus_docs)
                progress_messages.append("Extraction complete.")

                progress_msg = "\n".join(progress_messages)
                return ExtractionWorkflowResult(
                    progress_message=progress_msg,
                    error_message="",
                    extraction_result=extraction_result,
                    error_visibility=gr.update(visible=False),
                )

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
                return ExtractionWorkflowResult(
                    progress_message=progress_msg,
                    error_message=error_msg,
                    extraction_result=None,
                    error_visibility=gr.update(visible=True),
                )

        def unpack_result(
            result: ExtractionWorkflowResult,
        ) -> tuple[str, str, ExtractionResult | None, Any]:
            """Unpack ExtractionWorkflowResult for Gradio outputs.

            Args:
                result: Extraction workflow result

            Returns:
                Tuple unpacked for Gradio outputs
            """
            return (
                result.progress_message,
                result.error_message,
                result.extraction_result,
                result.error_visibility,
            )

        run_button.click(
            fn=lambda *args: unpack_result(run_extraction_workflow(*args)),
            inputs=[
                schema_file,
                schema_type,
                corpus_file,
                corpus_dir,
                null_policy,
                confidence_threshold,
            ],
            outputs=[
                progress_output,
                error_output,
                extraction_result_state,
                error_output,
            ],
        )

    return interface  # type: ignore[no-any-return]


def create_review_interface(  # noqa: PLR0915, C901
    extraction_result: ExtractionResult,
    extraction_result_state: gr.State,
) -> gr.Blocks:
    """Create Gradio interface for reviewing and resolving candidates.

    Args:
        extraction_result: Extraction results to review
        extraction_result_state: State component to store extraction result

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

        field_components: dict[str, Any] = {}

        for field_result in extraction_result.results:
            with gr.Accordion(
                label=f"Field: {field_result.field_name}",
                open=not field_result.consensus,
            ):
                # Show consensus status
                if field_result.consensus:
                    gr.Markdown(
                        f"✅ **Consensus detected**: "
                        f"{field_result.consensus.value} "
                        f"(confidence: {field_result.consensus.confidence:.2f})"
                    )
                else:
                    gr.Markdown("⚠️ **No consensus** - manual selection required")

                # Show candidates
                if field_result.candidates:
                    gr.Markdown("### Candidates")
                    # Store candidate choices with indices
                    candidate_choices = [
                        f"{i}: {c.value} (confidence: {c.confidence:.2f})"
                        for i, c in enumerate(field_result.candidates)
                    ]
                    candidate_radio = gr.Radio(
                        choices=candidate_choices,
                        label="Select Candidate",
                        value=(
                            f"{field_result.candidates.index(field_result.consensus)}: "
                            f"{field_result.consensus.value} "
                            f"(confidence: {field_result.consensus.confidence:.2f})"
                            if field_result.consensus
                            else None
                        ),
                    )
                    field_components[f"{field_result.field_name}_candidate"] = (
                        candidate_radio
                    )

                    # Show source context button
                    source_context_output = gr.Markdown(
                        label="Source Context", visible=False
                    )

                    def make_show_source_fn(
                        candidates: list[Candidate],
                    ) -> Callable[[str], tuple[str, Any]]:
                        """Create a function to show source context.

                        Args:
                            candidates: List of candidates for this field

                        Returns:
                            Function to show source context
                        """

                        def show_source(selected: str) -> tuple[str, Any]:
                            """Show source context for selected candidate.

                            Args:
                                selected: Selected candidate string

                            Returns:
                                Tuple of (source_context_markdown, visibility_update)
                            """
                            if not selected:
                                return "", gr.update(visible=False)

                            # Extract candidate index from selection string
                            try:
                                idx_str = selected.split(":")[0]
                                idx = int(idx_str)
                                if 0 <= idx < len(candidates):
                                    candidate = candidates[idx]
                                    context = show_source_context(candidate.sources)
                                    return context, gr.update(visible=True)
                            except (ValueError, IndexError):
                                pass

                            return "", gr.update(visible=False)

                        return show_source

                    show_source_btn = gr.Button("View Source Context")
                    show_source_btn.click(
                        fn=make_show_source_fn(field_result.candidates),
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

        # Save button
        save_button = gr.Button("Save Record", variant="primary")
        save_status = gr.Textbox(label="Save Status", interactive=False)

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
                resolutions: list[Resolution] = []
                resolved: dict[str, object] = {}
                provenance: dict[str, list[SourceRef]] = {}

                # Parse field values
                field_idx = 0
                for field_result in extraction_result.results:
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

                    # Determine which value to use
                    chosen_value: object | None = None
                    source_doc_id: str | None = None
                    source_location: str | None = None
                    custom_input = False
                    field_provenance: list[SourceRef] = []

                    if custom_value and custom_value.strip():
                        # Use custom value
                        chosen_value = custom_value.strip()
                        custom_input = True
                    elif candidate_value:
                        # Extract candidate index and find the candidate
                        try:
                            idx_str = candidate_value.split(":")[0]
                            idx = int(idx_str)
                            if 0 <= idx < len(field_result.candidates):
                                candidate = field_result.candidates[idx]
                                chosen_value = candidate.value
                                field_provenance = candidate.sources
                                if candidate.sources:
                                    source_doc_id = candidate.sources[0].doc_id
                                    source_location = candidate.sources[0].location
                        except (ValueError, IndexError):
                            pass

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

                # Create PersistedRecord
                record_id = slugify(
                    f"record-{extraction_result.run_id}-{datetime.now(UTC).isoformat()}"
                )
                record = PersistedRecord(
                    record_id=record_id,
                    resolved=resolved,
                    provenance=provenance,
                    audit={
                        "run_id": extraction_result.run_id,
                        "app_version": "0.0.0",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "user": None,
                        "config": {},
                        "schema_version": extraction_result.schema_version,
                    },
                )

                # Save record
                save_record(record)
            except Exception as e:
                logger.exception("Failed to save record")
                return f"Error saving record: {e}"
            else:
                return f"Record saved successfully! Record ID: {record_id}"

        # Store extraction result in a state component for the save function
        extraction_result_state_review = gr.State()

        def update_extraction_state(
            extraction_result: ExtractionResult | None,
        ) -> ExtractionResult | None:
            """Update the extraction result state for the review interface.

            Args:
                extraction_result: Extraction result to store

            Returns:
                The extraction result
            """
            return extraction_result

        # Update state when extraction completes
        extraction_result_state.change(
            fn=update_extraction_state,
            inputs=[extraction_result_state],
            outputs=[extraction_result_state_review],
        )

        save_button.click(
            fn=save_resolved_record,
            inputs=[extraction_result_state_review, *list(field_components.values())],
            outputs=[save_status],
        )

    return interface  # type: ignore[no-any-return]
