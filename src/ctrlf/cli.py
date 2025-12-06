"""Command-line interface for ctrlf."""

from __future__ import annotations

__all__ = "cli_entry", "main"

import hashlib
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import process_corpus
from ctrlf.app.logging_conf import configure_logging, get_logger
from ctrlf.app.schema_io import extend_schema
from ctrlf.app.structured_extract import (
    run_structured_extraction,
    visualize_extractions,
    write_jsonl,
)
from ctrlf.app.ui import _load_schema, _resolve_and_validate_path

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="ctrl-f",
    help="Schema-Grounded Corpus Extractor - Extract structured data from documents",
    add_completion=False,
)


def _noop_progress_callback(current: int, total: int) -> None:
    """No-op progress callback for CLI."""


def _generate_run_folder_name(extraction_result: object) -> str:
    """Generate run folder name from extraction result.

    Format: YYYY-MM-DD-{7char_hash}
    Hash is generated from the ISO timestamp string.

    Args:
        extraction_result: ExtractionResult with created_at timestamp

    Returns:
        Run folder name string (e.g., "2025-12-04-abcde12")
    """
    # Extract date from ISO timestamp
    created_at = extraction_result.created_at  # type: ignore[attr-defined]
    # Handle Z timezone suffix
    if created_at.endswith("Z"):
        created_at = created_at[:-1] + "+00:00"
    dt = datetime.fromisoformat(created_at)
    date_str = dt.strftime("%Y-%m-%d")

    # Generate short hash from timestamp string
    hash_obj = hashlib.md5(created_at.encode(), usedforsecurity=False)
    hash_hex = hash_obj.hexdigest()
    short_hash = hash_hex[:7]

    return f"{date_str}-{short_hash}"


@app.command()
def main(  # noqa: PLR0913, PLR0915
    schema: Path = typer.Option(  # noqa: B008
        ...,
        "--schema",
        "-s",
        help="Path to schema file (JSON Schema .json or Pydantic model .py)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    corpus: Path = typer.Option(  # noqa: B008
        ...,
        "--corpus",
        "-c",
        help="Path to corpus directory, file, or archive (ZIP/TAR/TAR.GZ)",
        exists=True,
    ),
    output: Path = typer.Option(  # noqa: B008
        ...,
        "--output",
        "-o",
        help="Output directory for extraction results",
    ),
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help='API provider: "ollama", "openai", or "gemini"',
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Model name (optional, uses provider default if not specified)",
    ),
    fuzzy_threshold: int = typer.Option(
        80,
        "--fuzzy-threshold",
        "-f",
        help="Fuzzy matching threshold (0-100, default: 80)",
        min=0,
        max=100,
    ),
) -> None:
    """Extract structured data from documents based on a schema.

    This command processes a corpus of documents and extracts structured data
    according to the provided schema. Results are saved to the output directory
    in multiple formats:
    - extraction_result.json: Complete extraction results with candidates
    - extractions.jsonl: Structured extraction output (JSONL format)
    - visualization.html: Interactive HTML visualization (if langextract available)
    """
    # Configure logging
    configure_logging(level="INFO")

    # Validate provider
    if provider not in ("ollama", "openai", "gemini"):
        msg = f'Invalid provider: {provider}. Must be "ollama", "openai", or "gemini"'
        raise typer.BadParameter(msg)

    # Strip model_name if provided
    model_name_clean = model_name.strip() if model_name and model_name.strip() else None

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir  # Will be updated after extraction

    console.print("[bold green]Schema-Grounded Corpus Extractor[/bold green]")
    console.print(f"Schema: {schema}")
    console.print(f"Corpus: {corpus}")
    console.print(f"Output: {output_dir}")
    console.print(f"Provider: {provider}")
    if model_name_clean:
        console.print(f"Model: {model_name_clean}")
    console.print(f"Fuzzy threshold: {fuzzy_threshold}")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Load schema
            task1 = progress.add_task("Loading schema...", total=None)
            schema_path_str = str(schema.resolve())
            raw_model_class = _load_schema(schema_path_str)
            model_class = extend_schema(raw_model_class)
            progress.update(
                task1, description="[green]Schema loaded successfully[/green]"
            )
            console.print(
                f"[green]✓[/green] Schema loaded: "
                f"{len(model_class.model_fields)} fields"
            )

            # Step 2: Process corpus
            task2 = progress.add_task("Processing corpus...", total=None)
            corpus_path_str = _resolve_and_validate_path(str(corpus))
            corpus_docs = process_corpus(
                corpus_path_str,
                progress_callback=_noop_progress_callback,
            )
            progress.update(
                task2, description="[green]Corpus processed successfully[/green]"
            )
            console.print(f"[green]✓[/green] Processed {len(corpus_docs)} document(s)")

            # Step 3: Run extraction
            task3 = progress.add_task("Running extraction...", total=None)
            extraction_result, _instrumentation = run_extraction(
                model_class,
                corpus_docs,
                provider=provider,
                model_name=model_name_clean,
                fuzzy_threshold=fuzzy_threshold,
            )
            progress.update(task3, description="[green]Extraction completed[/green]")
            console.print("[green]✓[/green] Extraction completed")

            # Step 4: Create run-specific output subfolder
            run_folder_name = _generate_run_folder_name(extraction_result)
            run_output_dir = output_dir / run_folder_name
            run_output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓[/green] Created run folder: {run_folder_name}")

            # Step 5: Save results
            task4 = progress.add_task("Saving results...", total=None)

            # Save extraction result as JSON
            extraction_result_path = run_output_dir / "extraction_result.json"
            with extraction_result_path.open("w", encoding="utf-8") as f:
                json.dump(extraction_result.model_dump(mode="json"), f, indent=2)
            console.print(
                f"[green]✓[/green] Saved extraction result: {extraction_result_path}"
            )

            # Run structured extraction and save JSONL
            jsonl_lines = run_structured_extraction(
                raw_model_class,  # Use non-extended schema for JSONL
                corpus_docs,
                provider=provider,
                model=model_name_clean,
                fuzzy_threshold=fuzzy_threshold,
            )
            jsonl_path = run_output_dir / "extractions.jsonl"
            write_jsonl(jsonl_lines, jsonl_path)
            console.print(f"[green]✓[/green] Saved JSONL: {jsonl_path}")

            # Generate visualization if langextract is available
            try:
                html_path = run_output_dir / "visualization.html"
                visualize_extractions(jsonl_path, output_html_path=html_path)
                console.print(f"[green]✓[/green] Saved visualization: {html_path}")
            except ImportError:
                console.print(
                    "[yellow]⚠[/yellow] Visualization skipped "
                    "(langextract not available)"
                )

            progress.update(
                task4, description="[green]Results saved successfully[/green]"
            )

        console.print()
        console.print("[bold green]✓ Extraction completed successfully![/bold green]")
        console.print(f"Results saved to: {run_output_dir}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Abort from None
    except Exception as e:
        logger.exception("Extraction failed")
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


def cli_entry() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli_entry()
