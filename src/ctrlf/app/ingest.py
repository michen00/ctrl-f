"""Document ingestion module for converting documents to Markdown."""

from __future__ import annotations

__all__ = "CorpusDocument", "convert_document_to_markdown", "process_corpus"

import hashlib
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from markitdown import MarkItDown  # type: ignore[import-not-found]

from ctrlf.app.logging_conf import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)
md = MarkItDown(enable_plugins=False)


class CorpusDocument(NamedTuple):
    """Represents a processed document from the corpus.

    Attributes:
        doc_id: Stable document identifier
        markdown: Converted Markdown content
        source_map: Source location mapping
    """

    doc_id: str
    markdown: str
    source_map: dict[str, Any]


def _generate_doc_id(file_path: str) -> str:
    """Generate a stable document ID from file path.

    Args:
        file_path: Path to the document file

    Returns:
        Stable document identifier
    """
    # Use file path hash for stable ID (not for security)
    return hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()


def _extract_location_info(
    source_map: dict[str, Any], span_start: int, span_end: int
) -> str:
    """Extract location information from source map.

    Args:
        source_map: Source mapping from markitdown
        span_start: Start position of span in markdown
        span_end: End position of span in markdown

    Returns:
        Location descriptor (page/line or char-range)
    """
    # Try to get page/line info from source_map
    # Fallback to char-range if unavailable
    if "pages" in source_map:
        # Try to find page number for this span
        for page_num, page_info in source_map.get("pages", {}).items():
            if (
                isinstance(page_info, dict)
                and "start" in page_info
                and page_info["start"] <= span_start <= page_info.get("end", span_start)
            ):
                return f"page {page_num}"
    # Fallback to char-range
    return f"char-range [{span_start}:{span_end}]"


def convert_document_to_markdown(
    file_path: str,
) -> tuple[str, dict[str, Any]]:
    """Convert a document to Markdown and return source mapping.

    Uses markitdown to convert any supported file type to markdown.
    Let markitdown handle file type detection and conversion.

    Args:
        file_path: Path to document file

    Returns:
        Tuple of (markdown_content, source_map)

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If conversion fails (corrupted file, encoding error,
            or unsupported format)
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    try:
        # Convert using markitdown - it will handle file type detection
        result = md.convert(str(path))
        markdown_content = result.text_content

        # Build source map from result metadata
        source_map: dict[str, Any] = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }

        # Try to extract page/line info if available
        if hasattr(result, "metadata") and result.metadata:
            source_map.update(result.metadata)

        # Add fallback location info
        if "pages" not in source_map and "lines" not in source_map:
            source_map["location_type"] = "char-range"
    except Exception as e:
        msg = f"Failed to convert document {file_path}: {e}"
        logger.exception(
            "document_conversion_failed", file_path=file_path, error=str(e)
        )
        raise RuntimeError(msg) from e
    else:
        return markdown_content, source_map


def process_corpus(  # noqa: C901, PLR0912
    corpus_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CorpusDocument]:
    """Process entire corpus, converting all documents to Markdown.

    Args:
        corpus_path: Path to corpus (directory, zip, or tar archive)
        progress_callback: Optional callback(doc_count, total_docs) for progress

    Returns:
        List of (doc_id, markdown, source_map) tuples

    Raises:
        ValueError: If corpus_path is invalid
    """
    path = Path(corpus_path)
    if not path.exists():
        msg = f"Corpus path does not exist: {corpus_path}"
        raise ValueError(msg)

    results: list[CorpusDocument] = []
    files_to_process: list[Path] = []
    temp_dir_context: tempfile.TemporaryDirectory[str] | None = None

    # Collect files to process
    if path.is_file():
        # Handle zip/tar archives
        if path.suffix.lower() == ".zip":
            temp_dir_context = tempfile.TemporaryDirectory(prefix="ctrlf_extract_")
            temp_dir = Path(temp_dir_context.name)
            with zipfile.ZipFile(path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    Path(member)
                    # Extract all files - let markitdown handle conversion
                    # Skip directories
                    if not member.endswith("/"):
                        zip_ref.extract(member, temp_dir)
                        extracted_path = temp_dir / member
                        if extracted_path.is_file():
                            files_to_process.append(extracted_path)
        elif path.name.lower().endswith(".tar.gz") or path.suffix.lower() == ".tar":
            # Handle tar and tar.gz archives
            temp_dir_context = tempfile.TemporaryDirectory(prefix="ctrlf_extract_")
            temp_dir = Path(temp_dir_context.name)
            # Determine tar mode based on file extension
            mode = "r:gz" if path.name.lower().endswith(".tar.gz") else "r"
            with tarfile.open(path, mode) as tar_ref:  # type: ignore[call-overload]
                for member in tar_ref.getmembers():
                    if member.isfile():
                        # Extract all files - let markitdown handle conversion
                        tar_ref.extract(member, temp_dir)
                        files_to_process.append(temp_dir / member.name)
        else:
            # Single file (not an archive) - let markitdown handle conversion
            files_to_process.append(path)
    elif path.is_dir():
        # Directory of files (recursive search)
        # Process all files - let markitdown handle what it can convert
        # Common document extensions that markitdown typically supports
        common_extensions = [
            "*.pdf",
            "*.docx",
            "*.doc",
            "*.html",
            "*.htm",
            "*.txt",
            "*.md",
            "*.xlsx",
            "*.xls",
            "*.pptx",
            "*.ppt",
            "*.rtf",
            "*.odt",
            "*.ods",
            "*.csv",
            "*.tsv",
        ]
        for ext in common_extensions:
            files_to_process.extend(path.rglob(ext))

    total_files = len(files_to_process)
    logger.info("corpus_processing_started", total_files=total_files)

    try:
        # Process each file
        for idx, file_path in enumerate(files_to_process, 1):
            try:
                markdown, source_map = convert_document_to_markdown(str(file_path))
                doc_id = _generate_doc_id(str(file_path))
                results.append(
                    CorpusDocument(
                        doc_id=doc_id, markdown=markdown, source_map=source_map
                    )
                )

                if progress_callback:
                    progress_callback(idx, total_files)

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "file_processing_failed",
                    file_path=str(file_path),
                    error=str(e),
                )
                # Continue processing other files

        logger.info(
            "corpus_processing_completed", processed=len(results), total=total_files
        )
        return results
    finally:
        # Clean up temporary directory if it was created
        if temp_dir_context is not None:
            temp_dir_context.cleanup()
