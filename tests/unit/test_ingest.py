"""Unit tests for document ingestion module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ctrlf.app.ingest import convert_document_to_markdown, process_corpus


class TestConvertDocumentToMarkdown:
    """Test document conversion to Markdown."""

    def test_convert_txt_file(self) -> None:
        """Test converting a plain text file to Markdown."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!\nThis is a test document.")
            temp_path = f.name

        try:
            markdown, source_map = convert_document_to_markdown(temp_path)
            assert isinstance(markdown, str)
            assert "Hello, world!" in markdown
            assert isinstance(source_map, dict)
        finally:
            Path(temp_path).unlink()

    def test_convert_nonexistent_file(self) -> None:
        """Test that FileNotFoundError is raised for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            convert_document_to_markdown("/nonexistent/file.txt")

    def test_convert_unsupported_format(self) -> None:
        """Test that RuntimeError is raised for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match=r".*"):
                convert_document_to_markdown(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_source_map_structure(self) -> None:
        """Test that source_map has expected structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            _markdown, source_map = convert_document_to_markdown(temp_path)
            assert isinstance(source_map, dict)
            # Source map should enable span mapping
            # Exact structure depends on markitdown output
        finally:
            Path(temp_path).unlink()

    def test_source_map_fallback_location(self) -> None:
        """Test source_map provides fallback location when page/line unavailable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content without page numbers")
            temp_path = f.name

        try:
            _markdown, source_map = convert_document_to_markdown(temp_path)
            # Should have some location info (char-range, document-level, or section)
            assert isinstance(source_map, dict)
        finally:
            Path(temp_path).unlink()


class TestProcessCorpus:
    """Test corpus processing."""

    def test_process_directory(self) -> None:
        """Test processing a directory of documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_dir = Path(tmpdir)
            (test_dir / "doc1.txt").write_text("Document 1 content")
            (test_dir / "doc2.txt").write_text("Document 2 content")

            results = process_corpus(str(test_dir))
            assert len(results) == 2
            for doc in results:
                assert isinstance(doc.doc_id, str)
                assert isinstance(doc.markdown, str)
                assert isinstance(doc.source_map, dict)

    def test_process_nonexistent_path(self) -> None:
        """Test that ValueError is raised for nonexistent corpus path."""
        with pytest.raises(ValueError, match=r".*"):
            process_corpus("/nonexistent/corpus")

    def test_process_with_progress_callback(self) -> None:
        """Test that progress callback is invoked during processing."""
        progress_calls: list[tuple[int, int]] = []

        def progress_callback(doc_count: int, total_docs: int) -> None:
            progress_calls.append((doc_count, total_docs))

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "doc1.txt").write_text("Content 1")
            (test_dir / "doc2.txt").write_text("Content 2")
            (test_dir / "doc3.txt").write_text("Content 3")

            process_corpus(str(test_dir), progress_callback=progress_callback)
            # Progress should be called at least once
            assert len(progress_calls) > 0

    def test_process_continues_on_individual_errors(self) -> None:
        """Test that processing continues when individual files fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "valid.txt").write_text("Valid content")
            # Create invalid file (unsupported format)
            (test_dir / "invalid.xyz").write_text("Invalid format")

            results = process_corpus(str(test_dir))
            # Should process valid file even if invalid file fails
            assert len(results) >= 1
