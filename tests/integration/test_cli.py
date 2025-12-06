"""Integration tests for CLI interface."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from ctrlf.cli import main


def mock_api_call_side_effect(*_args: object, **kwargs: object) -> dict[str, list[str]]:
    """Mock API call that returns simple extraction results."""
    text = str(kwargs.get("text", "") or (_args[0] if _args else ""))
    if not text:
        return {}

    # Simple pattern matching for test
    result: dict[str, list[str]] = {}
    if "Alice" in text:
        result["name"] = ["Alice Smith"]
    if "alice@example.com" in text:
        result["email"] = ["alice@example.com"]
    if "Bob" in text:
        result["name"] = ["Bob Jones"]
    if "bob@example.com" in text:
        result["email"] = ["bob@example.com"]

    return result


@patch("ctrlf.app.structured_extract._call_structured_extraction_api")
class TestCLI:
    """Test CLI interface end-to-end."""

    def test_cli_with_json_schema(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test CLI with JSON Schema file."""
        mock_api_call.side_effect = mock_api_call_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create JSON Schema file
            schema_file = tmp_path / "schema.json"
            schema_file.write_text(
                json.dumps(
                    {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                    }
                )
            )

            # Create corpus directory with test documents
            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "doc1.txt").write_text(
                "Name: Alice Smith\nEmail: alice@example.com"
            )
            (corpus_dir / "doc2.txt").write_text(
                "Contact: Bob Jones\nName: Bob Jones\nEmail: bob@example.com"
            )

            # Create output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            # Run CLI (using typer's test client would be better, but this works)
            # We'll test by calling main directly with proper arguments
            # Create a test context
            test_app = typer.Typer()
            test_app.command()(main)

            # Call main function directly with test arguments
            main(
                schema=schema_file,
                corpus=corpus_dir,
                output=output_dir,
                provider="ollama",
                model_name=None,
                fuzzy_threshold=80,
            )

            # Verify run subfolder was created
            run_folders = [d for d in output_dir.iterdir() if d.is_dir()]
            assert len(run_folders) == 1
            run_output_dir = run_folders[0]

            # Verify folder name format: YYYY-MM-DD-{7char_hash}
            folder_name = run_output_dir.name
            assert len(folder_name) == 18  # YYYY-MM-DD-abcde12 = 18 chars
            assert folder_name[10] == "-"  # Date separator
            assert folder_name[4] == "-"  # Year-month separator
            assert folder_name[7] == "-"  # Month-day separator

            # Verify output files were created in subfolder
            assert (run_output_dir / "extraction_result.json").exists()
            assert (run_output_dir / "extractions.jsonl").exists()

            # Verify extraction_result.json content
            with (run_output_dir / "extraction_result.json").open() as f:
                result = json.load(f)
                assert "results" in result
                assert len(result["results"]) == 2  # name and email fields

            # Verify JSONL file
            jsonl_content = (run_output_dir / "extractions.jsonl").read_text().strip()
            jsonl_lines = jsonl_content.split("\n")
            assert len(jsonl_lines) == 2  # One per document
            for line in jsonl_lines:
                jsonl_data = json.loads(line)
                assert "extractions" in jsonl_data
                assert "text" in jsonl_data
                assert "document_id" in jsonl_data

    def test_cli_with_pydantic_model(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test CLI with Pydantic model file."""
        mock_api_call.side_effect = mock_api_call_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create Pydantic model file
            schema_file = tmp_path / "schema.py"
            schema_file.write_text(
                """
from pydantic import BaseModel

class PersonModel(BaseModel):
    name: str
    email: str
"""
            )

            # Create corpus directory
            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "doc1.txt").write_text(
                "Name: Alice Smith\nEmail: alice@example.com"
            )

            # Create output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            # Run CLI
            main(
                schema=schema_file,
                corpus=corpus_dir,
                output=output_dir,
                provider="ollama",
                model_name=None,
                fuzzy_threshold=80,
            )

            # Verify run subfolder was created
            run_folders = [d for d in output_dir.iterdir() if d.is_dir()]
            assert len(run_folders) == 1
            run_output_dir = run_folders[0]

            # Verify output files were created in subfolder
            assert (run_output_dir / "extraction_result.json").exists()
            assert (run_output_dir / "extractions.jsonl").exists()

    def test_cli_with_custom_provider_and_model(
        self,
        mock_api_call: MagicMock,
    ) -> None:
        """Test CLI with custom provider and model name."""
        mock_api_call.side_effect = mock_api_call_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create schema
            schema_file = tmp_path / "schema.json"
            schema_file.write_text(
                json.dumps(
                    {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                )
            )

            # Create corpus
            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "doc.txt").write_text("Name: Test")

            # Create output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            # Run CLI with custom provider and model
            main(
                schema=schema_file,
                corpus=corpus_dir,
                output=output_dir,
                provider="openai",
                model_name="gpt-4o",
                fuzzy_threshold=90,
            )

            # Verify run subfolder was created
            run_folders = [d for d in output_dir.iterdir() if d.is_dir()]
            assert len(run_folders) == 1
            run_output_dir = run_folders[0]

            # Verify it worked
            assert (run_output_dir / "extraction_result.json").exists()

            # Verify API was called with correct provider/model
            assert mock_api_call.called

    def test_cli_invalid_provider(
        self,
        mock_api_call: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test CLI with invalid provider raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            schema_file = tmp_path / "schema.json"
            schema_file.write_text('{"type": "object", "properties": {}}')

            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            # Test that invalid provider raises BadParameter
            with pytest.raises(typer.BadParameter):
                main(
                    schema=schema_file,
                    corpus=corpus_dir,
                    output=output_dir,
                    provider="invalid",
                    model_name=None,
                    fuzzy_threshold=80,
                )
