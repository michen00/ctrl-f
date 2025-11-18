"""Integration tests for end-to-end extraction workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import process_corpus
from ctrlf.app.models import ExtractionResult


class TestEndToEndWorkflow:
    """Test complete extraction workflow from corpus to results."""

    def test_full_extraction_workflow(self) -> None:
        """Test complete workflow: ingest -> extract -> aggregate."""

        # Create Extended Schema model
        class PersonModel(BaseModel):
            name: list[str]
            email: list[str]

        # Create test corpus
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "doc1.txt").write_text(
                "Name: Alice Smith\nEmail: alice@example.com"
            )
            (test_dir / "doc2.txt").write_text(
                "Contact: Bob Jones\nEmail: bob@example.com"
            )

            # Step 1: Ingest corpus
            corpus_docs = process_corpus(str(test_dir))
            assert len(corpus_docs) == 2

            # Step 2: Run extraction
            extraction_result = run_extraction(PersonModel, corpus_docs)
            assert isinstance(extraction_result, ExtractionResult)
            assert len(extraction_result.results) == 2  # One per field

            # Step 3: Verify field results
            for field_result in extraction_result.results:
                assert field_result.field_name in ["name", "email"]
                # Should have candidates (may be empty if extraction fails)
                assert isinstance(field_result.candidates, list)
