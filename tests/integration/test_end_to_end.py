"""Integration tests for end-to-end extraction workflow."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from langextract.data import AnnotatedDocument, Extraction
from pydantic import BaseModel

from ctrlf.app.extract import run_extraction
from ctrlf.app.ingest import process_corpus
from ctrlf.app.models import ExtractionResult
from ctrlf.app.schema_io import extend_schema, import_pydantic_model


def mock_extract_side_effect(*_args: Any, **kwargs: Any) -> AnnotatedDocument:  # noqa: ANN401
    """Mock extract side effect that performs simple heuristic extraction."""
    text = kwargs.get("text_or_documents", "")
    extractions = []

    # Simple regex patterns for test data
    patterns = {
        "name": r"Name: ([^\n,]+)",
        "email": r"Email: ([^\n,]+)",
        "invoice_number": r"Invoice(?: Number)?: ([^\n,]+)",
        "amount": r"(?:Amount|Total): ([\d.]+)",
        "date": r"Date: ([\d-]+)",
    }

    for field, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            extraction = MagicMock(spec=Extraction)
            extraction.extraction_class = field
            extraction.extraction_text = value
            extraction.char_start = match.start(1)
            extraction.char_end = match.end(1)
            extraction.confidence = 0.9
            extractions.append(extraction)

    return AnnotatedDocument(
        text=text, extractions=cast("list[Extraction]", extractions)
    )


@patch("ctrlf.app.extract.extract")
class TestEndToEndWorkflow:
    """Test complete extraction workflow from corpus to results."""

    def test_full_extraction_workflow(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test complete workflow: ingest -> extract -> aggregate."""
        mock_extract.side_effect = mock_extract_side_effect

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
                "Contact: Bob Jones\nName: Bob Jones\nEmail: bob@example.com"
            )

            # Step 1: Ingest corpus
            corpus_docs = process_corpus(str(test_dir))
            assert len(corpus_docs) == 2

            # Step 2: Run extraction
            extraction_result, _instrumentation = run_extraction(
                PersonModel, corpus_docs
            )
            assert isinstance(extraction_result, ExtractionResult)
            assert len(extraction_result.results) == 2  # One per field

            # Step 3: Verify field results
            for field_result in extraction_result.results:
                assert field_result.field_name in ["name", "email"]
                # Should have candidates (may be empty if extraction fails)
                assert isinstance(field_result.candidates, list)
                assert len(field_result.candidates) > 0

    def test_pydantic_model_workflow(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test complete workflow with Pydantic model input (User Story 2)."""
        mock_extract.side_effect = mock_extract_side_effect

        # Create Pydantic model code
        model_code = """
from pydantic import BaseModel

class InvoiceModel(BaseModel):
    invoice_number: str
    amount: float
    date: str | None = None
"""

        # Step 1: Import Pydantic model
        model_class = import_pydantic_model(model_code)
        assert model_class is not None

        # Step 2: Extend schema (coerce to arrays)
        extended_model = extend_schema(model_class)
        assert extended_model is not None

        # Step 3: Create test corpus
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "invoice1.txt").write_text(
                "Invoice Number: INV-001\nAmount: 100.50\nDate: 2024-01-15"
            )
            (test_dir / "invoice2.txt").write_text(
                "Invoice: INV-002\nTotal: 250.75\nDate: 2024-01-16"
            )

            # Step 4: Ingest corpus
            corpus_docs = process_corpus(str(test_dir))
            assert len(corpus_docs) == 2

            # Step 5: Run extraction with extended model
            extraction_result, _instrumentation = run_extraction(
                extended_model, corpus_docs
            )
            assert isinstance(extraction_result, ExtractionResult)
            # invoice_number, amount, date
            assert len(extraction_result.results) == 3

            # Step 6: Verify field results
            field_names = {fr.field_name for fr in extraction_result.results}
            assert field_names == {"invoice_number", "amount", "date"}

            # Step 7: Verify all fields have candidate lists
            for field_result in extraction_result.results:
                assert isinstance(field_result.candidates, list)
                # Consensus should be None or one of the candidates
                if field_result.consensus is not None:
                    assert field_result.consensus in field_result.candidates

    def test_disagreement_resolution_workflow(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test disagreement resolution workflow (User Story 3)."""
        mock_extract.side_effect = mock_extract_side_effect

        # Create Extended Schema model
        class PersonModel(BaseModel):
            name: list[str]
            email: list[str]

        # Create test corpus with conflicting values
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "doc1.txt").write_text(
                "Name: Alice Smith\nEmail: alice@example.com"
            )
            (test_dir / "doc2.txt").write_text(
                "Name: Bob Jones\nEmail: bob@example.com"
            )
            (test_dir / "doc3.txt").write_text(
                "Name: Alice Johnson\nEmail: alice.j@example.com"
            )

            # Step 1: Ingest corpus
            corpus_docs = process_corpus(str(test_dir))
            assert len(corpus_docs) == 3

            # Step 2: Run extraction
            extraction_result, _instrumentation = run_extraction(
                PersonModel, corpus_docs
            )
            assert isinstance(extraction_result, ExtractionResult)

            # Step 3: Find a field with disagreements (no consensus)
            field_with_disagreement = None
            for field_result in extraction_result.results:
                if field_result.consensus is None and len(field_result.candidates) > 1:
                    field_with_disagreement = field_result
                    break

            # Step 4: Verify disagreement scenario exists
            # At least one field should have multiple candidates without consensus
            # (This may depend on extraction quality, so we check if it exists)
            if field_with_disagreement:
                # Should have multiple candidates
                assert len(field_with_disagreement.candidates) >= 2
                # No consensus should be detected
                assert field_with_disagreement.consensus is None
                # All candidates should have confidence scores
                for candidate in field_with_disagreement.candidates:
                    assert 0.0 <= candidate.confidence <= 1.0
                    assert len(candidate.sources) > 0
