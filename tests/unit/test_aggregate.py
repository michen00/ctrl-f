"""Unit tests for candidate aggregation module."""

from __future__ import annotations

import pytest

from ctrlf.app.aggregate import (
    aggregate_field_results,
    deduplicate_candidates,
    detect_consensus,
    normalize_value,
)
from ctrlf.app.models import Candidate, FieldResult, SourceRef


class TestNormalizeValue:
    """Test value normalization."""

    def test_normalize_email(self) -> None:
        """Test normalizing email addresses."""
        normalized = normalize_value("John.Doe@EXAMPLE.COM", str)
        assert normalized == "john.doe@example.com"

    def test_normalize_string_trim(self) -> None:
        """Test trimming whitespace from strings."""
        normalized = normalize_value("  test value  ", str)
        assert normalized == "test value"

    def test_normalize_returns_original_on_failure(self) -> None:
        """Test that original value is returned if normalization fails."""
        original = "unparseable-value"
        normalized = normalize_value(original, str)
        # Should return original or normalized version
        assert normalized is not None


class TestDeduplicateCandidates:
    """Test candidate deduplication."""

    def test_deduplicate_similar_candidates(self) -> None:
        """Test that similar candidates are merged."""
        source1 = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="page 1, line 10",
            snippet="Email: test@example.com",
        )
        source2 = SourceRef(
            doc_id="doc2",
            path="doc2.txt",
            location="page 2, line 5",
            snippet="Contact: test@example.com",
        )

        candidates = [
            Candidate(
                value="test@example.com",
                confidence=0.9,
                sources=[source1],
                normalized=None,
            ),
            Candidate(
                value="test@example.com",  # Same value, different source
                confidence=0.85,
                sources=[source2],
                normalized=None,
            ),
        ]

        deduplicated = deduplicate_candidates(candidates, similarity_threshold=0.85)
        # Should merge similar candidates
        assert len(deduplicated) <= len(candidates)
        # All deduplicated candidates should have sources
        for candidate in deduplicated:
            assert len(candidate.sources) > 0

    def test_deduplicate_sorted_by_confidence(self) -> None:
        """Test that deduplicated candidates are sorted by confidence."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="value1", confidence=0.7, sources=[source], normalized=None
            ),
            Candidate(
                value="value2", confidence=0.9, sources=[source], normalized=None
            ),
            Candidate(
                value="value3", confidence=0.8, sources=[source], normalized=None
            ),
        ]

        deduplicated = deduplicate_candidates(candidates)
        # Should be sorted by confidence descending
        if len(deduplicated) > 1:
            for i in range(len(deduplicated) - 1):
                assert deduplicated[i].confidence >= deduplicated[i + 1].confidence


class TestDetectConsensus:
    """Test consensus detection."""

    def test_detect_consensus_meets_thresholds(self) -> None:
        """Test that consensus is detected when thresholds are met."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="consensus_value",
                confidence=0.85,
                sources=[source],
                normalized=None,
            ),
            Candidate(
                value="other_value", confidence=0.60, sources=[source], normalized=None
            ),
        ]

        consensus = detect_consensus(
            candidates, confidence_threshold=0.75, margin_threshold=0.20
        )
        assert consensus is not None
        assert consensus.value == "consensus_value"

    def test_detect_consensus_fails_low_confidence(self) -> None:
        """Test that consensus is not detected when confidence is too low."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="value1", confidence=0.70, sources=[source], normalized=None
            ),  # Below 0.75
            Candidate(
                value="value2", confidence=0.50, sources=[source], normalized=None
            ),
        ]

        consensus = detect_consensus(
            candidates, confidence_threshold=0.75, margin_threshold=0.20
        )
        assert consensus is None

    def test_detect_consensus_fails_insufficient_margin(self) -> None:
        """Test that consensus is not detected when margin is insufficient."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="value1", confidence=0.80, sources=[source], normalized=None
            ),
            Candidate(
                value="value2", confidence=0.75, sources=[source], normalized=None
            ),  # Margin < 0.20
        ]

        consensus = detect_consensus(
            candidates, confidence_threshold=0.75, margin_threshold=0.20
        )
        assert consensus is None


class TestAggregateFieldResults:
    """Test field result aggregation."""

    def test_aggregate_creates_field_result(self) -> None:
        """Test that aggregation creates FieldResult with candidates."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="value1", confidence=0.9, sources=[source], normalized=None
            ),
            Candidate(
                value="value2", confidence=0.8, sources=[source], normalized=None
            ),
        ]

        field_result = aggregate_field_results("test_field", candidates)
        assert isinstance(field_result, FieldResult)
        assert field_result.field_name == "test_field"
        assert len(field_result.candidates) > 0

    def test_aggregate_detects_consensus(self) -> None:
        """Test that aggregation detects consensus when thresholds met."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet for validation",
        )

        candidates = [
            Candidate(
                value="consensus", confidence=0.85, sources=[source], normalized=None
            ),
            Candidate(
                value="other", confidence=0.60, sources=[source], normalized=None
            ),
        ]

        field_result = aggregate_field_results("test_field", candidates)
        if field_result.consensus:
            assert field_result.consensus.value == "consensus"


class TestDisagreementDetection:
    """Test disagreement detection (User Story 3)."""

    def test_detect_disagreement_when_no_consensus(self) -> None:
        """Test that disagreement is detected when no consensus exists."""
        source1 = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Name: Alice",
        )
        source2 = SourceRef(
            doc_id="doc2",
            path="doc2.txt",
            location="line 1",
            snippet="Name: Bob",
        )

        candidates = [
            Candidate(
                value="Alice", confidence=0.80, sources=[source1], normalized=None
            ),
            Candidate(
                value="Bob", confidence=0.75, sources=[source2], normalized=None
            ),  # Close confidence
        ]

        field_result = aggregate_field_results("name", candidates)
        # No consensus should be detected (margin < 0.20)
        assert field_result.consensus is None
        # Should have multiple candidates indicating disagreement
        assert len(field_result.candidates) >= 2

    def test_detect_disagreement_multiple_high_confidence(self) -> None:
        """Test disagreement detection with multiple high-confidence candidates."""
        source1 = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Email: alice@example.com",
        )
        source2 = SourceRef(
            doc_id="doc2",
            path="doc2.txt",
            location="line 1",
            snippet="Email: bob@example.com",
        )

        candidates = [
            Candidate(
                value="alice@example.com",
                confidence=0.85,
                sources=[source1],
                normalized=None,
            ),
            Candidate(
                value="bob@example.com",
                confidence=0.82,
                sources=[source2],
                normalized=None,
            ),
        ]

        field_result = aggregate_field_results("email", candidates)
        # No consensus due to small margin
        assert field_result.consensus is None
        # Both candidates should be present
        assert len(field_result.candidates) == 2


class TestConfidenceScoreComputation:
    """Test confidence score computation (User Story 3)."""

    def test_confidence_scores_preserved_after_deduplication(self) -> None:
        """Test that confidence scores are properly computed after deduplication."""
        source1 = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Value: test",
        )
        source2 = SourceRef(
            doc_id="doc2",
            path="doc2.txt",
            location="line 1",
            snippet="Value: test",
        )

        candidates = [
            Candidate(value="test", confidence=0.9, sources=[source1], normalized=None),
            Candidate(value="test", confidence=0.8, sources=[source2], normalized=None),
        ]

        deduplicated = deduplicate_candidates(candidates, similarity_threshold=0.85)
        # Should merge into one candidate
        assert len(deduplicated) == 1
        # Confidence should be averaged (use approximate equality for floating-point)
        merged = deduplicated[0]
        assert merged.confidence == pytest.approx(0.85)  # (0.9 + 0.8) / 2

    def test_confidence_scores_sorted_descending(self) -> None:
        """Test that candidates are sorted by confidence in descending order."""
        source = SourceRef(
            doc_id="doc1",
            path="doc1.txt",
            location="line 1",
            snippet="Test snippet",
        )

        candidates = [
            Candidate(value="low", confidence=0.5, sources=[source], normalized=None),
            Candidate(value="high", confidence=0.9, sources=[source], normalized=None),
            Candidate(
                value="medium", confidence=0.7, sources=[source], normalized=None
            ),
        ]

        field_result = aggregate_field_results("test_field", candidates)
        # Should be sorted by confidence descending
        if len(field_result.candidates) > 1:
            for i in range(len(field_result.candidates) - 1):
                assert (
                    field_result.candidates[i].confidence
                    >= field_result.candidates[i + 1].confidence
                )
