"""Candidate aggregation module for deduplication and consensus detection."""

from __future__ import annotations

__all__ = (
    "aggregate_field_results",
    "deduplicate_candidates",
    "detect_consensus",
    "normalize_value",
)

from thefuzz import fuzz  # type: ignore[import-not-found]

from ctrlf.app.logging_conf import get_logger
from ctrlf.app.models import Candidate, FieldResult

logger = get_logger(__name__)


def normalize_value(value: object, field_type: type) -> object:
    """Normalize a candidate value based on field type.

    Args:
        value: Raw candidate value
        field_type: Schema field type

    Returns:
        Normalized value (e.g., lowercase email, ISO date, trimmed string)
    """
    if field_type is str:
        if isinstance(value, str):
            # Trim whitespace
            normalized = value.strip()
            # Normalize email addresses
            if "@" in normalized and "." in normalized.split("@")[1]:
                normalized = normalized.lower()
            return normalized
        return str(value).strip()

    # For other types, return as-is (can be extended)
    return value


def deduplicate_candidates(
    candidates: list[Candidate],
    similarity_threshold: float = 0.85,
) -> list[Candidate]:
    """Group near-duplicate candidates using similarity matching.

    Args:
        candidates: List of candidate values
        similarity_threshold: Minimum similarity (0-1) to consider duplicates

    Returns:
        Deduplicated candidates with merged sources and averaged confidence
    """
    if not candidates:
        return []

    # Convert values to strings for comparison
    candidate_strings = [str(c.value) for c in candidates]

    # Group similar candidates
    groups: list[list[int]] = []
    used_indices: set[int] = set()

    for i, _candidate in enumerate(candidates):
        if i in used_indices:
            continue

        # Start a new group
        group = [i]
        used_indices.add(i)

        # Find similar candidates
        for j in range(i + 1, len(candidates)):
            if j in used_indices:
                continue

            similarity = fuzz.ratio(candidate_strings[i], candidate_strings[j]) / 100.0
            if similarity >= similarity_threshold:
                group.append(j)
                used_indices.add(j)

        groups.append(group)

    # Merge candidates in each group
    deduplicated: list[Candidate] = []
    for group_indices in groups:
        group_candidates = [candidates[i] for i in group_indices]

        # Use the candidate with highest confidence as base
        base_candidate = max(group_candidates, key=lambda c: c.confidence)

        # Merge sources from all candidates in group
        merged_sources = []
        for candidate in group_candidates:
            merged_sources.extend(candidate.sources)

        # Average confidence
        avg_confidence = sum(c.confidence for c in group_candidates) / len(
            group_candidates
        )

        # Create merged candidate
        merged_candidate = Candidate(
            value=base_candidate.value,
            normalized=base_candidate.normalized,
            confidence=avg_confidence,
            sources=merged_sources,
        )
        deduplicated.append(merged_candidate)

    # Sort by confidence (descending)
    deduplicated.sort(key=lambda c: c.confidence, reverse=True)

    return deduplicated


def detect_consensus(
    candidates: list[Candidate],
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.20,
) -> Candidate | None:
    """Detect if there's a consensus candidate meeting thresholds.

    Args:
        candidates: List of deduplicated candidates (sorted by confidence)
        confidence_threshold: Minimum confidence for consensus (default 0.75)
        margin_threshold: Minimum margin over next candidate (default 0.20)

    Returns:
        Consensus candidate if thresholds met, None otherwise
    """
    if not candidates:
        return None

    # Candidates should be sorted by confidence (descending)
    top_candidate = candidates[0]

    # Check confidence threshold
    if top_candidate.confidence < confidence_threshold:
        return None

    # Check margin threshold (if there's a second candidate)
    if len(candidates) > 1:
        second_confidence = candidates[1].confidence
        margin = top_candidate.confidence - second_confidence
        if margin < margin_threshold:
            return None

    return top_candidate


def aggregate_field_results(
    field_name: str,
    candidates: list[Candidate],
) -> FieldResult:
    """Aggregate candidates for a field into FieldResult with consensus detection.

    Args:
        field_name: Schema field name
        candidates: All candidates for this field

    Returns:
        Aggregated results with consensus if detected
    """
    # Normalize candidates
    normalized_candidates = []
    for candidate in candidates:
        normalized_value = normalize_value(candidate.value, type(candidate.value))
        normalized_candidate = Candidate(
            value=candidate.value,
            normalized=normalized_value,
            confidence=candidate.confidence,
            sources=candidate.sources,
        )
        normalized_candidates.append(normalized_candidate)

    # Deduplicate
    deduplicated = deduplicate_candidates(normalized_candidates)

    # Detect consensus
    consensus = detect_consensus(deduplicated)

    return FieldResult(
        field_name=field_name,
        candidates=deduplicated,
        consensus=consensus,
    )
