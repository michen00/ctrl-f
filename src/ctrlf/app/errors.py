"""Error handling utilities for graceful degradation."""

from __future__ import annotations

from typing import Any

__all__ = (
    "DocumentError",
    "ErrorSummary",
    "ExtractionError",
    "SchemaError",
    "StorageError",
    "collect_errors",
)


class ExtractionError(Exception):
    """Base exception for extraction-related errors."""


class SchemaError(Exception):
    """Exception for schema validation and conversion errors."""


class DocumentError(Exception):
    """Exception for document processing errors."""


class StorageError(Exception):
    """Exception for storage/database errors."""


class ErrorSummary:
    """Collects and summarizes errors during processing.

    Attributes:
        errors: List of error entries with document/field names and error types
    """

    def __init__(self) -> None:
        """Initialize empty error summary."""
        self.errors: list[dict[str, Any]] = []

    def add_error(
        self,
        error_type: str,
        item_name: str,
        error_message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add an error to the summary.

        Args:
            error_type: Type of error (e.g., "DocumentError", "ExtractionError")
            item_name: Name of the item that failed (document name, field name, etc.)
            error_message: Error message
            context: Optional additional context
        """
        error_entry: dict[str, Any] = {
            "type": error_type,
            "item": item_name,
            "message": error_message,
        }
        if context:
            error_entry["context"] = context
        self.errors.append(error_entry)

    def has_errors(self) -> bool:
        """Check if any errors were collected.

        Returns:
            True if errors exist, False otherwise
        """
        return len(self.errors) > 0

    def get_summary(self) -> str:
        """Get formatted error summary.

        Returns:
            Formatted string summarizing all errors
        """
        if not self.errors:
            return "No errors occurred."

        summary_lines = [f"Total errors: {len(self.errors)}", ""]
        for error in self.errors:
            summary_lines.append(
                f"- {error['type']}: {error['item']} - {error['message']}"
            )
            if "context" in error:
                summary_lines.append(f"  Context: {error['context']}")

        return "\n".join(summary_lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert error summary to dictionary.

        Returns:
            Dictionary representation of error summary
        """
        return {
            "total_errors": len(self.errors),
            "errors": self.errors,
        }


def collect_errors(
    error_summary: ErrorSummary,
    error_type: str,
    item_name: str,
    exception: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    """Helper function to collect errors into summary.

    Args:
        error_summary: ErrorSummary instance to add error to
        error_type: Type of error
        item_name: Name of the item that failed
        exception: Exception that was raised
        context: Optional additional context
    """
    error_summary.add_error(
        error_type=error_type,
        item_name=item_name,
        error_message=str(exception),
        context=context,
    )
