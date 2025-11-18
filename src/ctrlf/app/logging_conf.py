"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog  # type: ignore[import-not-found]

__all__ = ("configure_logging", "get_logger")


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with appropriate processors and formatters.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(*args: object, **initial_values: object) -> structlog.BoundLogger:
    """Get a configured structlog logger.

    Args:
        *args: Positional arguments passed to structlog.get_logger
        **initial_values: Initial context values

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(*args, **initial_values)
