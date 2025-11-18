"""Main server entrypoint for Gradio application."""

from __future__ import annotations

__all__ = ("main",)

from ctrlf.app.logging_conf import configure_logging, get_logger
from ctrlf.app.ui import create_upload_interface

logger = get_logger(__name__)


def main() -> None:
    """Launch Gradio application server."""
    # Configure logging
    configure_logging(level="INFO")

    logger.info("Starting Schema-Grounded Corpus Extractor server")

    # Create and launch the interface
    interface = create_upload_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
