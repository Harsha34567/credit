"""Logging utilities for the credit scoring project."""

import logging
from pathlib import Path


def initialize_logger(log_file: Path) -> logging.Logger:
    """Initialize a logger that writes to both console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("credit_scoring")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
