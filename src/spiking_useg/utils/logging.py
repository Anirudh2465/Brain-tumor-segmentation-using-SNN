"""
logging.py — Logging configuration for training runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    run_name: str = "train",
) -> logging.Logger:
    """Configure root logger with console and optional file handlers.

    Args:
        log_dir: If provided, also write logs to {log_dir}/{run_name}.log.
        level: Logging level (default INFO).
        run_name: Name prefix for the log file.

    Returns:
        Configured root logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{run_name}.log", mode="a")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger()
