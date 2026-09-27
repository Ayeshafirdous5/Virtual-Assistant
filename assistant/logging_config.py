"""Logging setup for the Virtual Assistant.

Uses Python's standard :mod:`logging` module only. Output goes to two places:

* the console, so the user sees what is happening while the app runs;
* ``logs/assistant.log``, a rotating file that keeps a rolling history
  without growing without bound.

Usage
-----
Call :func:`setup_logging` once, early in ``main``::

    from assistant.config import get_config
    from assistant.logging_config import setup_logging

    logger = setup_logging(get_config())

Any module can then get a namespaced logger with :func:`get_logger`::

    logger = get_logger(__name__)
    logger.info("something happened")

The log file is created only when logging is actually set up, so importing
this module (or running the tests) does not create files as a side effect.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from assistant.config import Config

# Name of the root logger for the whole application.
ROOT_LOGGER_NAME = "assistant"

# Format for the console: short and readable.
CONSOLE_FORMAT = "%(levelname)-8s %(name)-28s %(message)s"

# Format for the file: includes the timestamp, useful for history.
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"

# Marker set on a logger once it has been configured, so calling setup twice
# does not attach duplicate handlers and double every message.
_CONFIGURED_FLAG = "_assistant_logging_configured"


def _build_formatter(fmt: str) -> logging.Formatter:
    """Create a formatter with a readable timestamp."""
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(config: Config) -> logging.Logger:
    """Configure and return the application root logger.

    Creates ``logs/`` if needed and attaches a console handler plus a
    :class:`~logging.handlers.RotatingFileHandler`. Safe to call more than
    once: the second call is a no-op and just returns the logger.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)

    if getattr(root, _CONFIGURED_FLAG, False):
        return root

    # Create the log directory. Exists checks keep this cheap on repeat runs.
    config.log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config.log_level, logging.INFO)
    root.setLevel(level)
    # Log records are handled by our own handlers, not by the root handler.
    root.propagate = False

    # --- Console output --------------------------------------------------
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_build_formatter(CONSOLE_FORMAT))
    root.addHandler(console_handler)

    # --- Rotating file output --------------------------------------------
    try:
        file_handler = RotatingFileHandler(
            filename=config.log_file,
            maxBytes=config.log_file_max_bytes,
            backupCount=config.log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_build_formatter(FILE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        # A read-only disk or a locked file must not stop the assistant from
        # starting; the console handler above is still active.
        root.warning("File logging disabled, could not open log file: %s", exc)

    setattr(root, _CONFIGURED_FLAG, True)
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger nested under the application root.

    ``get_logger(__name__)`` inside ``assistant.tools.weather`` yields a
    logger named ``assistant.tools.weather``, so related messages group
    together in the log file.
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)

    if name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """Flush and close the file handler.

    Useful at the end of the program and in tests, so log records are written
    to disk before the process exits.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except (OSError, ValueError):
            # Already closed, or the stream went away during shutdown.
            pass
        root.removeHandler(handler)
    if hasattr(root, _CONFIGURED_FLAG):
        delattr(root, _CONFIGURED_FLAG)
