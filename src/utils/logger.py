"""
Centralised logging helper.

Every module in the project calls `get_logger(__name__)` so that all
log output shares a consistent format and level. The level can be set
from config.yaml (see logging.level).
"""

from __future__ import annotations

import logging
import sys

# Keep track of whether the root handler was already configured, so we
# don't attach duplicate handlers when many modules request loggers.
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once for the whole application.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(numeric_level)
        root.addHandler(handler)
        _CONFIGURED = True
    else:
        # Allow later calls to adjust the level.
        logging.getLogger().setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    If logging has not been configured yet, configure it with defaults.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)