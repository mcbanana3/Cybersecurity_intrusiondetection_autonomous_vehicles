"""
Configuration loader.

Loads the YAML config once and exposes it as a plain dictionary.
Using a single loader means every phase reads identical settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default path: <project_root>/config/config.yaml
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)


@lru_cache(maxsize=None)
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load and cache the project configuration.

    Args:
        config_path: Optional explicit path to a YAML config file.
            If omitted, the default ``config/config.yaml`` is used.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or invalid YAML.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        try:
            config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not config:
        raise ValueError(f"Configuration file is empty: {path}")

    logger.debug("Configuration loaded from %s", path)
    return config