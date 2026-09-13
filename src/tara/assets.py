"""
Asset registry.

Loads the logical assets defined in config.yaml and exposes lookup by
name. Assets are the elements of value whose compromise the TARA engine
scores. Everything is simulated / conceptual (no real hardware).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """One logical asset in the AEV system.

    Attributes:
        name: Unique asset name (matches CAN message 'asset' fields).
        type: Category, e.g. "Control ECU", "Network".
        cia: Primary security property, e.g. "Integrity".
        criticality: Qualitative importance, e.g. "High".
    """

    name: str
    type: str
    cia: str
    criticality: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssetRegistry:
    """Holds all assets and supports lookup by name."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self._assets: Dict[str, Asset] = {}
        for entry in self.config["assets"]:
            asset = Asset(
                name=str(entry["name"]),
                type=str(entry["type"]),
                cia=str(entry["cia"]),
                criticality=str(entry["criticality"]),
            )
            self._assets[asset.name] = asset
        logger.info("AssetRegistry loaded %d assets", len(self._assets))

    def get(self, name: str) -> Asset:
        """Return the asset by name.

        Falls back to a generic 'CAN Bus'-like asset if the name is
        unknown, so the pipeline never crashes on an unexpected label.
        """
        if name in self._assets:
            return self._assets[name]
        logger.warning("Unknown asset '%s' - using generic fallback", name)
        return Asset(name=name, type="Unknown", cia="Integrity",
                     criticality="Medium")

    def all(self) -> List[Asset]:
        """Return all assets as a list."""
        return list(self._assets.values())