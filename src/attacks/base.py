"""
Shared base types and helpers for the attack engine.

All attacks are SIMULATED and operate only on an in-memory pandas
DataFrame representing our synthetic CAN trace. Nothing here interacts
with real hardware, vehicles, or networks.

Every attack implements the `Attack` interface:
    apply(trace) -> (modified_trace, list_of_AttackEvent)

An AttackEvent records what happened so later phases (TARA) can map the
attack to an affected asset automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.generator.can_bus import CAN_COLUMNS
from src.generator.can_codec import PAYLOAD_LENGTH


@dataclass
class AttackEvent:
    """A record of one injected attack, for logging and TARA mapping.

    Attributes:
        attack_type: e.g. "spoofing", "replay".
        start_s: Attack window start (seconds).
        end_s: Attack window end (seconds).
        target_asset: Logical asset affected (maps to TARA).
        arbitration_id: The CAN ID involved.
        frames_affected: Number of frames added or modified.
        description: Human-readable summary.
    """

    attack_type: str
    start_s: float
    end_s: float
    target_asset: str
    arbitration_id: int
    frames_affected: int
    description: str

    def as_dict(self) -> Dict[str, Any]:
        """Return the event as a plain dictionary."""
        return {
            "attack_type": self.attack_type,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "target_asset": self.target_asset,
            "arbitration_id": self.arbitration_id,
            "id_hex": f"0x{self.arbitration_id:03X}",
            "frames_affected": self.frames_affected,
            "description": self.description,
        }


def make_frame(
    timestamp: float,
    arbitration_id: int,
    message_name: str,
    ecu: str,
    asset: str,
    payload: List[int],
    attack_type: str,
) -> Dict[str, Any]:
    """Build a single CAN-trace row (dict) matching CAN_COLUMNS.

    Args:
        timestamp: Frame time in seconds.
        arbitration_id: CAN ID (int).
        message_name: Message label.
        ecu: Sending ECU (or a spoofed name).
        asset: Logical asset.
        payload: List of 8 byte values (0-255).
        attack_type: Attack label for this frame ("none" if legitimate).

    Returns:
        A dict with every key in CAN_COLUMNS.
    """
    if len(payload) != PAYLOAD_LENGTH:
        raise ValueError(f"payload must be {PAYLOAD_LENGTH} bytes")

    row = {
        "timestamp": round(float(timestamp), 5),
        "arbitration_id": int(arbitration_id),
        "id_hex": f"0x{int(arbitration_id):03X}",
        "message_name": message_name,
        "ecu": ecu,
        "asset": asset,
        "dlc": PAYLOAD_LENGTH,
        "label": "attack" if attack_type != "none" else "normal",
        "attack_type": attack_type,
    }
    for i in range(PAYLOAD_LENGTH):
        row[f"b{i}"] = int(payload[i]) & 0xFF
    return row


def rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of row dicts into a DataFrame with CAN_COLUMNS order."""
    return pd.DataFrame(rows, columns=CAN_COLUMNS)


class Attack(ABC):
    """Abstract base class for all simulated attacks."""

    #: Short attack label used in the `attack_type` column.
    attack_type: str = "generic"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store the full project config.

        Args:
            config: The loaded configuration dictionary.
        """
        self.config = config
        self.params: Dict[str, Any] = config["attacks"].get(self.attack_type, {})

    @property
    def enabled(self) -> bool:
        """Whether this attack is switched on in config."""
        return bool(self.params.get("enabled", False))

    @abstractmethod
    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        """Apply the attack to a CAN trace.

        Args:
            trace: The current CAN trace DataFrame.

        Returns:
            A tuple of (new_trace, events). The trace is returned
            (possibly with rows added/modified); events describe what
            was injected.
        """
        raise NotImplementedError