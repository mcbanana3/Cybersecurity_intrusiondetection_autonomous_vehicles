"""
Denial-of-Service / jamming attack (simulated).

Floods the bus with very frequent, highest-priority (lowest ID) frames.
On a real CAN bus, low IDs win arbitration and can starve other traffic.
Here the detectable signature is an enormous frame-rate spike and
near-zero inter-arrival times during the attack window.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.attacks.base import Attack, AttackEvent, make_frame, rows_to_frame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DoSAttack(Attack):
    """Flood the bus with high-priority frames."""

    attack_type = "dos"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        start_s = float(self.params["start_s"])
        end_s = float(self.params["end_s"])
        flood_id = int(self.params["flood_id"])
        rate_s = float(self.params["flood_rate_ms"]) / 1000.0

        # A constant, meaningless high-priority payload.
        payload = [0xFF] * 8

        new_rows: List[Dict[str, Any]] = []
        t = start_s
        while t <= end_s:
            new_rows.append(
                make_frame(
                    timestamp=t,
                    arbitration_id=flood_id,
                    message_name="DOS_FLOOD",
                    ecu="ATTACKER",
                    asset="CAN Bus",
                    payload=payload,
                    attack_type=self.attack_type,
                )
            )
            t += rate_s

        if not new_rows:
            return trace, []

        merged = pd.concat([trace, rows_to_frame(new_rows)], ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        event = AttackEvent(
            attack_type=self.attack_type,
            start_s=start_s,
            end_s=end_s,
            target_asset="CAN Bus",
            arbitration_id=flood_id,
            frames_affected=len(new_rows),
            description=(
                f"Flooded bus with {len(new_rows)} high-priority "
                f"0x{flood_id:03X} frames at {rate_s*1000:.1f} ms interval."
            ),
        )
        logger.info(event.description)
        return merged, [event]