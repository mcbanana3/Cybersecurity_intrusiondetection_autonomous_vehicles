"""
Message injection attack (simulated).

Inserts frames using a ROGUE arbitration ID that never appears in normal
traffic. The presence of an unknown ID is the detectable signature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.attacks.base import Attack, AttackEvent, make_frame, rows_to_frame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InjectionAttack(Attack):
    """Inject frames with an unauthorized/unknown CAN ID."""

    attack_type = "injection"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        start_s = float(self.params["start_s"])
        end_s = float(self.params["end_s"])
        rogue_id = int(self.params["rogue_id"])
        rate_s = float(self.params["inject_rate_ms"]) / 1000.0

        seed = int(self.config["simulation"]["random_seed"])
        rng = np.random.default_rng(seed + 7)  # distinct sub-stream

        new_rows: List[Dict[str, Any]] = []
        t = start_s
        while t <= end_s:
            payload = [int(b) for b in rng.integers(0, 256, size=8)]
            new_rows.append(
                make_frame(
                    timestamp=t,
                    arbitration_id=rogue_id,
                    message_name="ROGUE_INJECT",
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
            arbitration_id=rogue_id,
            frames_affected=len(new_rows),
            description=(
                f"Injected {len(new_rows)} rogue 0x{rogue_id:03X} frames "
                f"(ID absent from normal traffic)."
            ),
        )
        logger.info(event.description)
        return merged, [event]