"""
Spoofing attack (simulated).

Injects extra frames that use a LEGITIMATE arbitration ID but carry a
FORGED payload (e.g. an implausible speed). Because the ID is valid but
the decoded value and frame timing are abnormal, this is detectable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.attacks.base import Attack, AttackEvent, make_frame, rows_to_frame
from src.generator.can_codec import encode_message, load_message_specs
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpoofingAttack(Attack):
    """Inject forged frames on a legitimate message ID."""

    attack_type = "spoofing"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        target_name = self.params["target_message"]
        start_s = float(self.params["start_s"])
        end_s = float(self.params["end_s"])
        rate_s = float(self.params["inject_rate_ms"]) / 1000.0
        forged_speed = float(self.params["forged_speed_kmh"])

        specs = {s.name: s for s in load_message_specs(self.config)}
        spec = specs[target_name]

        # Build a forged payload: correct message structure, fake speed.
        forged_values: Dict[str, float] = {s.name: 0.0 for s in spec.signals}
        if "speed_kmh" in forged_values:
            forged_values["speed_kmh"] = forged_speed

        payload = encode_message(spec, forged_values)

        new_rows: List[Dict[str, Any]] = []
        t = start_s
        while t <= end_s:
            new_rows.append(
                make_frame(
                    timestamp=t,
                    arbitration_id=spec.arbitration_id,
                    message_name=spec.name,
                    ecu="ATTACKER",
                    asset=spec.asset,
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
            target_asset=spec.asset,
            arbitration_id=spec.arbitration_id,
            frames_affected=len(new_rows),
            description=(
                f"Injected {len(new_rows)} spoofed {target_name} frames with "
                f"forged speed={forged_speed} km/h."
            ),
        )
        logger.info(event.description)
        return merged, [event]