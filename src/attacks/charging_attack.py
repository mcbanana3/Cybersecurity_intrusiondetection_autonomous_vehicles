"""
EV charging / grid attack (simulated).

Corrupts CHG_Status frames during charging so the reported grid
frequency becomes dangerously low and oscillating (false-data injection
against the charging controller / grid interface). Operates only on our
synthetic charging telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.attacks.base import Attack, AttackEvent
from src.generator.can_codec import decode_message, encode_message, load_message_specs
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChargingGridAttack(Attack):
    """Inject false grid-frequency data into charging status frames."""

    attack_type = "charging_attack"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        target_name = self.params["target_message"]
        start_s = float(self.params["start_s"])
        end_s = float(self.params["end_s"])
        false_freq = float(self.params["false_grid_freq_hz"])
        osc = float(self.params["oscillation_hz"])

        specs = {s.name: s for s in load_message_specs(self.config)}
        spec = specs[target_name]

        trace = trace.copy()
        idxs = trace.index[
            (trace["message_name"] == target_name)
            & (trace["timestamp"] >= start_s)
            & (trace["timestamp"] <= end_s)
            & (trace["attack_type"] == "none")
        ].to_numpy()

        if len(idxs) == 0:
            logger.warning("Charging attack: no CHG_Status frames in window")
            return trace, []

        asset = str(trace.loc[idxs[0], "asset"])
        arb_id = int(trace.loc[idxs[0], "arbitration_id"])

        for k, idx in enumerate(idxs):
            payload = [int(trace.loc[idx, f"b{i}"]) for i in range(8)]
            decoded = decode_message(spec, payload)

            # Force charging_state on and inject a low, oscillating frequency.
            decoded["charging_state"] = 1.0
            decoded["grid_frequency_hz"] = false_freq + osc * np.sin(k * 0.6)

            new_payload = encode_message(spec, decoded)
            for i in range(8):
                trace.at[idx, f"b{i}"] = int(new_payload[i])
            trace.at[idx, "label"] = "attack"
            trace.at[idx, "attack_type"] = self.attack_type

        event = AttackEvent(
            attack_type=self.attack_type,
            start_s=start_s,
            end_s=end_s,
            target_asset=asset,
            arbitration_id=arb_id,
            frames_affected=int(len(idxs)),
            description=(
                f"Injected false grid frequency (~{false_freq} Hz, "
                f"osc +/-{osc} Hz) into {len(idxs)} {target_name} frames."
            ),
        )
        logger.info(event.description)
        return trace, [event]