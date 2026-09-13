"""
Tampering attack (simulated).

Modifies the payload bytes of EXISTING legitimate frames within a time
window, so the decoded signal (e.g. battery SOC/voltage) becomes an
out-of-range or physically implausible value. Unlike spoofing/injection,
no new frames are added -- existing frames are corrupted in place.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.attacks.base import Attack, AttackEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TamperingAttack(Attack):
    """Corrupt payload bytes of existing frames in a window."""

    attack_type = "tampering"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        target_name = self.params["target_message"]
        start_s = float(self.params["start_s"])
        end_s = float(self.params["end_s"])
        fraction = float(self.params["fraction"])

        seed = int(self.config["simulation"]["random_seed"])
        rng = np.random.default_rng(seed + 13)

        trace = trace.copy()

        candidates = trace.index[
            (trace["message_name"] == target_name)
            & (trace["timestamp"] >= start_s)
            & (trace["timestamp"] <= end_s)
            & (trace["attack_type"] == "none")
        ].to_numpy()

        if len(candidates) == 0:
            logger.warning("Tampering: no frames to corrupt for %s", target_name)
            return trace, []

        n_corrupt = max(1, int(len(candidates) * fraction))
        chosen = rng.choice(candidates, size=n_corrupt, replace=False)

        asset = str(trace.loc[chosen[0], "asset"])
        arb_id = int(trace.loc[chosen[0], "arbitration_id"])

        for idx in chosen:
            # Corrupt two random payload bytes with extreme values.
            for byte_pos in rng.choice(range(8), size=2, replace=False):
                trace.at[idx, f"b{byte_pos}"] = int(rng.choice([0x00, 0xFF]))
            trace.at[idx, "label"] = "attack"
            trace.at[idx, "attack_type"] = self.attack_type

        event = AttackEvent(
            attack_type=self.attack_type,
            start_s=start_s,
            end_s=end_s,
            target_asset=asset,
            arbitration_id=arb_id,
            frames_affected=int(n_corrupt),
            description=(
                f"Tampered {n_corrupt} {target_name} frames "
                f"(corrupted payload bytes -> implausible battery values)."
            ),
        )
        logger.info(event.description)
        return trace, [event]