"""
Replay attack (simulated).

Captures a window of legitimate frames for a target message and
re-injects those exact payloads later in time. The stale (but
structurally valid) payloads reappearing out of context is the
detectable signature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.attacks.base import Attack, AttackEvent, make_frame, rows_to_frame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReplayAttack(Attack):
    """Re-inject previously captured legitimate frames."""

    attack_type = "replay"

    def apply(
        self, trace: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        if not self.enabled:
            return trace, []

        target_name = self.params["target_message"]
        cap_start = float(self.params["capture_start_s"])
        cap_end = float(self.params["capture_end_s"])
        replay_start = float(self.params["replay_start_s"])
        replay_end = float(self.params["end_s"])

        # Capture legitimate frames of the target message.
        mask = (
            (trace["message_name"] == target_name)
            & (trace["timestamp"] >= cap_start)
            & (trace["timestamp"] <= cap_end)
            & (trace["attack_type"] == "none")
        )
        captured = trace[mask].sort_values("timestamp")
        if captured.empty:
            logger.warning("Replay: no frames captured for %s", target_name)
            return trace, []

        # Re-emit them starting at replay_start, preserving relative timing,
        # looping until replay_end is reached.
        cap_times = captured["timestamp"].to_numpy()
        rel = cap_times - cap_times[0]
        window = replay_end - replay_start
        span = max(rel[-1], 1e-6)

        new_rows: List[Dict[str, Any]] = []
        offset = 0.0
        loops = 0
        while offset <= window:
            for (_, frame), r in zip(captured.iterrows(), rel):
                t = replay_start + offset + r
                if t > replay_end:
                    break
                payload = [int(frame[f"b{i}"]) for i in range(8)]
                new_rows.append(
                    make_frame(
                        timestamp=t,
                        arbitration_id=int(frame["arbitration_id"]),
                        message_name=str(frame["message_name"]),
                        ecu="ATTACKER",
                        asset=str(frame["asset"]),
                        payload=payload,
                        attack_type=self.attack_type,
                    )
                )
            offset += span
            loops += 1
            if loops > 1000:  # safety guard against pathological configs
                break

        if not new_rows:
            return trace, []

        merged = pd.concat([trace, rows_to_frame(new_rows)], ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        asset = str(captured.iloc[0]["asset"])
        arb_id = int(captured.iloc[0]["arbitration_id"])
        event = AttackEvent(
            attack_type=self.attack_type,
            start_s=replay_start,
            end_s=replay_end,
            target_asset=asset,
            arbitration_id=arb_id,
            frames_affected=len(new_rows),
            description=(
                f"Replayed {len(new_rows)} captured {target_name} frames "
                f"(captured {cap_start}-{cap_end}s, replayed "
                f"{replay_start}-{replay_end}s)."
            ),
        )
        logger.info(event.description)
        return merged, [event]