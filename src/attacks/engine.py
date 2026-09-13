"""
Attack engine orchestrator.

Runs all enabled attacks in sequence over a baseline CAN trace and
returns the attacked trace plus a combined attack log (one AttackEvent
per injected attack). Everything is SIMULATED on synthetic data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.attacks.base import AttackEvent
from src.attacks.charging_attack import ChargingGridAttack
from src.attacks.dos import DoSAttack
from src.attacks.injection import InjectionAttack
from src.attacks.replay import ReplayAttack
from src.attacks.spoofing import SpoofingAttack
from src.attacks.tampering import TamperingAttack
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttackEngine:
    """Applies a configurable set of simulated attacks to a CAN trace."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the engine and build the attack list.

        Args:
            config: Optional pre-loaded config. Loaded from disk if None.
        """
        self.config = config or load_config()
        # Order matters only for reproducibility; each attack is independent.
        self.attacks = [
            SpoofingAttack(self.config),
            ReplayAttack(self.config),
            DoSAttack(self.config),
            InjectionAttack(self.config),
            TamperingAttack(self.config),
            ChargingGridAttack(self.config),
        ]

    def run(
        self, baseline: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AttackEvent]]:
        """Apply all enabled attacks to a copy of the baseline trace.

        Args:
            baseline: The clean CAN trace from Phase 2.

        Returns:
            Tuple of (attacked_trace, events).
        """
        trace = baseline.copy()
        events: List[AttackEvent] = []

        for attack in self.attacks:
            if not attack.enabled:
                logger.info("Attack '%s' disabled - skipping", attack.attack_type)
                continue
            trace, ev = attack.apply(trace)
            events.extend(ev)

        trace = trace.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Attack engine complete: %d total frames, %d attack frames, "
            "%d attack events",
            len(trace),
            int((trace["label"] == "attack").sum()),
            len(events),
        )
        return trace, events

    def save(
        self,
        trace: pd.DataFrame,
        events: List[AttackEvent],
        trace_file: str = "attacked_can.csv",
        log_file: str = "attack_log.csv",
    ) -> Tuple[str, str]:
        """Save the attacked trace and the attack log to the output dir."""
        out_dir = self.config["simulation"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        trace_path = os.path.join(out_dir, trace_file)
        trace.to_csv(trace_path, index=False)

        log_path = os.path.join(out_dir, log_file)
        pd.DataFrame([e.as_dict() for e in events]).to_csv(log_path, index=False)

        logger.info("Saved attacked trace -> %s", trace_path)
        logger.info("Saved attack log     -> %s", log_path)
        return trace_path, log_path