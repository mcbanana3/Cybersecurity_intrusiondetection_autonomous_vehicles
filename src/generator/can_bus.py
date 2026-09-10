"""
CAN bus model.

Converts the time-series of physical signals (from Phase 1) into a
message-level CAN trace. Each configured message is transmitted
periodically at its own cycle time (with small timing jitter), and its
payload is encoded from the signal values sampled at that instant.

Output DataFrame (one row per CAN frame):
    timestamp        - time of the frame (seconds)
    arbitration_id   - CAN ID (integer)
    id_hex           - CAN ID as hex string (readability)
    message_name     - e.g. "PWT_Speed"
    ecu              - sending ECU
    asset            - logical asset (for TARA)
    dlc              - data length code (8 here)
    b0..b7           - the eight payload bytes (0-255)
    label            - "normal" (attacks added in Phase 3)
    attack_type      - "none"  (attacks added in Phase 3)

This is the clean baseline trace the attack engine will manipulate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.generator.can_codec import (
    MessageSpec,
    PAYLOAD_LENGTH,
    encode_message,
    load_message_specs,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Canonical CAN trace columns. Later phases import this.
CAN_COLUMNS = (
    ["timestamp", "arbitration_id", "id_hex", "message_name", "ecu", "asset", "dlc"]
    + [f"b{i}" for i in range(PAYLOAD_LENGTH)]
    + ["label", "attack_type"]
)


class CANBus:
    """Generates a CAN trace from a physical-signal DataFrame."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the bus model.

        Args:
            config: Optional pre-loaded config. Loaded from disk if None.
        """
        self.config = config or load_config()
        self.specs: List[MessageSpec] = load_message_specs(self.config)
        self.jitter = float(self.config["can"]["jitter_fraction"])
        seed = int(self.config["simulation"]["random_seed"])
        self.rng = np.random.default_rng(seed)
        logger.info("CANBus ready with %d message types", len(self.specs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_trace(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Produce a CAN message trace from physical signals.

        Args:
            signals: DataFrame from Phase 1 (must include a 'timestamp'
                column plus the signal columns referenced in config).

        Returns:
            A CAN trace DataFrame with columns :data:`CAN_COLUMNS`,
            sorted by timestamp.
        """
        if "timestamp" not in signals.columns:
            raise ValueError("signals DataFrame must contain 'timestamp'")

        t_start = float(signals["timestamp"].iloc[0])
        t_end = float(signals["timestamp"].iloc[-1])

        # Fast lookup: for any time, find the nearest signal sample.
        sig_times = signals["timestamp"].to_numpy()

        rows: List[list] = []
        for spec in self.specs:
            rows.extend(self._emit_message(spec, signals, sig_times, t_start, t_end))

        df = pd.DataFrame(rows, columns=CAN_COLUMNS)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Generated CAN trace with %d frames", len(df))
        return df

    def save(self, df: pd.DataFrame, filename: str = "normal_can.csv") -> str:
        """Save a CAN trace to the configured output directory."""
        import os

        out_dir = self.config["simulation"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        logger.info("Saved %d CAN frames to %s", len(df), path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit_message(
        self,
        spec: MessageSpec,
        signals: pd.DataFrame,
        sig_times: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> List[list]:
        """Emit all frames for one message type across the timeline."""
        cycle_s = spec.cycle_ms / 1000.0
        rows: List[list] = []

        t = t_start
        while t <= t_end:
            # Nearest signal sample to this transmit time.
            idx = int(np.searchsorted(sig_times, t))
            idx = min(idx, len(sig_times) - 1)
            sample = signals.iloc[idx]

            values = {sig.name: float(sample[sig.name]) for sig in spec.signals}
            payload = encode_message(spec, values)

            rows.append(
                [
                    round(t, 5),
                    spec.arbitration_id,
                    f"0x{spec.arbitration_id:03X}",
                    spec.name,
                    spec.ecu,
                    spec.asset,
                    PAYLOAD_LENGTH,
                    *payload,
                    "normal",
                    "none",
                ]
            )

            # Advance by one cycle with multiplicative jitter.
            jitter = 1.0 + self.rng.normal(0.0, self.jitter)
            jitter = max(0.5, jitter)  # never collapse the period
            t += cycle_s * jitter

        return rows