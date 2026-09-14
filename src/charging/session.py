"""
EV -> Charging Station -> Grid session model (simulated).

Models one DC charging session at 1 Hz:
    * The EV battery SOC rises as current flows in.
    * Charging current tapers as SOC approaches the target (CV phase).
    * Power = voltage * current.
    * The station adds load to the grid; grid frequency deviates from
      nominal in proportion to the load imbalance (simple linear model).

Also provides simulated charging/grid attacks:
    * false-data injection : overwrite grid frequency / current with fake
      values (as if the charging controller were fed forged telemetry).
    * oscillating load      : swing the station load up and down, which
      the grid model turns into oscillating frequency.

All values are synthetic. No real charger, EV, or grid is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHARGING_COLUMNS = [
    "t_s",
    "soc",
    "current_a",
    "voltage_v",
    "power_kw",
    "station_load_mw",
    "grid_frequency_hz",
    "label",
    "attack_type",
]


@dataclass
class ChargingConfig:
    """Resolved charging-session parameters."""

    duration_s: int
    start_soc: float
    target_soc: float
    voltage_v: float
    max_current_a: float
    grid_base_load_mw: float
    grid_sensitivity: float
    grid_nominal_hz: float


class ChargingSession:
    """Generates normal and attacked charging/grid telemetry."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        cs = self.config["charging_session"]
        self.cfg = ChargingConfig(
            duration_s=int(cs["duration_s"]),
            start_soc=float(cs["start_soc"]),
            target_soc=float(cs["target_soc"]),
            voltage_v=float(cs["station_voltage_v"]),
            max_current_a=float(cs["max_current_a"]),
            grid_base_load_mw=float(cs["grid_base_load_mw"]),
            grid_sensitivity=float(cs["grid_sensitivity"]),
            grid_nominal_hz=float(self.config["charging"]["grid_frequency_hz"]["nominal"]),
        )
        self.attack = cs["attack"]
        seed = int(self.config["simulation"]["random_seed"])
        self.rng = np.random.default_rng(seed + 21)

    # ------------------------------------------------------------------
    def generate_normal(self) -> pd.DataFrame:
        """Generate a clean charging session (no attack)."""
        return self._simulate(apply_attack=False)

    def generate_attacked(self) -> pd.DataFrame:
        """Generate a charging session with a simulated charging/grid attack."""
        return self._simulate(apply_attack=True)

    # ------------------------------------------------------------------
    def _simulate(self, apply_attack: bool) -> pd.DataFrame:
        c = self.cfg
        soc = c.start_soc
        rows = []

        a_start = int(self.attack["start_s"])
        a_end = int(self.attack["end_s"])
        false_freq = float(self.attack["false_freq_hz"])
        false_current = float(self.attack["false_current_a"])
        osc_amp = float(self.attack["osc_amplitude_mw"])

        for t in range(c.duration_s):
            attacked = apply_attack and (a_start <= t <= a_end)

            # ---- Normal charging physics ----
            headroom = max(0.0, (c.target_soc - soc) / c.target_soc)
            current = c.max_current_a * (0.35 + 0.65 * headroom)
            current += self.rng.normal(0.0, 1.5)
            current = float(np.clip(current, 0.0, c.max_current_a))

            voltage = c.voltage_v + self.rng.normal(0.0, 0.8)
            power_kw = voltage * current / 1000.0

            # Station load (MW) added to the grid.
            station_load = power_kw / 1000.0  # kW -> MW (single station, tiny)

            # ---- Attack effects ----
            if attacked:
                # False-data injection on current + oscillating load.
                current = false_current
                power_kw = voltage * current / 1000.0
                osc = osc_amp * np.sin((t - a_start) * 1.2)
                grid_load = c.grid_base_load_mw + station_load + osc
            else:
                grid_load = c.grid_base_load_mw + station_load

            # ---- Grid frequency from load imbalance (simple linear model) ----
            imbalance = grid_load - c.grid_base_load_mw
            grid_freq = c.grid_nominal_hz - c.grid_sensitivity * imbalance
            grid_freq += self.rng.normal(0.0, 0.01)

            if attacked:
                # False-data injection also forges the reported frequency.
                grid_freq = false_freq + 0.5 * np.sin((t - a_start) * 1.2)

            # Advance SOC (normal current used for real charge; attacked
            # current is 'reported' but we still advance modestly so the
            # session progresses).
            eff_current = false_current if attacked else current
            soc += (eff_current / c.max_current_a) * 0.8
            soc = float(np.clip(soc, 0.0, 100.0))

            rows.append([
                t,
                round(soc, 3),
                round(current, 3),
                round(voltage, 3),
                round(power_kw, 3),
                round(grid_load, 4),
                round(grid_freq, 4),
                "attack" if attacked else "normal",
                "charging_attack" if attacked else "none",
            ])

        df = pd.DataFrame(rows, columns=CHARGING_COLUMNS)
        logger.info(
            "Charging session generated (%s): %d samples, %d attacked",
            "attacked" if apply_attack else "normal",
            len(df), int((df["label"] == "attack").sum()),
        )
        return df