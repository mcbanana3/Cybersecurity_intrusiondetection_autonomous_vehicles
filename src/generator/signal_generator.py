"""
Synthetic AV/EV signal generator.

Produces physically-plausible, time-series vehicle and battery signals
for a single simulated "drive + charge" session. NO hardware, NO real
CAN bus, NO real vehicle -- everything here is generated in software.

Design goals:
    * Signals are correlated (braking lowers speed, driving drains SOC,
      current heats the battery, etc.) so that later attacks produce
      *detectable* deviations rather than noise-on-noise.
    * Output is a tidy pandas DataFrame with one row per time sample.
    * All randomness is seeded for reproducibility.

Output columns:
    timestamp        - seconds since session start (float)
    speed_kmh        - vehicle speed
    acceleration_ms2 - longitudinal acceleration
    steering_deg     - steering wheel angle
    brake_percent    - brake pedal application
    battery_soc      - state of charge (%)
    battery_temp_c   - battery temperature (Celsius)
    pack_voltage_v   - traction pack voltage
    pack_current_a   - pack current (+discharge while driving,
                                     -charge while charging)
    charging_state   - 0 = driving, 1 = charging
    grid_frequency_hz- local grid frequency (meaningful while charging)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# The canonical column order produced by this module. Later phases
# (CAN encoder, feature engineering) import this to stay consistent.
SIGNAL_COLUMNS = [
    "timestamp",
    "speed_kmh",
    "acceleration_ms2",
    "steering_deg",
    "brake_percent",
    "battery_soc",
    "battery_temp_c",
    "pack_voltage_v",
    "pack_current_a",
    "charging_state",
    "grid_frequency_hz",
]


@dataclass
class DrivePhase:
    """A segment of the drive cycle with a target behaviour.

    Attributes:
        name: Human-readable label (e.g. "accelerate").
        fraction: Portion of the *driving* time this phase occupies.
        target_speed: Speed (km/h) the vehicle aims toward in this phase.
    """

    name: str
    fraction: float
    target_speed: float


class SignalGenerator:
    """Generates one synthetic AV/EV drive + charge session.

    The session is: a drive cycle (accelerate -> cruise -> brake, repeated)
    followed by a charging period at the end.

    Example:
        >>> gen = SignalGenerator()
        >>> df = gen.generate()
        >>> df.shape[0] > 0
        True
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the generator.

        Args:
            config: Optional pre-loaded config dict. If omitted, the
                project config is loaded from disk.
        """
        self.config = config or load_config()

        sim = self.config["simulation"]
        self.seed: int = int(sim["random_seed"])
        self.sample_rate: int = int(sim["sample_rate_hz"])
        self.duration: int = int(sim["duration_seconds"])

        self.veh = self.config["vehicle"]
        self.bat = self.config["battery"]
        self.chg = self.config["charging"]

        self.rng = np.random.default_rng(self.seed)

        self.dt = 1.0 / self.sample_rate  # seconds between samples
        self.n_samples = int(self.duration * self.sample_rate)

        logger.info(
            "SignalGenerator ready: %d samples over %ds at %d Hz",
            self.n_samples,
            self.duration,
            self.sample_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> pd.DataFrame:
        """Generate the full session as a DataFrame.

        Returns:
            A DataFrame with columns :data:`SIGNAL_COLUMNS`, one row per
            time sample, labelled implicitly as "normal" behaviour.
        """
        # We spend ~75% of the session driving and ~25% charging.
        n_drive = int(self.n_samples * 0.75)
        n_charge = self.n_samples - n_drive

        speed, accel, steer, brake = self._generate_driving(n_drive)
        soc, temp, voltage, current, charging_state, grid = self._generate_battery(
            n_drive, n_charge, speed
        )

        # During the charging tail, the vehicle is stationary.
        speed = np.concatenate([speed, np.zeros(n_charge)])
        accel = np.concatenate([accel, np.zeros(n_charge)])
        steer = np.concatenate([steer, np.zeros(n_charge)])
        brake = np.concatenate([brake, np.full(n_charge, 100.0)])  # parked, brake held

        timestamps = np.arange(self.n_samples) * self.dt

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "speed_kmh": speed,
                "acceleration_ms2": accel,
                "steering_deg": steer,
                "brake_percent": brake,
                "battery_soc": soc,
                "battery_temp_c": temp,
                "pack_voltage_v": voltage,
                "pack_current_a": current,
                "charging_state": charging_state,
                "grid_frequency_hz": grid,
            },
            columns=SIGNAL_COLUMNS,
        )

        # Final safety clip so nothing escapes configured physical bounds.
        df = self._clip_to_bounds(df)
        logger.info("Generated normal session with shape %s", df.shape)
        return df

    def save(self, df: pd.DataFrame, filename: str = "normal_signals.csv") -> str:
        """Persist a generated DataFrame to the configured output dir.

        Args:
            df: The DataFrame to save.
            filename: Output CSV file name.

        Returns:
            The full path the file was written to.
        """
        import os

        out_dir = self.config["simulation"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
        return path

    # ------------------------------------------------------------------
    # Driving dynamics
    # ------------------------------------------------------------------
    def _generate_driving(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate correlated driving signals for ``n`` samples.

        The vehicle repeatedly accelerates toward a target speed, cruises,
        then brakes -- producing a realistic urban drive cycle.

        Returns:
            Tuple of (speed, acceleration, steering, brake) arrays.
        """
        speed = np.zeros(n)
        accel = np.zeros(n)
        steer = np.zeros(n)
        brake = np.zeros(n)

        v_max = self.veh["speed_kmh"]["max"]
        a_min = self.veh["acceleration_ms2"]["min"]
        a_max = self.veh["acceleration_ms2"]["max"]

        # Define repeating drive phases.
        phases = [
            DrivePhase("accelerate", 0.30, target_speed=0.55 * v_max),
            DrivePhase("cruise", 0.35, target_speed=0.55 * v_max),
            DrivePhase("accelerate2", 0.15, target_speed=0.80 * v_max),
            DrivePhase("brake", 0.20, target_speed=0.0),
        ]

        # Build a per-sample target speed profile by repeating the cycle.
        cycle_len = max(1, n // 2)  # two full cycles across the drive
        target_profile = self._build_target_profile(phases, cycle_len)
        # Tile/trim to exactly n samples.
        reps = int(np.ceil(n / len(target_profile)))
        target_profile = np.tile(target_profile, reps)[:n]

        current_speed = 0.0
        for i in range(n):
            target = target_profile[i]
            # Proportional controller toward the target speed.
            error = target - current_speed
            # Desired acceleration (km/h difference -> m/s^2, roughly).
            desired_accel = np.clip(error * 0.05, a_min, a_max)
            # Add small realistic noise.
            desired_accel += self.rng.normal(0.0, 0.15)
            desired_accel = float(np.clip(desired_accel, a_min, a_max))

            # Integrate acceleration into speed (convert m/s^2 to km/h/step).
            current_speed += desired_accel * self.dt * 3.6
            current_speed = float(np.clip(current_speed, 0.0, v_max))

            speed[i] = current_speed
            accel[i] = desired_accel
            # Brake applies when decelerating hard.
            brake[i] = float(np.clip(-desired_accel * 25.0, 0.0, 100.0)) if desired_accel < 0 else 0.0
            # Steering: gentle sinusoidal wandering plus noise (lane keeping).
            steer[i] = 15.0 * np.sin(i * 0.02) + self.rng.normal(0.0, 2.0)

        return speed, accel, steer, brake

    def _build_target_profile(
        self, phases: list[DrivePhase], cycle_len: int
    ) -> np.ndarray:
        """Expand drive phases into a per-sample target-speed array."""
        profile = []
        for phase in phases:
            count = max(1, int(cycle_len * phase.fraction))
            profile.extend([phase.target_speed] * count)
        return np.array(profile, dtype=float)

    # ------------------------------------------------------------------
    # Battery / charging dynamics
    # ------------------------------------------------------------------
    def _generate_battery(
        self,
        n_drive: int,
        n_charge: int,
        drive_speed: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Generate battery + grid signals for driving then charging.

        Driving drains SOC (current > 0) and warms the pack.
        Charging raises SOC (current < 0) and involves the grid.

        Returns:
            Tuple of (soc, temp, voltage, current, charging_state, grid_freq).
        """
        total = n_drive + n_charge

        soc = np.zeros(total)
        temp = np.zeros(total)
        voltage = np.zeros(total)
        current = np.zeros(total)
        charging_state = np.zeros(total)
        grid = np.zeros(total)

        soc_val = float(self.bat["soc_percent"]["start"])
        temp_val = float(self.bat["temperature_c"]["ambient"])
        nominal_v = float(self.bat["nominal_voltage_v"])
        max_current = float(self.bat["max_current_a"])
        ambient = float(self.bat["temperature_c"]["ambient"])

        grid_nom = float(self.chg["grid_frequency_hz"]["nominal"])
        grid_tol = float(self.chg["grid_frequency_hz"]["tolerance"])

        # ---- Driving phase ----
        for i in range(n_drive):
            # Power demand scales with speed -> discharge current.
            demand = (drive_speed[i] / self.veh["speed_kmh"]["max"])
            cur = demand * max_current * 0.7 + self.rng.normal(0.0, 3.0)
            cur = float(np.clip(cur, 0.0, max_current))

            # Deplete SOC proportional to current.
            soc_val -= (cur / max_current) * 0.02
            soc_val = float(np.clip(soc_val, self.bat["soc_percent"]["min"], 100.0))

            # Temperature rises with current, relaxes toward ambient.
            temp_val += (cur / max_current) * 0.03 - (temp_val - ambient) * 0.001
            temp_val = float(temp_val)

            # Voltage sags slightly under load and with lower SOC.
            v = nominal_v - (cur / max_current) * 8.0 + (soc_val - 85.0) * 0.1
            v += self.rng.normal(0.0, 0.5)

            soc[i] = soc_val
            temp[i] = temp_val
            voltage[i] = v
            current[i] = cur          # positive = discharging
            charging_state[i] = 0.0
            # Grid not actively used while driving; report nominal-ish value.
            grid[i] = grid_nom + self.rng.normal(0.0, grid_tol * 0.1)

        # ---- Charging phase ----
        station_current = max_current * 0.9  # charge near full station rate
        for j in range(n_charge):
            i = n_drive + j
            # Charging current tapers as SOC approaches 100 (CV phase).
            headroom = max(0.0, (100.0 - soc_val) / 100.0)
            cur = station_current * (0.3 + 0.7 * headroom)
            cur = float(np.clip(cur, 0.0, max_current))

            soc_val += (cur / max_current) * 0.05
            soc_val = float(np.clip(soc_val, 0.0, self.bat["soc_percent"]["max"]))

            # Charging warms the pack modestly.
            temp_val += (cur / max_current) * 0.02 - (temp_val - ambient) * 0.001

            v = nominal_v + (cur / max_current) * 6.0 + (soc_val - 85.0) * 0.1
            v += self.rng.normal(0.0, 0.5)

            soc[i] = soc_val
            temp[i] = temp_val
            voltage[i] = v
            current[i] = -cur         # negative = charging (into pack)
            charging_state[i] = 1.0
            # Grid frequency wanders within tolerance while charging.
            grid[i] = grid_nom + self.rng.normal(0.0, grid_tol * 0.3)

        return soc, temp, voltage, current, charging_state, grid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clip_to_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip signals to their configured physical limits."""
        df["speed_kmh"] = df["speed_kmh"].clip(
            self.veh["speed_kmh"]["min"], self.veh["speed_kmh"]["max"]
        )
        df["acceleration_ms2"] = df["acceleration_ms2"].clip(
            self.veh["acceleration_ms2"]["min"], self.veh["acceleration_ms2"]["max"]
        )
        df["steering_deg"] = df["steering_deg"].clip(
            self.veh["steering_deg"]["min"], self.veh["steering_deg"]["max"]
        )
        df["brake_percent"] = df["brake_percent"].clip(
            self.veh["brake_percent"]["min"], self.veh["brake_percent"]["max"]
        )
        df["battery_soc"] = df["battery_soc"].clip(
            self.bat["soc_percent"]["min"], self.bat["soc_percent"]["max"]
        )
        return df