"""
Runtime tamper detection via plausibility / range checks.

Complements HMAC by catching values that are cryptographically valid but
physically impossible (e.g. speed 200 km/h when limit is 160, SOC > 100,
grid frequency 47 Hz). This mirrors real ECU input validation and is a
concrete response to spoofing/tampering/charging attacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlausibilityResult:
    """Outcome of a plausibility check."""

    ok: bool
    signal: str
    value: float
    reason: str


class TamperDetector:
    """Range-checks decoded signal values against configured limits."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.limits: Dict[str, Dict[str, float]] = {
            sig: {"min": float(rng["min"]), "max": float(rng["max"])}
            for sig, rng in self.config["security"]["plausibility"].items()
        }

    def check_signal(self, signal: str, value: float) -> PlausibilityResult:
        """Check one signal value against its plausible range.

        Signals without a configured range are accepted (unknown => pass).
        """
        rng = self.limits.get(signal)
        if rng is None:
            return PlausibilityResult(True, signal, value, "no range configured")
        if value < rng["min"] or value > rng["max"]:
            return PlausibilityResult(
                False, signal, value,
                f"{value:.3f} outside [{rng['min']}, {rng['max']}]",
            )
        return PlausibilityResult(True, signal, value, "within range")

    def check_signals(self, values: Dict[str, float]) -> Dict[str, PlausibilityResult]:
        """Check a dict of signal->value; returns per-signal results."""
        return {sig: self.check_signal(sig, val) for sig, val in values.items()}

    def any_tampered(self, values: Dict[str, float]) -> bool:
        """Return True if any provided signal is out of range."""
        return any(not r.ok for r in self.check_signals(values).values())